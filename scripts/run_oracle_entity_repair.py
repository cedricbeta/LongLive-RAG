#!/usr/bin/env python3
"""Small fixed-target arm C pilot; no automatic locator, placement, or best-of.

prepare makes proposals and SAM2 mask previews. run requires a human review
bound to the packet hash, verifies an unedited replay, then calls the existing
EditRenderBackend once. export makes an anonymous A/B packet for human ratings.
This tests script-appearance repair on deliberately selected first shots, not
cross-shot identity preservation or an autonomous system's success rate.
adopt-machine forks an unrendered packet for an explicitly machine-reviewed
exploratory pilot. It never treats an assistant annotation as human ground truth.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from scripts.calibrate_entity_judge import archived_index
from utils.agent_edit import LATENT_GRID, mask_coverage, read_video_frames, save_mask_payload
from utils.agent_entity_judge import EntityDriftVerifier, Sam2Tracker, scale_box_norm1000
from utils.agent_runtime import Action, Verdict, file_sha256

# These are proposals from visual inspection, NOT human ground truth. All eight
# selected IDs are explicit: no role-matching rule or judge score selects them.
PROPOSALS = (
    ("ns090", "Iris", [0] * 8, "头发偏黑、衣服深色；设定要求红色卷发、苔绿色外套。"),
    ("ns180", "Noah", [0] * 8, "上衣为蓝色；设定要求深绿色工作衬衫。只评价衣服，不以发长为缺陷。"),
    ("ns190", "Iris", [0] * 8, "头发为白色／银色；设定要求红色卷发。"),
    ("ns220", "Clara", [0] * 8, "左侧坐着的 Clara 穿灰色外套；设定要求棕黄色长外套。"),
    ("ns310", "Clara", [0] * 8, "左侧 Clara 穿酒红／紫色外套；设定要求棕黄色长外套。"),
)
PROTOCOL = dict(entry_sigma=1.0, noise_seed=4243, attempts_per_case=1,
                place_absent="never", automatic_judge=False,
                selection="assistant-proposed explicit appearance defects; pending human confirmation",
                scope="five first-shot script-appearance violations; not cross-shot identity drift",
                control="unedited current replay must reproduce archived decoded pixels exactly",
                evaluation="human A/B appearance, action, other subjects, and all later shots; retain every result")
MACHINE_PROTOCOL = {**PROTOCOL,
    "selection": "assistant visual review of fixed proposals; no human confirmation or oracle ground truth",
    "evaluation": "assistant visual inspection; uncertain observations stay uncertain; not human ratings or independent blinded validation"}
CODE_FILES = ("scripts/run_oracle_entity_repair.py", "scripts/run_agent_edit_demo.py",
              "utils/agent_edit.py", "utils/agent_entity_judge.py",
              "pipeline/causal_diffusion_inference.py", "utils/edit_renoise_plans.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"unreadable video: {path}")
    return fps, count


def clip(source, target, start, count):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(source), "-an", "-map_metadata", "-1",
                    "-vf", f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)], check=True)


def mask_preview(frames, track, target, fps):
    width, height = frames[0].size
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
                             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)],
                            stdin=subprocess.PIPE)
    try:
        for frame, mask in zip(frames, track):
            rgb = np.array(frame)
            rgb[mask] = (rgb[mask] * .7 + np.array([0, 255, 255]) * .3).astype(np.uint8)
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"mask preview encoding failed: {target}")


def target_sheet(samples, selected, target):
    sheet = Image.new("RGB", (1280, 404), "#202020")
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    for i, sample in enumerate(samples):
        with Image.open(sample["frame"]) as source:
            im = source.convert("RGB")
        draw = ImageDraw.Draw(im)
        for index, box in enumerate(sample["boxes"]):
            draw.rectangle(box, outline="yellow" if index == selected[i] else "gray", width=5)
            draw.text((box[0] + 4, box[1] + 4), str(index), fill="black", stroke_fill="yellow",
                      stroke_width=2, font=ImageFont.truetype("DejaVuSans.ttf", 28))
        im.thumbnail((320, 176))
        x, y = i % 4 * 320, i // 4 * 202
        sheet.paste(im, (x, y + 26))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), f"chunk {i}: target ID {selected[i]}", fill="white", font=font)
    sheet.save(target)


def prepare(output, device):
    output.mkdir(parents=True, exist_ok=False)
    specs = json.loads((ROOT / "example/narrastream_bench/entity_specs_subset33.json").read_text())
    tracker = Sam2Tracker(device=device)
    cases, indexes = [], {}
    for prefix, entity, selected, defect in PROPOSALS:
        scene, = [s for s in specs if s.startswith(prefix)]
        source_root = ROOT / ("videos/narrastream_constraint_smoke_20260903" if prefix == "ns190"
                              else "videos/narrastream_threearm_20260827/armC_entity")
        source, = source_root.glob(f"*/{scene}")
        video, = (source / "renders").glob("baseline-*.mp4")
        digest = file_sha256(video)
        frames_dir = source / "entity_judge/frames" / f"{digest[:16]}_c00-07"
        case_dir = output / prefix
        case_dir.mkdir()
        shutil.copytree(source / "data" / scene, case_dir / "data" / scene)
        config = case_dir / "baseline.yaml"
        shutil.copy2(source / "configs/baseline.yaml", config)
        cfg = yaml.safe_load(config.read_text())
        if cfg.get("kv_rag", {}).get("enabled", False) or cfg["inference"]["kv_rag"]["enabled"]:
            raise ValueError("pilot requires KV-RAG off")
        durations = [int(x) for x in (case_dir / "data" / scene / "shot_durations.txt").read_text().split()]
        if durations[0] != 8:
            raise ValueError("fixed pilot expects eight chunks in shot zero")
        caption = json.loads((case_dir / "data" / scene / "0.json").read_text())["caption"]
        spec, = [s for s in specs[scene] if s["name"] == entity]
        cache = source.parent / "vlm_cache"
        if cache not in indexes:
            indexes[cache] = archived_index(cache)
        samples = []
        for c, chosen in enumerate(selected):
            frame = frames_dir / f"chunk{c:02d}.png"
            with Image.open(frame) as im:
                width, height = im.size
            record = indexes[cache][file_sha256(frame)]
            boxes = [scale_box_norm1000(s["bbox_2d"], width, height) for s in record["parsed"]["subjects"]]
            if any(b is None for b in boxes) or not 0 <= chosen < len(boxes):
                raise ValueError("invalid archived candidate or proposed ID")
            samples.append(dict(chunk=c, frame=str(frame), boxes=boxes, selected_id=chosen,
                                enumeration_fingerprint=record["cache_fingerprint"]))
        prompts = {entity: {c * 8: sample["boxes"][selected[c]] for c, sample in enumerate(samples)}}
        track = tracker.tracks(frames_dir, prompts, 64, (height, width))[entity]
        # Exact existing arm C SAM2 -> latent mask conversion, not a new mask rule.
        soft = EntityDriftVerifier._soft(SimpleNamespace(grid=LATENT_GRID, fpc=8), track, 8)
        if any(not m.any() for m in soft.values()):
            raise ValueError(f"empty target mask: {scene}")
        mask = save_mask_payload(soft, case_dir / "target_masks.pt", phrase=entity, video=video)
        fps, count = video_info(video)
        if count != sum(durations) * 8:
            raise ValueError("video duration does not match authored shot boundaries")
        frames = read_video_frames(video, 0, 64)
        if len(frames) != 64:
            raise ValueError("incomplete source shot")
        for c, sample in enumerate(samples):
            with Image.open(sample["frame"]) as archived:
                if not np.array_equal(np.array(frames[c * 8]), np.array(archived)):
                    raise ValueError("archived box prompts are not aligned with the baseline")
        target_sheet(samples, selected, case_dir / "targets.png")
        clip(video, case_dir / "source.mp4", 0, 64)
        mask_preview(frames, track, case_dir / "mask_preview.mp4", fps)
        paths = [video, config, mask, *sorted((case_dir / "data" / scene).iterdir()),
                 *(Path(s["frame"]) for s in samples), case_dir / "targets.png",
                 case_dir / "source.mp4", case_dir / "mask_preview.mp4"]
        cases.append(dict(id=prefix, scene=scene, shot=0, entity=entity, spec=spec, caption=caption,
                          defect_proposal=defect, samples=samples, video=str(video), video_sha256=digest,
                          mask_path=str(mask), baseline_config=str(config), fps=fps, frame_count=count,
                          repair_prompt=f"{caption.strip()} {entity} is {spec['description']}.",
                          coverage=mask_coverage(soft), files={str(p): file_sha256(p) for p in paths if p.is_file()}))
        print(json.dumps({"prepared": prefix, "human_review": "pending"}), flush=True)
    packet = dict(protocol=PROTOCOL, cases=cases,
                  code_hashes={str(ROOT / p): file_sha256(ROOT / p) for p in CODE_FILES})
    write_json(output / "packet.json", packet)
    review_page(output, packet)


def review_page(output, packet):
    digest = file_sha256(output / "packet.json")
    sections = []
    for case in packet["cases"]:
        cid = case["id"]
        sections.append(f'''<section><h2>{cid} · {case['entity']} · 第 1 镜头</h2>
<p>待确认缺陷：{html.escape(case['defect_proposal'])}</p>
<p>人物设定：{html.escape(case['spec']['description'])}</p>
<details><summary>原始 caption / 固定重画提示词</summary><p>{html.escape(case['repair_prompt'])}</p></details>
<img src="{cid}/targets.png" alt="8 个 chunk 的候选框，黄色是拟修目标">
<p>左：原视频；右：实际 SAM2 修复区域（青色覆盖）。请检查全程是否选对人、是否侵入其他人物。</p>
<video controls loop src="{cid}/source.mp4"></video><video controls loop src="{cid}/mask_preview.mp4"></video>
<p><label><input type="checkbox" class="approval" value="{cid}">我确认人物、缺陷及修复区域可用于这个小实验</label></p></section>''')
    page = '''<!doctype html><meta charset="utf-8"><title>人工目标确认 · 尚未重画</title>
<style>body{max-width:1300px;margin:24px auto;font:18px sans-serif;line-height:1.5}section{border-top:1px solid #aaa;padding:20px 0}img{width:100%}video{width:49%}button,input{font:inherit}</style>
<h1>5 个修复目标：请先确认</h1><p>这些是助手提出的候选，不是已完成人工标注。这里只测试“给对目标，现有重画能否修正外观”。每例仅重画一次，不添人、不让 VLM 选结果。</p>
<p>全部是第一镜头的设定外观错误，不能据此宣称跨镜头身份保持成功。框或遮罩有问题请直接指出案例；不要为了凑够五例而勾选。</p>'''
    page += "\n".join(sections)
    page += '<p>评审姓名／标识：<input id="reviewer"></p><button onclick="downloadReview()">下载目标确认 JSON</button>'
    page += "<script>const packetHash=" + json.dumps(digest) + ";\n"
    page += '''function downloadReview(){const reviewer=document.getElementById('reviewer').value.trim();
const checked=[...document.querySelectorAll('.approval:checked')];
if(!reviewer||checked.length!==5){alert('请填写评审标识并逐例确认；有问题请反馈，不要勾选。');return;}
const cases=Object.fromEntries(checked.map(x=>[x.value,{target_and_defect_confirmed:true,mask_confirmed:true}]));
const review={packet_sha256:packetHash,reviewer,reviewer_kind:'human',reviewed_at:new Date().toISOString(),cases};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{type:'application/json'}));a.download='human_review.json';a.click();URL.revokeObjectURL(a.href);}
</script>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def validate_review(packet, review, packet_path):
    if packet["protocol"] not in (PROTOCOL, MACHINE_PROTOCOL):
        raise ValueError("unexpected pilot protocol")
    expected_kind = "assistant" if packet["protocol"] == MACHINE_PROTOCOL else "human"
    if review.get("reviewer_kind") != expected_kind or not str(review.get("reviewer", "")).strip():
        raise ValueError("an actual human target review is required" if expected_kind == "human"
                         else "machine protocol requires explicit assistant-review provenance")
    if review.get("packet_sha256") != file_sha256(packet_path):
        raise ValueError("review is not bound to this exact packet")
    if set(review.get("cases", {})) != {c["id"] for c in packet["cases"]}:
        raise ValueError("review must cover all fixed cases; do not silently drop failures")
    for case in packet["cases"]:
        reviewed = review["cases"][case["id"]]
        if reviewed.get("target_and_defect_confirmed") is not True or reviewed.get("mask_confirmed") is not True:
            raise ValueError(f"target, defect or mask not confirmed: {case['id']}")
        for name, expected in case["files"].items():
            if file_sha256(name) != expected:
                raise ValueError(f"reviewed artifact changed: {name}")
    for name, expected in packet["code_hashes"].items():
        if file_sha256(name) != expected:
            raise ValueError(f"implementation changed since protocol freeze: {name}")


def adopt_machine(source, output, review_path):
    """Reuse already checked artifacts, while retaining the original protocol."""
    source_packet = source / "packet.json"
    packet = json.loads(source_packet.read_text())
    review = json.loads(review_path.read_text())
    if packet["protocol"] != PROTOCOL or review.get("reviewer_kind") != "assistant":
        raise ValueError("expected a human-pending packet and an explicit assistant review")
    if review.get("packet_sha256") != file_sha256(source_packet):
        raise ValueError("assistant review must identify the original packet")
    if not review.get("user_requested_machine_pilot"):
        raise ValueError("machine pilot must be explicitly requested, not silently substituted")
    for case in packet["cases"]:
        if (source / case["id"] / "started.json").exists():
            raise ValueError("do not fork an already attempted pilot as a new attempt")
        for name, digest in case["files"].items():
            if file_sha256(name) != digest:
                raise ValueError(f"source artifact changed: {name}")
    for name, digest in packet["code_hashes"].items():
        if Path(name) != Path(__file__).resolve() and file_sha256(name) != digest:
            raise ValueError(f"executor changed since target preparation: {name}")
    output.mkdir(parents=True, exist_ok=False)
    def relocated(name):
        path = Path(name)
        return str(output / path.relative_to(source)) if path.is_relative_to(source) else str(path)
    for case in packet["cases"]:
        shutil.copytree(source / case["id"], output / case["id"])
        case["mask_path"] = relocated(case["mask_path"])
        case["baseline_config"] = relocated(case["baseline_config"])
        case["files"] = {relocated(name): digest for name, digest in case["files"].items()}
    packet["protocol"] = deepcopy(MACHINE_PROTOCOL)
    packet["source_packet"] = dict(path=str(source_packet), sha256=file_sha256(source_packet),
                                   assistant_review_sha256=file_sha256(review_path))
    packet["code_hashes"] = {str(ROOT / p): file_sha256(ROOT / p) for p in CODE_FILES}
    write_json(output / "packet.json", packet)
    review["source_packet_sha256"] = review["packet_sha256"]
    review["packet_sha256"] = file_sha256(output / "packet.json")
    validate_review(packet, review, output / "packet.json")
    write_json(output / "assistant_review.json", review)
    print(json.dumps({"output": str(output), "reviewer_kind": "assistant", "human_validated": False}), flush=True)


def run_case(output, review_path, case_id, gpu):
    from scripts.audit_narrastream_baselines import pixels
    from scripts.run_agent_edit_demo import EditRenderBackend, build_render_config, find_render, render

    packet_path = output / "packet.json"
    packet, review = json.loads(packet_path.read_text()), json.loads(review_path.read_text())
    validate_review(packet, review, packet_path)
    kind = review["reviewer_kind"]
    case, = [c for c in packet["cases"] if c["id"] == case_id]
    case_dir = output / case_id
    # Exclusive marker prevents accidental best-of retries or concurrent renders.
    write_json(case_dir / "started.json", dict(packet_sha256=file_sha256(packet_path),
               target_review=review, target_review_sha256=file_sha256(review_path), reviewer_kind=kind, gpu=gpu))
    try:
        control_cfg = build_render_config(Path(case["baseline_config"]), scene=case["scene"],
            data_dir=case_dir / "data", output_folder=case_dir / "control", prefix="no_edit",
            edit=None, target=case_dir / "configs/no_edit.yaml")
        render(control_cfg, gpu=gpu, log_path=case_dir / "logs/no_edit.log")
        control = find_render(case_dir / "control", "no_edit", case["scene"])
        exact = file_sha256(control) == case["video_sha256"] or pixels(control) == pixels(Path(case["video"]))
        write_json(case_dir / "control_check.json", dict(decoded_pixels_equal=exact,
                   video=str(control), sha256=file_sha256(control)))
        if not exact:
            raise RuntimeError("unedited replay differs from reviewed baseline; stop and review this confound")
        backend = EditRenderBackend(scene=case["scene"], scene_dir=case_dir / "data" / case["scene"],
            out_dir=case_dir, base_config=Path(case["baseline_config"]), baseline_video=Path(case["video"]),
            render_gpu=gpu, clipseg_device="cpu", noise_seed=packet["protocol"]["noise_seed"] - 1)
        action = Action("edit_shot", {"shot": 0, "mask_path": case["mask_path"],
            "entry_sigma": packet["protocol"]["entry_sigma"], "prompt": case["repair_prompt"]},
            reason="human_confirmed_oracle_target" if kind == "human" else "assistant_proposed_exploratory_target").normalized(shot_index=0)
        _, result = backend.edit(0, 1, action, Verdict(False, f"{kind}_reviewed_appearance_defect"))
        write_json(case_dir / "result.json", dict(status=f"rendered_not_{kind}_rated", result=result,
                   packet_sha256=file_sha256(packet_path), target_review_sha256=file_sha256(review_path),
                   reviewer_kind=kind, human_validated=kind == "human"))
    except Exception as error:
        write_json(case_dir / "failure.json", dict(error=f"{type(error).__name__}: {error}"))
        raise


def export(output):
    packet = json.loads((output / "packet.json").read_text())
    reviewer_kind = "assistant" if packet["protocol"] == MACHINE_PROTOCOL else "human"
    results = [json.loads((output / c["id"] / "result.json").read_text()) for c in packet["cases"]]
    # Refuse partial-success export; every predeclared case must have a result.
    for case, result in zip(packet["cases"], results):
        if result["packet_sha256"] != file_sha256(output / "packet.json"):
            raise ValueError("result belongs to a different packet")
        if file_sha256(case["video"]) != case["video_sha256"] or file_sha256(result["result"]["edited_video"]) != result["result"]["edited_video_sha256"]:
            raise ValueError("comparison video changed")
    public = output / "blind_review"
    public.mkdir(exist_ok=False)
    order = list(zip(packet["cases"], results))
    secrets.SystemRandom().shuffle(order)
    key, items, sections = {}, {}, []
    for index, (case, result) in enumerate(order, 1):
        item = f"item{index:02d}"
        baseline_side = secrets.choice(["A", "B"])
        key[item] = dict(case=case["id"], baseline_side=baseline_side)
        items[item] = {side: {field: "unrated" for field in
            ("appearance", "action", "other_people", "later_shots")} for side in ("A", "B")}
        for side in ("A", "B"):
            source = Path(case["video"] if side == baseline_side else result["result"]["edited_video"])
            clip(source, public / f"{item}_{side}_target.mp4", 0, 64)
            clip(source, public / f"{item}_{side}_later.mp4", 64, case["frame_count"] - 64)
        sections.append(f'<section><h2>{item} · {html.escape(case["entity"])}</h2><p>人物设定：{html.escape(case["spec"]["description"])}</p><p>原始动作：{html.escape(case["caption"])}</p>'
            + ''.join(f'<h3>{label}（左 A / 右 B）</h3><video controls loop src="{item}_A_{suffix}.mp4"></video><video controls loop src="{item}_B_{suffix}.mp4"></video>'
                      for suffix, label in (("target", "目标镜头"), ("later", "全部后续镜头"))) + '</section>')
    write_json(output / "blind_key.private.json", key)
    write_json(public / "ratings.json", dict(reviewer="", reviewer_kind=reviewer_kind, cases=items,
        allowed_values=["pass", "fail", "uncertain", "not_applicable"],
        instructions="Judge A and B separately. Improvement alone is not an appearance pass; the specified defect must actually be corrected."))
    title = "人工 A/B 评价" if reviewer_kind == "human" else "模型辅助 A/B 检查（不是人评）"
    (public / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
        '<style>body{max-width:1300px;margin:auto;font:18px sans-serif}video{width:49%}section{border-top:1px solid #aaa}</style>'
        f'<h1>{title}</h1><p>分别判断：人物外观符合设定、原始动作、其他人物是否被破坏、后续镜头是否退化。'
        '填写 ratings.json，对 A、B 分别标 pass / fail / uncertain / not_applicable；外观只是略有改善但仍不符合设定，不能标 pass。不要查看包外的 private key。'
        '最好由未看过目标筛选图包的另一位评审完成；看过 baseline 的评审不算完全盲评。</p>'
        + '\n'.join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "adopt-machine", "run", "export"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command == "prepare":
            p.add_argument("--device", default="cuda:0")
        elif command == "adopt-machine":
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--review", type=Path, required=True)
        elif command == "run":
            p.add_argument("--review", type=Path, required=True)
            p.add_argument("--case", choices=[p[0] for p in PROPOSALS], required=True)
            p.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        prepare(output, args.device)
    elif args.command == "adopt-machine":
        adopt_machine(args.source.resolve(), output, args.review.resolve())
    elif args.command == "run":
        run_case(output, args.review.resolve(), args.case, args.gpu)
    else:
        export(output)


if __name__ == "__main__":
    main()
