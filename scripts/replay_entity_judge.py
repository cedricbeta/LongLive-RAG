#!/usr/bin/env python3
"""Replay the entity judge on archived frames; never calls a generation backend.

Example:
  venv/bin/python scripts/replay_entity_judge.py --source_run \
    videos/narrastream_place_smoke_20260904/ns290/ns290_at_the_restaurant_prep \
    --case 1:0:baseline --case 1:1:edit_shot01_attempt1 \
    --case 2:0:edit_shot01_attempt1 --case 2:2:edit_shot02_attempt2 \
    --case 3:0:edit_shot01_attempt1 --output videos/entity_assignment_replay/ns290

Enumeration proposals and pre-judgment references are frozen from the archived
run. Only assignment/comparison are queried again. All candidates are judged
independently: this diagnoses the judge, not the delivery policy or video quality.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.agent_entity_judge import EntityDriftVerifier, Sam2Tracker, VLMComparer, VLMLocalizer, load_entity_specs, scale_box_norm1000
from utils.agent_runtime import NormativeGraph, ShotObservation, file_sha256
from utils.agent_vlm import FingerprintedVLMClient

IMPLEMENTATION_HASHES = {name: file_sha256(ROOT / name) for name in (
    "utils/agent_entity_judge.py", "utils/agent_constraints.py", "utils/agent_vlm.py", "scripts/replay_entity_judge.py")}


class FrozenEnumeration(VLMLocalizer):
    def __init__(self, client, old_cache: Path):
        super().__init__(client)
        self.by_image = {}
        for path in sorted((old_cache / "entity_enumerate").glob("*.json")):
            record = json.loads(path.read_text())
            if "parsed" not in record or record.get("fatal"):
                continue
            hashes = record.get("ordered_image_hashes", [])
            if len(hashes) != 1:
                continue
            prior = self.by_image.get(hashes[0])
            if prior and prior["parsed"] != record["parsed"]:
                raise ValueError(f"multiple enumeration answers for identical image: {path}")
            self.by_image[hashes[0]] = record
        self.used = {}
        self.old_locations = {}
        for path in sorted((old_cache / "entity_locate").glob("*.json")):
            record = json.loads(path.read_text())
            if "parsed" not in record or record.get("fatal"):
                continue
            prompt = json.loads(record["request"]["messages"][0]["content"][0]["text"])
            hashes = record["ordered_image_hashes"]
            key = (hashes[-1], prompt["entity"], hashes[0] if len(hashes) > 1 else None)
            self.old_locations[key] = record["parsed"]

    def archived_boxes(self, frames, old, refs, num_chunks):
        """Older ledgers omitted boxes; recover the exact localization calls."""
        result = copy.deepcopy(old)
        width, height = self._size(frames / "chunk00.png")
        for name, entity in result["metrics"].get("entities", {}).items():
            if entity.get("boxes") or not entity.get("expected"):
                continue
            boxes = {}
            for c in range(num_chunks):
                key = (file_sha256(frames / f"chunk{c:02d}.png"), name, file_sha256(refs[name]) if name in refs else None)
                location = self.old_locations.get(key)
                if location and location["present"]:
                    box = scale_box_norm1000(location["bbox_2d"], width, height)
                    if box:
                        boxes[c] = box
            entity["boxes"] = boxes
            entity["boxes_source"] = "archived_entity_locate_transcripts"
        return result

    def enumerate_subjects(self, frame, *, shot_index, inputs):
        digest = file_sha256(frame)
        record = self.by_image.get(digest)
        if record is None:
            raise RuntimeError(f"no archived enumeration for {frame}; frozen replay will not substitute a fresh call")
        width, height = self._size(frame)
        boxes = [scale_box_norm1000(s["bbox_2d"], width, height) for s in record["parsed"]["subjects"]]
        self.used[str(frame)] = {"image_sha256": digest, "old_call_fingerprint": record["cache_fingerprint"]}
        return [box for box in boxes if box]


class ArchivedFrameVerifier(EntityDriftVerifier):
    def __init__(self, *args, archived_frames: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.archived_frames = archived_frames

    def _shot_frames(self, video, chunk_start, chunk_end):
        from PIL import Image
        directory = self.archived_frames / f"{file_sha256(video)[:16]}_c{chunk_start:02d}-{chunk_end:02d}"
        n = (chunk_end - chunk_start + 1) * self.fpc
        for name in ("0.jpg", f"{n - 1}.jpg", "chunk00.png"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"archived frame missing: {directory / name}")
        with Image.open(directory / "0.jpg") as image:
            width, height = image.size
        return directory, n, (height, width)


def prior_references(old):
    """Freeze exactly the references the archived candidate was judged against."""
    refs = {}
    registrations = old["metrics"].get("reference_registration", {})
    for name, available in old["metrics"].get("reference_available", {}).items():
        if not available:
            continue
        provenance = registrations.get(name, {}).get("provenance") or {}
        path = Path(provenance.get("reference_crop", ""))
        if not path.is_file():
            raise RuntimeError(f"missing pre-judgment reference for {name}: {path}")
        if file_sha256(path) != provenance["reference_crop_sha256"]:
            raise RuntimeError(f"archived reference changed: {path}")
        refs[name] = path
    return refs


def draw_comparison(frames, old, new, target):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    colors = ["#20ee70", "#ffbd36", "#28caff", "#ed75e6", "#ffffff", "#ff6262"]
    entities = sorted(set(old.get("entities", {})) | set(new.get("entities", {})))
    chunks = [0, 4, 7]
    panels = []
    for c in chunks:
        for title, metrics in (("archived", old), ("candidate assignment", new)):
            with Image.open(frames / f"chunk{c:02d}.png") as original:
                image = original.copy()
            draw = ImageDraw.Draw(image)
            for i, name in enumerate(entities):
                entity = metrics.get("entities", {}).get(name, {})
                boxes = entity.get("boxes", {})
                box = boxes.get(c, boxes.get(str(c)))
                if box:
                    draw.rectangle(box, outline=colors[i % len(colors)], width=4)
                    draw.text((box[0] + 4, max(30, box[1])), name, font=font, fill=colors[i % len(colors)], stroke_width=1, stroke_fill="black")
            draw.rectangle((0, 0, image.width, 30), fill="black")
            draw.text((6, 0), f"{title} / chunk {c}", font=font, fill="white")
            image.thumbnail((640, 352))
            panels.append(image)
    sheet = Image.new("RGB", (1280, 352 * len(chunks)))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * 640, (i // 2) * 352))
    sheet.save(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_run", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True, help="shot:attempt:archived_render_prefix")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--entity_specs", type=Path, default=ROOT / "example/narrastream_bench/entity_specs_subset33.json")
    ap.add_argument("--data_root", type=Path, default=ROOT / "example/narrastream_bench/multitext_subset33_b8")
    ap.add_argument("--base_url")
    ap.add_argument("--judge_cache", type=Path)
    ap.add_argument("--judge_seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tracker_device", default="cuda:1")
    ap.add_argument("--sglang_python", default="/home/chendong/miniconda3/envs/sglang/bin/python")
    ap.add_argument("--mem_fraction_static", type=float, default=0.45)
    ap.add_argument("--startup_timeout", type=int, default=300)
    args = ap.parse_args()
    source = args.source_run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = source.name
    specs = load_entity_specs(args.entity_specs, scene)
    scene_dir = args.data_root / scene
    graph = NormativeGraph.compile_scene(scene_dir, allow_missing_global=True)
    durations = [int(v) for v in (scene_dir / "shot_durations.txt").read_text().split()]
    rows = [json.loads(line) for line in (source / "ledger.jsonl").read_text().splitlines()]
    proc = None
    try:
        base_url = args.base_url
        if not base_url:
            # This helper only launches a judge server; no render function is called.
            from scripts.run_agent_edit_demo import start_vlm_server
            proc, base_url = start_vlm_server(args, output)
        client = FingerprintedVLMClient(base_url=base_url, model=args.model,
            cache_dir=args.judge_cache or output / "vlm_cache", max_tokens=1024, seed=args.judge_seed)
        localizer = FrozenEnumeration(client, source.parent / "vlm_cache")
        tracker = Sam2Tracker(device=args.tracker_device)
        results = {"scene": scene, "source_run": str(source), "entity_specs_sha256": file_sha256(args.entity_specs),
                   "implementation_sha256_at_start": IMPLEMENTATION_HASHES,
                   "sampling": client.sampling_config, "enumeration_policy": "frozen_archived_proposals",
                   "reference_policy": "frozen_pre_judgment_archived_references", "place_absent": "never",
                   "video_generation_calls": 0, "cases": []}
        for case in args.case:
            shot_s, attempt_s, prefix = case.split(":", 2)
            shot, attempt = int(shot_s), int(attempt_s)
            old = next(r for r in rows if r["event"] == "verifier" and r["shot_index"] == shot and r["attempt"] == attempt)
            videos = list((source / "renders").glob(f"{prefix}-rank0-{scene}-seed*_regular.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"expected one archived video for {case}: {videos}")
            video = videos[0]
            c0, c1 = sum(durations[:shot]), sum(durations[:shot + 1]) - 1
            verifier = ArchivedFrameVerifier(specs, localizer=localizer, comparer=VLMComparer(client), tracker=tracker,
                work_dir=output / case.replace(":", "_"), archived_frames=source / "entity_judge/frames", place_absent="never")
            verifier.reference_crops = prior_references(old)
            for name in verifier.reference_crops:
                verifier.reference_provenance[name] = old["metrics"]["reference_registration"][name]["provenance"]
            observation = ShotObservation(shot_index=shot, attempt=attempt, frames=[], crops=[],
                telemetry={"shot_video": str(video), "chunk_start": c0, "chunk_end": c1},
                inputs=(), input_fingerprint=old["input_fingerprint"])
            frames, _, _ = verifier._shot_frames(video, c0, c1)
            old = localizer.archived_boxes(frames, old, verifier.reference_crops, c1 - c0 + 1)
            started = time.time()
            verdict = verifier.judge(observation, graph.view(shot))
            frames, _, _ = verifier._shot_frames(video, c0, c1)
            sheet = output / f"shot{shot:02d}_attempt{attempt}_boxes.png"
            draw_comparison(frames, old["metrics"], verdict.metrics, sheet)
            results["cases"].append({"case": case, "video": str(video), "video_sha256": file_sha256(video),
                "wall_seconds": time.time() - started, "comparison_sheet": str(sheet),
                "old": {"accept": old["accept"], "reason": old["reason"], "metrics": old["metrics"]},
                "new": {"accept": verdict.accept, "reason": verdict.reason, "metrics": dict(verdict.metrics)}})
            results["frozen_enumerations"] = localizer.used
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"case": case, "old": old["reason"], "new": verdict.reason,
                "entities": {n: {k: e[k] for k in ("located_chunks", "matches", "crop_votes", "presence")} for n, e in verdict.metrics["entities"].items()}}), flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
