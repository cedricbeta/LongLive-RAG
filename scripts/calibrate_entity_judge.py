#!/usr/bin/env python3
"""Frozen, no-render calibration packets and stage-separated judge checks.

prepare reads archived enumeration responses, never a judge's role/appearance
decisions. Labels are added separately after visual review, before run. run uses
reviewed candidate IDs for comparer crops, isolating assignment from appearance.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
from utils.agent_entity_judge import EntitySpec, VLMComparer, VLMLocalizer, scale_box_norm1000
from utils.agent_runtime import file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def archived_index(cache):
    result = {}
    for path in sorted((cache / "entity_enumerate").glob("*.json")):
        record = json.loads(path.read_text())
        hashes = record.get("ordered_image_hashes", [])
        if len(hashes) != 1 or "parsed" not in record:
            continue
        prior = result.get(hashes[0])
        if prior and prior["parsed"] != record["parsed"]:
            raise ValueError(f"ambiguous archived enumeration: {path}")
        result[hashes[0]] = record
    return result


def prepare(output):
    if (output / "packet.json").exists():
        raise FileExistsError("packet already exists; use a fresh output directory")
    specs_path = ROOT / "example/narrastream_bench/entity_specs_subset33.json"
    specs = json.loads(specs_path.read_text())
    sources = []
    for scene in ("ns190", "ns290"):
        replay = ROOT / f"videos/entity_assignment_replay_20260905/indexed/{scene}/results.json"
        data = json.loads(replay.read_text())
        for case in data["cases"]:
            shot, attempt, _ = case["case"].split(":", 2)
            refs = {}
            for name, available in case["old"]["metrics"].get("reference_available", {}).items():
                if available:
                    prov = case["old"]["metrics"]["reference_registration"][name]["provenance"]
                    path = Path(prov["reference_crop"])
                    if file_sha256(path) != prov["reference_crop_sha256"]:
                        raise ValueError(f"changed reference: {path}")
                    refs[name] = str(path)
            sources.append(dict(scene=data["scene"], source=Path(data["source_run"]), video=Path(case["video"]),
                                shot=int(shot), attempt=int(attempt), split="development", refs=refs))
    old_root = ROOT / "videos/narrastream_threearm_20260827"
    # Fixed scene-level holdout, disjoint from both development scenes.
    for prefix in ("ns000", "ns040", "ns090", "ns130", "ns210", "ns310"):
        scene = next(s for s in specs if s.startswith(prefix))
        source, = (old_root / "armC_entity").glob(f"g*/{scene}")
        video, = (source / "renders").glob("baseline-*.mp4")
        for shot in (0, 1):
            # Script-only appearance tests avoid importing potentially wrong
            # historical identity anchors into the independent holdout labels.
            sources.append(dict(scene=scene, source=source, video=video, shot=shot,
                                attempt=0, split="holdout", refs={}))
    indexes, cases, skipped = {}, [], []
    overlay = VLMLocalizer(SimpleNamespace(cache_dir=output / "inputs"))
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    for item in sources:
        scene, shot = item["scene"], item["shot"]
        case_id = f"{scene[:5]}_s{shot:02d}_a{item['attempt']}"
        data_dir = ROOT / "example/narrastream_bench/multitext_subset33_b8" / scene
        caption = json.loads((data_dir / f"{shot}.json").read_text())["caption"]
        durations = [int(v) for v in (data_dir / "shot_durations.txt").read_text().split()]
        c0, c1 = sum(durations[:shot]), sum(durations[:shot + 1]) - 1
        digest = file_sha256(item["video"])
        frames = item["source"] / "entity_judge/frames" / f"{digest[:16]}_c{c0:02d}-{c1:02d}"
        if not all((frames / f"chunk{c:02d}.png").is_file() for c in (0, 3, 7)):
            if item["split"] != "holdout":
                raise FileNotFoundError(frames)
            skipped.append({"id": case_id, "reason": "baseline frames not archived; no edited render substitution"})
            continue
        cache = item["source"].parent / "vlm_cache"
        if cache not in indexes:
            indexes[cache] = archived_index(cache)
        samples = []
        sheet = Image.new("RGB", (1920, 382), "#202020")
        for i, c in enumerate((0, 3, 7)):
            frame = frames / f"chunk{c:02d}.png"
            record = indexes[cache][file_sha256(frame)]
            with Image.open(frame) as im:
                width, height = im.size
            boxes = [scale_box_norm1000(s["bbox_2d"], width, height) for s in record["parsed"]["subjects"]]
            boxes = [b for b in boxes if b]
            annotated = overlay._numbered_candidates(frame, boxes)
            with Image.open(annotated) as im:
                panel = im.convert("RGB")
                panel.thumbnail((640, 352))
            sheet.paste(panel, (640 * i, 30))
            ImageDraw.Draw(sheet).text((640 * i + 5, 4), f"{case_id} / chunk {c}", font=font, fill="white")
            samples.append(dict(chunk=c, frame=str(frame), frame_sha256=file_sha256(frame),
                                boxes=boxes, enumeration_fingerprint=record["cache_fingerprint"]))
        sheet_path = output / "sheets" / f"{case_id}.png"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path)
        active = [s for s in specs[scene] if EntitySpec.from_value(s).presence_in(shot) != "not_required"]
        cases.append(dict(id=case_id, split=item["split"], scene=scene, shot=shot, attempt=item["attempt"],
                          caption=caption, entities=active, references=item["refs"],
                          reference_sha256={n: file_sha256(p) for n, p in item["refs"].items()},
                          video=str(item["video"]), video_sha256=digest, samples=samples, sheet=str(sheet_path)))
    write_json(output / "packet.json", dict(schema_version=1, spec_sha256=file_sha256(specs_path),
        sample_chunks=[0, 3, 7], selection="known development failures plus six fixed disjoint holdout scenes; shots 0 and 1",
        labels_source="pending independent visual review; no Qwen role/appearance verdict copied", skipped=skipped, cases=cases))
    print(json.dumps({"cases": len(cases), "packet": str(output / "packet.json")}), flush=True)


def validate_packet(packet, labels, packet_path):
    if file_sha256(packet_path) != labels["packet_sha256"]:
        raise ValueError("packet changed after review")
    if set(labels["cases"]) != {c["id"] for c in packet["cases"]}:
        raise ValueError("review must cover every case exactly")
    for case in packet["cases"]:
        reviewed = labels["cases"][case["id"]]
        if set(reviewed) != {e["name"] for e in case["entities"]}:
            raise ValueError(f"incomplete role review: {case['id']}")
        for i, sample in enumerate(case["samples"]):
            if file_sha256(sample["frame"]) != sample["frame_sha256"]:
                raise ValueError(f"frame changed: {sample['frame']}")
            claimed = []
            for label in reviewed.values():
                if len(label["candidate_ids"]) != 3 or len(label["appearance"]) != 3:
                    raise ValueError("each role needs three sample labels")
                index = label["candidate_ids"][i]
                if index is not None and (type(index) is not int or not -1 <= index < len(sample["boxes"])):
                    raise ValueError(f"invalid reviewed ID: {case['id']}")
                if index is not None and index >= 0:
                    claimed.append(index)
                elif label["appearance"][i] is not None:
                    raise ValueError("appearance label requires a visible reviewed subject")
            if len(claimed) != len(set(claimed)):
                raise ValueError("review assigns one candidate to multiple roles")
        for name, path in case["references"].items():
            if file_sha256(path) != case["reference_sha256"][name]:
                raise ValueError(f"reference changed: {path}")


def score_cases(cases, labels):
    counts = dict(assignment_correct=0, assignment_total=0, assignment_excluded=0, assignment_abstentions=0,
                  appearance_correct=0, appearance_total=0, appearance_excluded=0,
                  drift_detected=0, drift_total=0, normal_accepted=0, normal_total=0,
                  false_reference_registrations=0, false_present_on_reviewed_absence=0)
    errors = []
    for result in cases:
        for name, label in labels["cases"][result["id"]].items():
            for i, expected in enumerate(label["candidate_ids"]):
                actual = result["assignments"][i][name]
                if expected is None:
                    counts["assignment_excluded"] += 1
                    continue
                counts["assignment_total"] += 1
                correct = (actual["status"] == "absent") if expected == -1 else (
                    actual["status"] == "assigned" and actual["subject_index"] == expected)
                counts["assignment_correct"] += int(correct)
                counts["assignment_abstentions"] += int(actual["status"] == "uncertain")
                counts["false_present_on_reviewed_absence"] += int(expected == -1 and actual["status"] == "assigned")
                if not correct:
                    errors.append(dict(case=result["id"], entity=name, sample=i, stage="assignment", expected=expected, actual=actual))
            comparison = result["comparisons"].get(name)
            by_sample = {} if comparison is None else dict(zip(comparison["sample_indices"], comparison["crop_votes"]))
            for i, expected in enumerate(label["appearance"]):
                if expected is None:
                    counts["appearance_excluded"] += 1
                    continue
                actual = by_sample.get(i)
                correct = actual is expected
                counts["appearance_total"] += 1
                counts["appearance_correct"] += int(correct)
                if expected:
                    counts["normal_total"] += 1
                    counts["normal_accepted"] += int(correct)
                else:
                    counts["drift_total"] += 1
                    counts["drift_detected"] += int(correct)
                if not correct:
                    errors.append(dict(case=result["id"], entity=name, sample=i, stage="appearance", expected=expected, actual=actual))
            if (comparison and not comparison["has_reference"] and False in label["appearance"]
                    and comparison["reference_registration_eligible"]):
                counts["false_reference_registrations"] += 1
    def ratio(n, d):
        return counts[n] / counts[d] if counts[d] else None
    return {**counts, "assignment_accuracy": ratio("assignment_correct", "assignment_total"),
            "appearance_accuracy": ratio("appearance_correct", "appearance_total"),
            "drift_recall": ratio("drift_detected", "drift_total"),
            "normal_specificity": ratio("normal_accepted", "normal_total"), "errors": errors}


def gate_checks(summary, policy, split):
    checks = {}
    for metric in ("assignment_accuracy", "appearance_accuracy"):
        checks[metric] = summary[metric] is not None and summary[metric] >= policy[f"{split}_{metric}_min"]
    if split == "holdout":
        for metric in ("drift_recall", "normal_specificity"):
            checks[metric] = summary[metric] is not None and summary[metric] >= policy[f"holdout_{metric}_min"]
    for metric in ("false_reference_registrations", "false_present_on_reviewed_absence"):
        checks[metric] = summary[metric] <= policy[f"{metric}_max"]
    return {"passed": all(checks.values()), "checks": checks}


def run(args):
    from utils.agent_appearance_judge import ObservedAppearanceComparer
    from utils.agent_appearance_evidence import EvidenceAppearanceComparer
    from utils.agent_appearance_color import ColorCheckedAppearanceComparer, set_part_device
    from utils.agent_role_assignment import ContinuityRoleLocalizer
    if "color" in args.variants:
        set_part_device(args.part_device)
    packet = json.loads(args.packet.read_text())
    labels = json.loads(args.labels.read_text())
    validate_packet(packet, labels, args.packet)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: file_sha256(ROOT / name) for name in (
        "utils/agent_entity_judge.py", "utils/agent_appearance_judge.py", "utils/agent_appearance_evidence.py", "utils/agent_role_assignment.py",
        "utils/agent_appearance_color.py", "utils/agent_vlm.py", "scripts/calibrate_entity_judge.py")}
    for name in hashes:
        snapshot = args.output / "implementation" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and file_sha256(snapshot) != hashes[name]:
            raise ValueError(f"implementation snapshot already differs: {snapshot}")
        shutil.copyfile(ROOT / name, snapshot)
    proc = None
    try:
        base_url = args.base_url
        if not base_url:
            from scripts.run_agent_edit_demo import start_vlm_server
            proc, base_url = start_vlm_server(args, args.output)
        write_json(args.output / "server_endpoint.json", {"base_url": base_url, "owned_pid": proc.pid if proc else None})
        client = FingerprintedVLMClient(base_url=base_url, model=args.model,
            cache_dir=args.cache or args.output / "vlm_cache", max_tokens=1024, seed=0)
        localizer = ContinuityRoleLocalizer(client) if args.role_variant == "indexed" else VLMLocalizer(client)
        selected = [c for c in packet["cases"] if c["split"] == args.split]
        for variant in args.variants:
            target = args.output / f"{variant}_{args.split}.json"
            if target.exists():
                raise FileExistsError(f"result already exists; choose a fresh output: {target}")
            comparer = {"baseline": VLMComparer, "observed": ObservedAppearanceComparer, "evidence": EvidenceAppearanceComparer,
                        "color": ColorCheckedAppearanceComparer}[variant](client)
            result = dict(variant=variant, role_variant=args.role_variant, stage="assignment_only" if args.assignment_only else "both",
                split=args.split, packet_sha256=file_sha256(args.packet),
                labels_sha256=file_sha256(args.labels), implementation_sha256=hashes, sampling=client.sampling_config,
                comparer_crops="reviewer-assigned candidate boxes, independent of predicted role IDs",
                video_generation_calls=0, cases=[], completed=False)
            for case in selected:
                started = time.time()
                entities = [EntitySpec.from_value(e) for e in case["entities"]]
                refs = {n: Path(p) for n, p in case["references"].items()}
                assignments, previous, comparisons = [], {}, {}
                for sample in case["samples"]:
                    rows = localizer.assign_subjects(entities, refs, Path(sample["frame"]), sample["boxes"],
                        caption=case["caption"], previous_boxes=previous, shot_index=case["shot"], inputs=())
                    assignments.append(rows)
                    previous = {n: sample["boxes"][row["subject_index"]] for n, row in rows.items() if row["status"] == "assigned"}
                for entity in entities:
                    if args.assignment_only:
                        break
                    label = labels["cases"][case["id"]][entity.name]
                    if not any(v is not None for v in label["appearance"]):
                        continue
                    crops, indices = [], []
                    for i, (sample, index) in enumerate(zip(case["samples"], label["candidate_ids"])):
                        if index is None or index < 0:
                            continue
                        path = args.output / "crops" / f"{case['id']}_{entity.name}_{i}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with Image.open(sample["frame"]) as frame:
                            frame.crop(tuple(sample["boxes"][index])).save(path)
                        crops.append(path)
                        indices.append(i)
                    comparison = comparer.compare(entity, refs.get(entity.name), crops, shot_index=case["shot"], inputs=())
                    comparisons[entity.name] = {**comparison, "sample_indices": indices, "has_reference": entity.name in refs}
                result["cases"].append(dict(id=case["id"], assignments=assignments, comparisons=comparisons, wall_seconds=time.time() - started))
                result["summary"] = score_cases(result["cases"], labels)
                write_json(target, result)
                print(json.dumps({"variant": variant, "case": case["id"], "finished": len(result["cases"]), "total": len(selected)}), flush=True)
            result["completed"] = True
            result["gate"] = None if args.assignment_only else gate_checks(result["summary"], labels["gate"], args.split)
            write_json(target, result)
            print(json.dumps({"variant": variant, "gate": result["gate"], "summary": {k:v for k,v in result["summary"].items() if k != "errors"}}), flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)


def merge(args):
    """Join independently evaluated stages, never mix labels or packet versions."""
    assignment = json.loads(args.assignment_results.read_text())
    comparison = json.loads(args.comparison_results.read_text())
    labels = json.loads(args.labels.read_text())
    if not assignment.get("completed") or not comparison.get("completed"):
        raise ValueError("both stage runs must be complete")
    if assignment.get("stage") != "assignment_only":
        raise ValueError("assignment input must be an assignment-only run")
    for key in ("packet_sha256", "labels_sha256", "split", "sampling"):
        if assignment[key] != comparison[key]:
            raise ValueError(f"cannot merge different {key}")
    if file_sha256(args.labels) != assignment["labels_sha256"]:
        raise ValueError("labels changed")
    by_id = {c["id"]: c for c in comparison["cases"]}
    if set(by_id) != {c["id"] for c in assignment["cases"]}:
        raise ValueError("stages cover different cases")
    result = {**assignment, "stage": "merged_independent_stages", "variant": comparison["variant"],
              "source_stages": {"assignment": {"path": str(args.assignment_results), "sha256": file_sha256(args.assignment_results)},
                                "comparison": {"path": str(args.comparison_results), "sha256": file_sha256(args.comparison_results)}},
              "comparison_implementation_sha256": comparison["implementation_sha256"],
              "cases": [{**c, "comparisons": by_id[c["id"]]["comparisons"]} for c in assignment["cases"]]}
    result["summary"] = score_cases(result["cases"], labels)
    result["gate"] = gate_checks(result["summary"], labels["gate"], result["split"])
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, result)
    print(json.dumps({"gate": result["gate"], "summary": {k:v for k,v in result["summary"].items() if k != "errors"}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run", "merge"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--packet", type=Path, default=ROOT / "videos/entity_judge_calibration_20260905/review/packet.json")
    ap.add_argument("--labels", type=Path, default=ROOT / "docs/entity_judge_calibration_20260905/labels.json")
    ap.add_argument("--variants", nargs="+", choices=["baseline", "observed", "evidence", "color"], default=["baseline", "observed"])
    ap.add_argument("--part_device", default="cuda:0", help="CLIPSeg device for the color variant's hair/clothing parts")
    ap.add_argument("--role_variant", choices=["baseline", "indexed"], default="baseline")
    ap.add_argument("--assignment_only", action="store_true")
    ap.add_argument("--assignment_results", type=Path)
    ap.add_argument("--comparison_results", type=Path)
    ap.add_argument("--split", choices=["development", "holdout"], default="development")
    ap.add_argument("--base_url")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sglang_python", default="/home/chendong/miniconda3/envs/sglang/bin/python")
    ap.add_argument("--mem_fraction_static", type=float, default=0.45)
    ap.add_argument("--startup_timeout", type=int, default=300)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve())
    elif args.command == "merge":
        merge(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
