#!/usr/bin/env python
"""Read NarraStream evaluation runs side by side without averaging away the signal.

Two modes:

consistency  A/B: two evaluation output dirs of the SAME videos (e.g. A evaluated twice
             with fresh caches).  Identical inputs must give identical scores; any
             per-sample difference is evaluator noise.  Reports per-sample and
             per-segment (entity_grounding) differences and fallback counts per run.

touched      A vs C: paired per-segment comparison restricted to the segments the
             closed loop actually changed (accepted repairs from the C generation
             root's summary.json files), with untouched segments as the control.
             Per-segment scores come from the local metric caches (entity_grounding
             items; vlm_score segment_scores from the raw response).

Neither mode certifies quality; both make the arithmetic visible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

JUDGED = ("conditional_adjacent", "conditional_longrange", "entity_grounding", "vlm_score")


def per_sample(output: Path) -> dict[int, dict]:
    data = json.loads((output / "results_latest.json").read_text())
    return {int(s["sample_id"]): s for s in data["per_sample"]}


def eg_segments(output: Path) -> dict[tuple[int, int], dict]:
    """(sample_id, segment) -> {score, fallback}"""
    out = {}
    for path in glob.glob(str(output / "local_cache/metrics/entity_grounding__*/items/segments/sample_*__item_*.json")):
        name = os.path.basename(path)
        sid = int(name.split("__")[0].split("_")[1]); seg = int(name.rsplit("item_", 1)[1].split(".")[0])
        d = json.loads(Path(path).read_text())
        out[(sid, seg)] = {"score": float(d.get("score", float("nan"))),
                           "fallback": "error" if "error" in d else ("empty" if d.get("reason") == "empty_entities" else None)}
    return out


def vlm_segments(output: Path) -> dict[tuple[int, int], float]:
    out = {}
    for path in glob.glob(str(output / "local_cache/metrics/vlm_score*/samples/sample_*.json")):
        sid = int(os.path.basename(path).split("_")[1].split(".")[0])
        d = json.loads(Path(path).read_text())
        scores = d.get("segment_scores") or (d.get("parsed") or {}).get("segment_scores")
        if not scores:
            raw = d.get("raw_response")
            if isinstance(raw, str):
                try:
                    scores = json.loads(raw).get("segment_scores")
                except Exception:
                    scores = None
        if scores:
            for seg, value in enumerate(scores):
                out[(sid, seg)] = float(value) / 100.0 if float(value) > 1.0 else float(value)
    return out


def fallback_counts(output: Path) -> dict:
    audit = output / "audit.json"
    if not audit.exists():
        return {}
    rows = json.loads(audit.read_text()).get("observed_fallbacks", [])
    by = Counter(r["path"].split("/metrics/")[1].split("__")[0] for r in rows if "/metrics/" in r["path"])
    return {"total": len(rows), "by_metric": dict(by),
            "nudged_requests": json.loads(audit.read_text()).get("nudged_requests")}


def consistency(a: Path, b: Path) -> dict:
    pa, pb = per_sample(a), per_sample(b)
    common = sorted(set(pa) & set(pb))
    report = {"samples": len(common), "per_metric": {}, "fallbacks": {"A": fallback_counts(a), "B": fallback_counts(b)}}
    for metric in JUDGED:
        diffs = [(sid, (pb[sid][metric] - pa[sid][metric]) * 100) for sid in common if metric in pa[sid] and metric in pb[sid]]
        nonzero = [(sid, round(d, 2)) for sid, d in diffs if abs(d) > 1e-6]
        report["per_metric"][metric] = {"samples": len(diffs), "nonzero": len(nonzero),
                                        "max_abs": round(max((abs(d) for _, d in diffs), default=0.0), 2),
                                        "mean_abs": round(sum(abs(d) for _, d in diffs) / max(1, len(diffs)), 3),
                                        "nonzero_samples": nonzero[:40]}
    ea, eb = eg_segments(a), eg_segments(b)
    seg_common = sorted(set(ea) & set(eb))
    seg_diff = [(k, round((eb[k]["score"] - ea[k]["score"]) * 100, 2), ea[k]["fallback"], eb[k]["fallback"]) for k in seg_common
                if abs(eb[k]["score"] - ea[k]["score"]) > 1e-6]
    report["entity_grounding_segments"] = {"segments": len(seg_common), "differing": len(seg_diff),
                                           "differing_with_fallback_on_either_side": sum(1 for _, _, fa, fb in seg_diff if fa or fb),
                                           "examples": [{"sample": k[0], "segment": k[1], "delta": d, "A_fallback": fa, "B_fallback": fb} for k, d, fa, fb in seg_diff[:40]]}
    return report


def touched_segments(generation_root: Path, eval_data: Path) -> dict[int, set[int]]:
    """sample_id -> set of segment indices whose shot had an accepted repair."""
    scene_of = {int(r["sample_id"]): r["scene"] for r in json.loads(eval_data.read_text())}
    touched = {}
    for sid, scene in scene_of.items():
        summary = generation_root / scene / "summary.json"
        if not summary.exists():
            continue
        record = json.loads(summary.read_text())["scenes"][scene]
        touched[sid] = {int(r["shot"]) for r in record.get("accepted_repairs", [])}
    return touched


def touched(a: Path, c: Path, generation_root: Path, eval_data: Path) -> dict:
    tmap = touched_segments(generation_root, eval_data)
    report = {"samples_with_accepted_repair": sorted(sid for sid, shots in tmap.items() if shots), "metrics": {}}
    for name, loader in (("entity_grounding", lambda o: {k: v["score"] for k, v in eg_segments(o).items()}), ("vlm_score", vlm_segments)):
        sa, sc = loader(a), loader(c)
        groups = defaultdict(list)
        for key in sorted(set(sa) & set(sc)):
            sid, seg = key
            if sid not in tmap:
                continue
            shots = tmap[sid]
            if not shots:
                group = "untouched_scene"                 # delivered video identical to A
            elif seg in shots:
                group = "touched"                        # the repaired shot itself
            elif seg < min(shots):
                group = "upstream_identical"             # before the first repair: identical frames
            else:
                group = "downstream_replayed"            # after a repair: re-generated by the causal replay
            groups[group].append((sc[key] - sa[key]) * 100)
        report["metrics"][name] = {
            g: {"segments": len(v), "mean_delta": round(sum(v) / len(v), 2) if v else None,
                "positive": sum(1 for d in v if d > 1e-6), "negative": sum(1 for d in v if d < -1e-6),
                "deltas": [round(d, 1) for d in v] if g == "touched" else None}
            for g, v in groups.items()}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("consistency"); p1.add_argument("run_a", type=Path); p1.add_argument("run_b", type=Path)
    p2 = sub.add_parser("touched"); p2.add_argument("run_a", type=Path); p2.add_argument("run_c", type=Path)
    p2.add_argument("--generation-root", type=Path, required=True, help="C generation root holding <scene>/summary.json")
    p2.add_argument("--eval-data", type=Path, required=True, help="C eval_data JSON (sample_id -> scene)")
    for p in (p1, p2):
        p.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = consistency(args.run_a, args.run_b) if args.mode == "consistency" else touched(args.run_a, args.run_c, args.generation_root, args.eval_data)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
