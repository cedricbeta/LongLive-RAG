#!/usr/bin/env python
"""Turn a judge-only pass over baseline videos into a per-shot defect table.

Reads every <root>/<scene>/ledger.jsonl written by run_agent_edit_demo.py with
--max_edits_per_shot 0 and lists, per scene and shot, the attempt-0 verdict:
reason, each expected entity's assignment/appearance status, which colour
parts the pixel veto flagged, and the failure type it implies.  Writes
defects.json plus a contact sheet per flagged shot (baseline mid-frame with the
entity boxes) for human spot checks.  This is a machine-proposed label set,
not ground truth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def failure_type(reason: str, entities: dict) -> str:
    if reason.startswith("accept"):
        return "none" if reason == "accept_entities_match" else "unknown_assignment"
    if reason in ("reject_entity_drift", "reject_identity_reference_low_confidence"):
        return "appearance_drift"
    if reason.startswith("reject_entity_absent"):
        return "absence"
    if reason == "reject_entity_assignment_uncertain":
        return "unknown_assignment"
    if reason == "reject_unassigned_subject":
        return "extra_subject"
    return reason


def summarize(root: Path) -> dict:
    rows, counts = [], Counter()
    for ledger in sorted(root.glob("*/ledger.jsonl")):
        scene = ledger.parent.name
        for line in ledger.read_text().splitlines():
            d = json.loads(line)
            if d.get("event") != "verifier" or int(d.get("attempt", -1)) != 0:
                continue
            m = d.get("metrics", {})
            ents = {}
            for name, e in m.get("entities", {}).items():
                if not e.get("expected"):
                    continue
                parts = []
                for vote in e.get("pixel_color_votes", []) or []:
                    parts += [r for r, c in vote.items() if c]
                ents[name] = {
                    "present": bool(e.get("located_chunks")), "matches": e.get("matches"),
                    "uncertain": bool(e.get("assignment_uncertain")), "confidence": e.get("confidence"),
                    "basis": e.get("comparison_basis"), "differences": e.get("differences", "")[:160],
                    "vetoed_parts": sorted(set(parts)),
                }
            kind = failure_type(d.get("reason", ""), ents)
            counts[kind] += 1
            rows.append({"scene": scene, "shot": int(d["shot_index"]), "reason": d.get("reason"), "failure_type": kind,
                         "unknown_roles": m.get("accepted_with_unknown_assignment", []), "entities": ents})
    return {"root": str(root), "shots": len(rows), "failure_types": dict(counts), "rows": rows}


def contact_sheets(root: Path, rows: list, out_dir: Path, fpc: int = 8, shot_chunks: int = 8) -> None:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = {p.parent.parent.name: p for p in root.glob("*/renders/baseline-rank*_regular.mp4")}
    for row in rows:
        if row["failure_type"] == "none":
            continue
        video = videos.get(row["scene"])
        if video is None:
            continue
        cap = cv2.VideoCapture(str(video))
        index = row["shot"] * shot_chunks * fpc + (shot_chunks * fpc) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        label = f"{row['scene'][:20]} shot{row['shot']} {row['failure_type']}"
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y = 56
        for name, e in row["entities"].items():
            text = f"{name}: {'present' if e['present'] else 'absent'}{' uncertain' if e['uncertain'] else ''} matches={e['matches']} veto={','.join(e['vetoed_parts']) or '-'}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 26
        cv2.imwrite(str(out_dir / f"{row['scene']}_shot{row['shot']}_{row['failure_type']}.png"), frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="output root of the judge-only demo run (one dir per scene)")
    ap.add_argument("--output", type=Path, help="defects.json path (default <root>/defects.json)")
    ap.add_argument("--sheets", action="store_true", help="also write contact sheets for flagged shots")
    args = ap.parse_args()
    report = summarize(args.root)
    target = args.output or args.root / "defects.json"
    target.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    if args.sheets:
        contact_sheets(args.root, report["rows"], args.root / "defect_sheets")
    print(json.dumps({"shots": report["shots"], "failure_types": report["failure_types"], "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
