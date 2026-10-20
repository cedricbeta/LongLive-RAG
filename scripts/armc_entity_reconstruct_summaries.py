"""Rebuild per-scene summary.json for arm-C entity scenes whose driver crashed after the scene
finished (all 6 shots committed) but before the group-level summary was written.

Delivery-video rule mirrors EditRenderBackend.commit(): active_video advances only on a
non-quarantined commit whose attempt >= 1 was produced by an edit_shot action (the action's
result.edited_video names that attempt); quarantined commits and plain reseeds keep it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "videos/narrastream_threearm_20260827/armC_entity"
NUM_SHOTS = 6


def reconstruct(scene_dir: Path) -> dict | None:
    scene = scene_dir.name
    ledger = scene_dir / "ledger.jsonl"
    if not ledger.exists():
        return None
    rows = [json.loads(l) for l in ledger.open()]
    commits = [r for r in rows if r.get("event") == "commit"]
    if len(commits) < NUM_SHOTS:
        return None
    renders = scene_dir / "renders"
    baseline = sorted(renders.glob(f"baseline-rank*-{scene}-seed0_regular.mp4"))
    if len(baseline) != 1:
        return None
    active = baseline[0]
    last_edit: dict[int, list[str]] = {}
    delivered, edited_commits = [], []
    for r in rows:
        if r.get("event") == "action" and r.get("tool") == "edit_shot":
            ev = str((r.get("result") or {}).get("edited_video", ""))
            last_edit.setdefault(int(r["shot_index"]), []).append(ev)
        elif r.get("event") == "commit":
            shot, attempt, q = int(r["shot_index"]), int(r["attempt"]), bool(r["quarantined"])
            edited = False
            if attempt >= 1 and not q:
                for ev in last_edit.get(shot, []):
                    if f"edit_shot{shot:02d}_attempt{attempt}-" in ev and Path(ev).exists():
                        active = Path(ev)
                        edited = True
            rec = {"shot": shot, "attempt": attempt, "quarantined": q, "video": str(active), "edited": edited}
            delivered.append(rec)
            if edited:
                edited_commits.append(rec)
    return {
        "baseline_video": str(baseline[0]),
        "final_delivery_video": str(active),
        "ledger": str(ledger),
        "two_up": str(scene_dir / "two_up.mp4"),
        "shots": len(commits),
        "quarantined_shots": [c["shot"] for c in delivered if c["quarantined"]],
        "edited_commits": edited_commits,
        "delivered": delivered,
        "passed": True,
        "reconstructed_from_ledger": True,
    }


def main() -> None:
    written = 0
    for scene_dir in sorted(OUT.glob("g*/ns*")):
        if (scene_dir / "summary.json").exists():
            continue
        rec = reconstruct(scene_dir)
        if rec is None:
            print(f"skip {scene_dir.relative_to(OUT)} (incomplete)")
            continue
        (scene_dir / "summary.json").write_text(
            json.dumps({"schema_version": 1, "scenes": {scene_dir.name: rec}}, indent=2, sort_keys=True) + "\n"
        )
        print(f"{scene_dir.relative_to(OUT)}: delivery={Path(rec['final_delivery_video']).name} quarantined={rec['quarantined_shots']}")
        written += 1
    print(f"{written} summaries written")


if __name__ == "__main__":
    sys.exit(main())
