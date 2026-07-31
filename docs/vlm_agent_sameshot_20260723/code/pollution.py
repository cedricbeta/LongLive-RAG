"""Retrieval-pollution diagnostic.

For the latentmem arm (observe-only VLM labels), measure how often the
similarity retriever selects blocks the VLM labeled untrustworthy, and how
that rate evolves over the video. For the vlm_guided arm, report gate and
golden-anchor activity.
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="arms_out")
parser.add_argument("--prompt_indices", default="0,1,2,6")
args = parser.parse_args()

for pi in [int(i) for i in args.prompt_indices.split(",")]:
    print(f"\n===== prompt {pi} =====")
    for arm in ["latentmem", "vlm_guided"]:
        path = os.path.join(args.root, arm, f"prompt{pi:03d}_log.json")
        if not os.path.exists(path):
            continue
        log = json.load(open(path))
        blocks = [e for e in log["vlm_log"] if "frames" in e]
        bad_frames = set()
        for e in blocks:
            if not e.get("admit", True):
                bad_frames.update(e["frames"])
        n_rej = sum(1 for e in blocks if not e.get("admit", True))
        print(f"[{arm}] blocks={len(blocks)} rejected={n_rej} "
              f"({100*n_rej/max(len(blocks),1):.0f}%) vlm_calls={log['vlm_calls']}")
        if arm == "vlm_guided":
            print(f"  golden={log['golden_frames']}")

        mem = log["memory_log"]
        if not mem:
            continue
        half = len(mem) // 2
        for name, seg in [("first half", mem[:half]), ("second half", mem[half:]), ("all", mem)]:
            tot = hit = 0
            for m in seg:
                sel = m["selected_global_frames"][0]
                tot += len(sel)
                hit += sum(1 for f in sel if f in bad_frames)
            if tot:
                print(f"  {name:12s}: {hit}/{tot} retrieved slots point at "
                      f"VLM-rejected blocks ({100*hit/tot:.1f}%)")
        if arm == "vlm_guided":
            forced_used = sum(len(m.get("forced_global_frames", [])) for m in mem)
            print(f"  forced golden slots used: {forced_used} across {len(mem)} retrieval events")
        rej_by_block = [(e['block_start_frame'], e.get('reason', '')) for e in blocks if not e.get('admit', True)]
        for bf, reason in rej_by_block[:6]:
            print(f"    rejected@frame{bf}: {reason}")
