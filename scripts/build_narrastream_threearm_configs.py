#!/usr/bin/env python3
"""Generate per-scene configs for the NarraStream three-arm comparison.

Pre-registration: docs/laststand-and-narrastream-prereg.zh.md.

  arm A  baseline      — native rollout + sink, kv_rag fully disabled
  arm B  concat inject — KV-RAG scene-memory retrieval + appended columns
                          (the mechanism-class proxy for IAMFlow/MemFlow)
  arm C  repair loop   — NOT built here: driven by scripts/run_agent_edit_demo.py
                          (judge -> edit_shot -> re-verify), which renders its own
                          deterministic baseline identical to arm A.

Configs derive from the laststand baseline yaml (itself derived from
configs/inference_kv_rag_long_multishot.yaml) so KV plan/layers stay identical
to the pre-registered arm; only data source, length and kv_rag.enabled differ.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "videos/laststand_20260827/configs/laststand__sandy_beach_driftwood__baseline_rerun.yaml"
SUBSET = ROOT / "example/narrastream_bench/multitext_subset33_b8"
BLOCKS_PER_SCENE = 48  # 6 segments x 8 blocks
FRAMES = BLOCKS_PER_SCENE * 8  # 384 pixel frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="videos/narrastream_threearm_20260827")
    ap.add_argument("--subset", default=str(SUBSET))
    args = ap.parse_args()

    subset = Path(args.subset)
    out_root = ROOT / args.out_root
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    template = yaml.safe_load(TEMPLATE.read_text())
    scenes = sorted(
        d.name for d in subset.iterdir() if d.is_dir() and (d / "0.json").exists()
    )

    manifest = {"template": str(TEMPLATE), "subset": str(subset), "scenes": scenes, "arms": {}}
    for arm in ("armA_baseline", "armB_concat"):
        written = []
        for scene in scenes:
            cfg = copy.deepcopy(template)
            cfg["data"]["data_path"] = str(subset.relative_to(ROOT))
            cfg["num_output_frames"] = FRAMES
            shape = cfg["data"]["image_or_video_shape"]
            shape[1] = FRAMES
            cfg["sample_names"] = [scene]
            # The pipeline reads inference.output_folder / inference.filename_prefix;
            # the top-level output_folder is set too for tooling that keys off it.
            # (First campaign run lost an arm per scene to this: only the top-level
            # key was rewritten, so all 66 renders collided in the template's
            # inference.output_folder under identical filenames.)
            out_folder = f"{args.out_root}/renders/{scene}__{arm}"
            cfg["output_folder"] = out_folder
            cfg["inference"]["output_folder"] = out_folder
            cfg["inference"]["filename_prefix"] = arm
            cfg["inference"]["kv_rag"].pop("attn_probe_dir", None)
            for knob in (
                "inject_active_steps",
                "inject_head_filter",
                "store_phase_matched_layers",
            ):
                cfg["inference"]["kv_rag"].pop(knob, None)
            if arm == "armA_baseline":
                cfg["inference"]["kv_rag"]["enabled"] = False
            name = f"ns3arm__{scene}__{arm}.yaml"
            (cfg_dir / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(name)
        manifest["arms"][arm] = written

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {sum(len(v) for v in manifest['arms'].values())} configs "
          f"for {len(scenes)} scenes to {cfg_dir}")


if __name__ == "__main__":
    main()
