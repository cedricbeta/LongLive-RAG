#!/usr/bin/env python3
"""Build the four preregistered "last-stand" appended-KV arms.

The attention down-weighting study (docs/attention-downweighting-study.zh.md)
left the appended-injection route two cheap, mechanistically grounded targets:
inject only while the memory is still consumed (denoise steps 0-1; L29 step
decay 4x), target the L29 memory heads found by the E5 census (18-30x
per-column mass), and store the L14 memory K in the query's noisy phase
(E4: clean-vs-step0 cosine 0.43). This script emits one config per arm:

    {bearded_watchmaker_workshop, sandy_beach_driftwood}
        x {laststand_full, baseline_rerun}, seed 0

``laststand_full`` turns all three utils/kv_rag.py knobs on
(inject_active_steps=[0,1], inject_head_filter={29: E5 top-2 heads},
store_phase_matched_layers=[14]); ``baseline_rerun`` re-renders the identical
KV injection plan with every knob off, so the pair is a paired comparison.

Baseline discovery mirrors scripts/build_capsule_channel_test.py: prefer the
frozen replay config that already carries the exact kv_only injection plan
(the attn-probe replay config, then the capsule-pilot probe config), and fall
back to the repo-committed kv_only plan with a sample_names scene filter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

SCENES = ("bearded_watchmaker_workshop", "sandy_beach_driftwood")
ARMS = ("laststand_full", "baseline_rerun")

# E5 head census: layer-29 "memory heads" (per-column mass / pinned reference).
# Computed from docs/figures/attention_downweighting/<scene>_analysis_summary
# .json when present; both scenes agree on heads {23, 1} (watchmaker 46x/26x,
# driftwood 30x/19x), so this default is a faithful stand-in when the JSON is
# unavailable on the render host.
DEFAULT_L29_MEMORY_HEADS = [1, 23]
MEMORY_HEAD_LAYER = 29
PHASE_MATCHED_LAYERS = [14]
ACTIVE_STEPS = [0, 1]

# Keys of the read-only attn probe; stripped from reused replay configs so the
# render arms do not re-dump 26GB of probe tensors.
ATTN_PROBE_KEYS = (
    "attn_probe_dir",
    "attn_probe_max_q_rows",
    "attn_probe_full_q_first_step",
    "attn_probe_store_k",
)


def top2_memory_heads(figures_dir: Path, scene: str) -> tuple[list[int], str]:
    """Top-2 layer-29 heads from the E5 census JSON, with provenance."""
    summary_path = figures_dir / f"{scene}_analysis_summary.json"
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        ratios = data["e5_head_ratio"][str(MEMORY_HEAD_LAYER)]
        order = sorted(range(len(ratios)), key=lambda i: -float(ratios[i]))
        heads = sorted(int(i) for i in order[:2])
        return heads, str(summary_path.relative_to(ROOT))
    except Exception:
        return list(DEFAULT_L29_MEMORY_HEADS), (
            f"default {DEFAULT_L29_MEMORY_HEADS} (E5 JSON missing at "
            f"{summary_path}; both scenes' census agreed on these heads)"
        )


def find_baseline_config(scene: str) -> tuple[dict, str]:
    """Load the per-scene baseline config, mirroring the channel-test lookup."""
    candidates = (
        ROOT / f"videos/attn_probe_20260820/configs/{scene}.yaml",
        ROOT / f"videos/capsule_pilot_20260715/configs/probe_seed0__{scene}.yaml",
    )
    for path in candidates:
        if path.exists():
            return (
                yaml.safe_load(path.read_text(encoding="utf-8")),
                str(path.relative_to(ROOT)),
            )
    # Repo-committed fallback: the kv_only long-multishot plan, filtered to the
    # scene via the sample_names dataset filter.
    path = ROOT / "configs/inference_kv_rag_long_multishot.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.setdefault("data", {})["data_path"] = "example/benchmark_hard_multishot"
    config["data_path"] = "example/benchmark_hard_multishot"
    return config, str(path.relative_to(ROOT))


def kv_rag_plan(config: dict) -> dict:
    """The kv_only injection plan the probe replays ran (lambda=0)."""
    plan = (config.get("inference") or {}).get("kv_rag")
    if not (isinstance(plan, dict) and plan.get("enabled")):
        # Probe-era configs (capsule pilot) disabled kv_rag; restore the
        # committed kv_only plan so every arm injects the same way.
        base = yaml.safe_load(
            (ROOT / "configs/inference_kv_rag_long_multishot.yaml").read_text(
                encoding="utf-8"
            )
        )
        plan = base["inference"]["kv_rag"]
    plan = deepcopy(plan)
    for key in ATTN_PROBE_KEYS:
        plan.pop(key, None)
    plan["persistent_logit_bias_lambda"] = 0.0  # lambda=0, as in the replays
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="videos/laststand_20260827")
    parser.add_argument(
        "--figures_dir", default="docs/figures/attention_downweighting"
    )
    args = parser.parse_args()
    out_root = (ROOT / args.out_root).resolve()
    figures_dir = (ROOT / args.figures_dir).resolve()
    configs_dir = out_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in SCENES:
        base, base_source = find_baseline_config(scene)
        plan = kv_rag_plan(base)
        heads, heads_source = top2_memory_heads(figures_dir, scene)
        for arm in ARMS:
            arm_id = f"{scene}__{arm}"
            config = deepcopy(base)
            render_dir = out_root / f"renders/{arm_id}"
            config["output_folder"] = str(render_dir.relative_to(ROOT))
            config["sample_names"] = [scene]
            config.setdefault("logging", {})["seed"] = 0
            inference = config.setdefault("inference", {})
            inference["output_folder"] = str(render_dir.relative_to(ROOT))
            inference["filename_prefix"] = f"laststand_{arm}"
            inference["filename_from_sample_name"] = True
            # This round renders the appended-KV route only.
            if "capsule_probe" in inference:
                inference["capsule_probe"] = {"enabled": False}
            if "capsule_edit" in inference:
                inference["capsule_edit"] = {"enabled": False}
            arm_plan = deepcopy(plan)
            if arm == "laststand_full":
                arm_plan["inject_active_steps"] = list(ACTIVE_STEPS)
                arm_plan["inject_head_filter"] = {MEMORY_HEAD_LAYER: list(heads)}
                arm_plan["store_phase_matched_layers"] = list(PHASE_MATCHED_LAYERS)
            inference["kv_rag"] = arm_plan
            config_path = configs_dir / f"laststand__{arm_id}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, width=1000),
                encoding="utf-8",
            )
            records.append({
                "arm_id": arm_id,
                "scene": scene,
                "arm": arm,
                "seed": 0,
                "config": str(config_path.relative_to(ROOT)),
                "render_dir": str(render_dir.relative_to(ROOT)),
                "baseline_config_source": base_source,
                "inject_active_steps": (
                    list(ACTIVE_STEPS) if arm == "laststand_full" else None
                ),
                "inject_head_filter": (
                    {str(MEMORY_HEAD_LAYER): list(heads)}
                    if arm == "laststand_full" else None
                ),
                "store_phase_matched_layers": (
                    list(PHASE_MATCHED_LAYERS) if arm == "laststand_full" else None
                ),
                "memory_heads_source": heads_source,
                "command": (
                    "CUDA_VISIBLE_DEVICES=<gpu> "
                    "PYTORCH_ALLOC_CONF=expandable_segments:True "
                    f"python inference.py --config_path {config_path.relative_to(ROOT)}"
                ),
                "status": "not_started",
            })

    manifest = {
        "schema_version": 1,
        "track": "appended_kv_laststand",
        "preregistered_arms": len(records),
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "knobs": {
            "inject_active_steps": ACTIVE_STEPS,
            "inject_head_filter_layer": MEMORY_HEAD_LAYER,
            "store_phase_matched_layers": PHASE_MATCHED_LAYERS,
        },
        "mechanism_basis": {
            "active_steps": "L29 injected-mass decays 0.0138->0.0034 over steps 0-3",
            "head_filter": "E5 census: two L29 heads at 18-30x per-column mass",
            "phase_matched": "E4: L14 chunk-K cosine(step3, step0) = 0.43",
        },
        "kv_injection_plan": "same kv_only lambda=0 plan as the probe replays / channel test",
        "runs": records,
    }
    manifest_path = out_root / "laststand_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "planned_renders": len(records),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "configs": [r["config"] for r in records],
    }, indent=2))


if __name__ == "__main__":
    main()
