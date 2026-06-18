#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Closed-judge and Qwen3 pre-gate validation for the VLM consistency loop.

The objective requires the closed-source judge to be validated before it gates
any optimizer loop. This command stops fail-closed when the reused-OAuth judge
or the local Qwen3 optimizer cannot validate; it never falls back to a local
judge or Qwen2.5 optimizer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import (  # noqa: E402
    build_fail_closed_ledger,
    run_oauth_judge_validation,
    validate_qwen3_sglang_server,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="docs/closed_judge_loop")
    parser.add_argument("--judge_provider", default="auto", choices=("auto", "claude", "codex"))
    parser.add_argument("--claude_model", default="claude-opus-4-8")
    parser.add_argument("--codex_model", default="gpt-5.5")
    parser.add_argument("--claude_credentials_path", default=None)
    parser.add_argument("--sglang_python", default=None)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen_startup_timeout", type=int, default=300)
    parser.add_argument("--qwen_mem_fraction_static", type=float, default=0.45)
    parser.add_argument(
        "--numeric_metrics_json",
        default="docs/multiview_gate_results/round14_scene_memory_mechanism_sweep.json",
        help="Optional prior numeric metric JSON recorded in the fail-closed ledger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    validation = run_oauth_judge_validation(
        output_dir=output_root,
        primary_model=args.claude_model,
        codex_model=args.codex_model,
        provider=args.judge_provider,
        claude_credentials_path=args.claude_credentials_path,
    )
    validation["closed_judge_validated"] = bool(validation.get("passed"))

    optimizer_validation = None
    if validation.get("passed"):
        optimizer_validation = validate_qwen3_sglang_server(
            output_dir=output_root / "qwen3_optimizer",
            sglang_python=args.sglang_python,
            model=args.qwen_model,
            mem_fraction_static=args.qwen_mem_fraction_static,
            startup_timeout=args.qwen_startup_timeout,
        )
        validation["optimizer_validation"] = optimizer_validation
        if optimizer_validation.get("blocked_reason"):
            validation["passed"] = False
            validation["blocked_reason"] = (
                "Qwen3 optimizer failed validation: "
                + str(optimizer_validation.get("blocked_reason"))
            )

    validation["wall_clock_seconds"] = time.time() - started
    validation_path = output_root / "judge_validation.json"
    write_json(validation_path, validation)

    ledger = build_fail_closed_ledger(
        validation=validation,
        output_dir=output_root,
        numeric_metrics_path=args.numeric_metrics_json,
    )
    ledger["resources"]["wall_clock_seconds"] = validation["wall_clock_seconds"]
    if optimizer_validation is not None:
        ledger["separation"]["optimizer_probe"] = optimizer_validation
        ledger["resources"]["gpu_time_seconds"] = optimizer_validation.get("gpu_time_seconds", 0)
    ledger["judge_validation"]["oauth_only"] = True
    ledger["judge_validation"]["paid_api_key_used"] = False
    ledger["judge_validation"]["closed_judge_validated"] = bool(validation.get("closed_judge_validated"))
    ledger["ablation"]["evaluated"] = False
    ledger["ablation"]["reason"] = validation.get("blocked_reason") or (
        "pre-gates passed, but full generation ablation is not run by this validation-only command"
    )
    ledger_path = output_root / "ledger.json"
    write_json(ledger_path, ledger)

    if not validation.get("passed"):
        print(f"[closed-judge] BLOCKED fail-closed: {validation.get('blocked_reason')}")
        print(f"[closed-judge] wrote {validation_path.resolve()}")
        print(f"[closed-judge] wrote {ledger_path.resolve()}")
        return

    # Keep this explicit: after validation passes the next stage must run the
    # optimizer/generation ablation with a separate local VLM transcript.
    print("[closed-judge] judge and Qwen3 optimizer validation passed; generation ablation is now admissible.")
    print(f"[closed-judge] wrote {validation_path.resolve()}")
    print(f"[closed-judge] wrote {ledger_path.resolve()}")


if __name__ == "__main__":
    main()
