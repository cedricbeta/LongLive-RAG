"""Start one SGLang Qwen3-VL server and keep it up until a STOP file appears.

Unlike scripts/run_agent_edit_demo.start_vlm_server this does not cap the
static memory fraction at 0.45 and does not require model_type == qwen3_vl,
so it can serve the 30B-A3B mixture model (model_type qwen3_vl_moe, ~58 GB of
weights) used for the judge calibration.  Writes <out>/base_url.txt once
/v1/models answers, <out>/EXITED when the server is gone.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.closed_judge import find_free_port  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--mem-fraction-static", type=float, default=0.45)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--sglang-python", default="/home/chendong/miniconda3/envs/sglang/bin/python")
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--revision")
    ap.add_argument("--context-length", type=int)
    ap.add_argument("--max-running-requests", type=int)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("STOP", "EXITED", "base_url.txt"):
        (out / stale).unlink(missing_ok=True)
    port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    cmd = [
        args.sglang_python, "-m", "sglang.launch_server",
        "--model-path", args.model, "--host", "127.0.0.1", "--port", str(port),
        "--enable-multimodal", "--trust-remote-code",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--tp", str(args.tp),
        "--served-model-name", args.model,
        "--constrained-json-disable-any-whitespace",
    ]
    for flag, value in (("--revision", args.revision), ("--context-length", args.context_length),
                        ("--max-running-requests", args.max_running_requests)):
        if value is not None:
            cmd.extend([flag, str(value)])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    log = (out / "sglang.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    (out / "server.pid").write_text(str(proc.pid))
    deadline = time.time() + args.startup_timeout
    ready = False
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5):
                ready = True
                break
        except Exception:
            time.sleep(5)
    if not ready:
        if proc.poll() is None:
            proc.terminate()
        (out / "EXITED").write_text("startup failed")
        sys.exit(1)
    (out / "base_url.txt").write_text(base_url)
    print(base_url, flush=True)
    try:
        while not (out / "STOP").exists() and proc.poll() is None:
            time.sleep(10)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
        (out / "EXITED").write_text(f"exited {proc.returncode}")


if __name__ == "__main__":
    main()
