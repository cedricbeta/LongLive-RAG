"""Opt-in CLI smoke against our already-running loopback judge; no cloud calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("BENCHMARK_LIVE_BASE_URL"), reason="Opt-in local judge test")
def test_audited_cli_completes_and_saves_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    inputs = json.loads((root / "videos/benchmark_20260909/evaluation33_v3/inputs/A.json").read_text())
    data = tmp_path / "sample.json"
    data.write_text(json.dumps(inputs[:1]))
    output = tmp_path / "scores"
    command = [sys.executable, str(root / "scripts/run_audited_narrastream_eval.py"),
        "--eval-data", str(data), "--output", str(output), "--cache-root", str(tmp_path / "cache"),
        "--base-url", os.environ["BENCHMARK_LIVE_BASE_URL"], "--expected-count", "1",
        "--device", "cpu", "--api-workers", "1", "--metrics", "subject_consistency"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2"})
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    audit = json.loads((output / "audit.json").read_text())
    scores = json.loads((output / "results_latest.json").read_text())
    assert audit["status"] == "metrics_completed"
    assert audit["valid_for_quality_claims"] is False
    assert scores["aggregated"]["metric_coverage"]["subject_consistency"]["valid_count"] == 1
    assert audit["observed_fallbacks"] == []
    assert (output / "api_events.jsonl").is_file()
