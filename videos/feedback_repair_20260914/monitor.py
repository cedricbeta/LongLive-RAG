"""Refresh the prepared comparison and release only this campaign's judge."""
import json
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def alive(pid):
    try:
        status = Path(f"/proc/{int(pid)}/stat").read_text()
        return status.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


campaign = int((OUT / "campaign.pid").read_text())
report = [str(ROOT / "venv/bin/python"),
          str(OUT / "source_feedback/scripts/run_feedback_repair_comparison.py"),
          "report", "--output", str(OUT)]
while True:
    statuses = [json.loads(p.read_text()) for p in (OUT / "runs").glob("*/*/status.json")]
    active = alive(campaign) or any(s.get("status") == "running" and alive(s.get("pid", 0)) for s in statuses)
    if not active:
        break
    # Full gallery creation is deferred until the campaign's own report has
    # finished, avoiding concurrent writes or duplicate video encoding.
    (OUT / "progress.json").write_text(json.dumps({
        "updated_unix": time.time(), "statuses": [
            {k: s.get(k) for k in ("scene", "arm", "status", "wall_seconds")}
            for s in statuses
        ]}, indent=2) + "\n")
    time.sleep(30)
try:
    result = subprocess.run(report, cwd=ROOT)
    (OUT / "monitor_status.json").write_text(json.dumps({"campaign_ended": True, "report_returncode": result.returncode}) + "\n")
finally:
    # launch_qwen3vl_server watches this run-specific path. No other GPU
    # process or service is addressed.
    (OUT / "server/STOP").touch()
