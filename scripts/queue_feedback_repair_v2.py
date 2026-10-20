#!/usr/bin/env python3
"""Freeze feedback v2, wait for idle GPUs, run cases, and retain all outcomes."""
from __future__ import annotations

import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_feedback_repair_comparison import command, digest, snapshot_source, verify, write_json

SCENES = ["ns090_iris_stands_at_the", "ns310_at_the_flower_shop",
          "ns020_at_the_studio_equipment", "ns130_at_the_studio_equipment"]
LABELS = ("baseline", "original", "feedback v1", "feedback v2")


def prepare(args):
    out, reference = args.output, args.reference
    if (out / "manifest.json").exists():
        raise RuntimeError("Already frozen; use run/report or a new output directory")
    old = json.loads((reference / "manifest.json").read_text())
    verify(old)
    by_scene = {case["scene"]: case for case in old["cases"]}
    if not args.scenes or set(args.scenes) - set(by_scene):
        raise ValueError("Select cases with existing frozen references")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen"
    frozen.mkdir()
    for name in ("base.yaml", "entity_specs.json"):
        shutil.copy2(reference / "frozen" / name, frozen / name)
    hashes, cases = {}, []
    for scene in args.scenes:
        old_case = by_scene[scene]
        shutil.copytree(reference / "frozen/data" / scene, frozen / "data" / scene)
        baseline = Path(old_case["baseline_video"]).resolve()
        archived = {}
        for arm in ("original", "feedback"):
            root = reference / "runs" / arm / scene
            state = json.loads((root / "status.json").read_text())
            summary_path = root / scene / "summary.json"
            record = json.loads(summary_path.read_text())["scenes"][scene]
            if state["status"] != "completed" or not record.get("run_completed"):
                raise ValueError(f"Reference run is incomplete: {arm}/{scene}")
            video = str(Path(record["final_delivery_video"]).resolve())
            archived[arm] = {"video": video, "summary": str(summary_path), "wall_seconds": state["wall_seconds"],
                             **{k: record[k] for k in ("accepted_repairs", "edit_attempts", "unresolved_shots")}}
            hashes[video] = digest(video)
            hashes[str(summary_path)] = digest(summary_path)
            hashes[str(root / "status.json")] = digest(root / "status.json")
        hashes[str(baseline)] = digest(baseline)
        case = {**old_case, "baseline_video": str(baseline), "archived": archived}
        if scene.startswith("ns090"):
            case.update(focus_shots_0based=[0, 2], reason_zh="Iris 头发：重点看第 3 镜头是否改变范围、修好颜色且保留脸部。")
        cases.append(case)
    source_hashes = snapshot_source(out / "source_feedback")
    hashes.update({str(out / "source_feedback" / p): h for p, h in source_hashes.items()})
    hashes.update({str(p): digest(p) for p in frozen.rglob("*") if p.is_file()})
    manifest = {"schema_version": 2, "created_unix": time.time(), "reference_experiment": str(reference),
                "runtime_python": old["runtime_python"], "model": old["model"], "cases": cases,
                "budget": old["budget"], "source_hashes": hashes,
                "gpu_policy": {"render_min_free_mib": 75000, "judge_min_free_mib": 70000,
                               "max_utilization": 5, "stable_observations": 2, "poll_seconds": 30,
                               "require_no_compute_processes": True},
                "comparison_policy": "Run v2 only; compare with hash-verified archived baseline/original/v1 on identical frozen inputs and edit budgets.",
                "limitations": ["Four development cases selected after observing v1 failures; not a held-out benchmark",
                                "Judge acceptance and parameter changes do not establish human-perceived quality",
                                "Archived original/v1 use historical GPU assignments; timing is descriptive",
                                "Pretrained weight directories and Python environments are shared, not copied",
                                "Equal edit caps, not equal total compute; no automatic hidden retries"]}
    verify(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "progress.json", {"state": "prepared", "updated_unix": time.time(), "cases": args.scenes})
    report(out)
    print(json.dumps({"prepared": str(out), "cases": args.scenes}, ensure_ascii=False), flush=True)


def gpu_inventory():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = query(["--query-gpu=index,uuid,memory.free,utilization.gpu"])
    processes = query(["--query-compute-apps=gpu_uuid,pid"])
    occupied = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    result = []
    for line in rows.splitlines():
        index, uuid, free, utilization = [part.strip() for part in line.split(",")]
        result.append({"index": int(index), "uuid": uuid, "free_mib": int(free),
                       "utilization": int(utilization), "occupied": uuid in occupied})
    return result


def eligible_gpu(row, minimum, policy):
    return not row["occupied"] and row["free_mib"] >= minimum and row["utilization"] <= policy["max_utilization"]


def choose_pair(rows, policy):
    for render in sorted(rows, key=lambda row: (-row["free_mib"], row["index"])):
        if not eligible_gpu(render, policy["render_min_free_mib"], policy):
            continue
        for judge in sorted(rows, key=lambda row: (-row["free_mib"], row["index"])):
            if judge["index"] != render["index"] and eligible_gpu(judge, policy["judge_min_free_mib"], policy):
                return render["index"], judge["index"]
    return None


def acquire_gpu_locks(indices):
    handles = []
    try:
        # Same cooperative lock used by the earlier feedback comparison.
        for index in sorted(indices):
            handle = Path(f"/tmp/longlive_feedback_render_gpu{index}.lock").open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return []


def terminate_owned_group(child):
    """Address only a process group this worker created with start_new_session."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


class StopRequested(Exception):
    pass


class Worker:
    def __init__(self, out, manifest):
        self.out, self.manifest = out, manifest
        self.policy = manifest["gpu_policy"]
        self.child = self.server = None
        self.locks = []
        self.started = time.time()

    def checkpoint(self, state, **details):
        write_json(self.out / "progress.json", {"state": state, "pid": os.getpid(), "started_unix": self.started,
                   "updated_unix": time.time(), **details})

    def check_stop(self):
        if (self.out / "STOP").exists():
            raise StopRequested()

    def wait_pair(self, scene):
        previous, count = None, 0
        while True:
            self.check_stop()
            try:
                rows = gpu_inventory()
                pair = choose_pair(rows, self.policy)
                count = count + 1 if pair and pair == previous else 1 if pair else 0
                previous = pair
                self.checkpoint("waiting_for_gpus", next_scene=scene, gpu_inventory=rows,
                                required=self.policy, stable_observations=count)
                if count >= self.policy["stable_observations"]:
                    self.locks = acquire_gpu_locks(pair)
                    if self.locks:
                        # Recheck after acquiring cooperative locks; other users
                        # need not participate in this lock protocol.
                        checked = {row["index"]: row for row in gpu_inventory()}
                        if (eligible_gpu(checked[pair[0]], self.policy["render_min_free_mib"], self.policy)
                                and eligible_gpu(checked[pair[1]], self.policy["judge_min_free_mib"], self.policy)):
                            return pair
                        self.cleanup()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                count, previous = 0, None
                self.checkpoint("waiting_for_gpu_probe", next_scene=scene, error=str(error))
            time.sleep(self.policy["poll_seconds"])

    def start_server(self, gpu, scene):
        server_dir = self.out / "servers" / f"{scene}_{int(time.time())}"
        server_dir.mkdir(parents=True)
        cmd = [self.manifest["runtime_python"], "-u", str(self.out / "source_feedback/scripts/launch_qwen3vl_server.py"),
               "--gpu", str(gpu), "--out", str(server_dir), "--model", self.manifest["model"],
               "--mem-fraction-static", "0.45", "--startup-timeout", "1800"]
        with (server_dir / "launcher.log").open("w") as log:
            self.server = subprocess.Popen(cmd, cwd=self.out / "source_feedback", stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        write_json(server_dir / "ownership.json", {"launcher_pid": self.server.pid, "gpu": gpu, "command": cmd})
        deadline = time.time() + 1830
        while time.time() < deadline:
            self.check_stop()
            if self.server.poll() is not None:
                raise RuntimeError(f"Judge startup exited {self.server.returncode}: {server_dir}")
            if (server_dir / "base_url.txt").exists():
                return (server_dir / "base_url.txt").read_text().strip()
            self.checkpoint("starting_judge", next_scene=scene, judge_gpu=gpu, server_dir=str(server_dir))
            time.sleep(10)
        raise RuntimeError(f"Judge startup timed out: {server_dir}")

    def cleanup(self):
        for child in (self.child, self.server):
            if child is not None:
                terminate_owned_group(child)
        self.child = self.server = None
        for handle in self.locks:
            handle.close()
        self.locks = []

    def run_case(self, case, gpu, base_url):
        scene = case["scene"]
        source, root, cmd = command(self.out, self.manifest, scene, "feedback", gpu, base_url)
        verify(self.manifest)
        render_dir = root / scene / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        (render_dir / f"baseline-rank0-{scene}-seed0_regular.mp4").symlink_to(case["baseline_video"])
        start = time.time()
        state = {"scene": scene, "arm": "feedback_v2", "status": "running", "started_unix": start,
                 "render_gpu": gpu, "command": cmd, "source": str(source)}
        write_json(root / "status.json", state)
        with (root / "driver.log").open("x") as log:
            self.child = subprocess.Popen(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["pid"] = self.child.pid
        write_json(root / "status.json", state)
        while self.child.poll() is None:
            self.check_stop()
            self.checkpoint("running", scene=scene, render_gpu=gpu, child_pid=self.child.pid,
                            elapsed_seconds=time.time() - start)
            if self.server.poll() is not None:
                state.update(status="judge_failed", error=f"Judge exited {self.server.returncode}")
                terminate_owned_group(self.child)
                break
            if time.time() - start > self.manifest["budget"]["case_timeout_seconds"]:
                state["status"] = "timeout"
                terminate_owned_group(self.child)
                break
            time.sleep(10)
        code = self.child.returncode
        if state["status"] == "running":
            state["status"] = "completed" if code == 0 else "failed"
        try:
            verify(self.manifest)
        except RuntimeError as error:
            state.update(status="invalid_frozen_inputs", error=str(error))
        state.update(returncode=code, wall_seconds=time.time() - start)
        write_json(root / "status.json", state)
        print(f"{scene}: {state['status']}", flush=True)

    def run(self):
        def interrupt(signum, frame):
            raise StopRequested()
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        terminal = "completed"
        current = None
        try:
            for case in self.manifest["cases"]:
                current = case["scene"]
                status_path = self.out / "runs/feedback" / current / "status.json"
                if status_path.exists():
                    old = json.loads(status_path.read_text())
                    if old["status"] == "running":
                        raise RuntimeError(f"Unresolved prior running process; inspect before resuming: {status_path}")
                    continue
                # Release the judge between cases; never reserve it indefinitely
                # while waiting for a render card or after a failed case.
                while True:
                    render_gpu, judge_gpu = self.wait_pair(current)
                    try:
                        verify(self.manifest)
                        base_url = self.start_server(judge_gpu, current)
                        render = next(row for row in gpu_inventory() if row["index"] == render_gpu)
                        if not eligible_gpu(render, self.policy["render_min_free_mib"], self.policy):
                            self.checkpoint("waiting_for_gpus", next_scene=current, reason="render GPU became busy during judge startup")
                            continue
                        self.run_case(case, render_gpu, base_url)
                        break
                    finally:
                        self.cleanup()
                report(self.out)
            states = [json.loads(p.read_text())["status"] for p in (self.out / "runs/feedback").glob("*/status.json")]
            terminal = "completed" if len(states) == len(self.manifest["cases"]) and all(s == "completed" for s in states) else "finished_with_failures"
        except StopRequested:
            terminal = "stopped"
        except Exception as error:
            terminal = "worker_failed"
            write_json(self.out / "worker_error.json", {"scene": current, "error": f"{type(error).__name__}: {error}"})
            print(f"Worker failed: {error}", flush=True)
        finally:
            self.cleanup()
            if current:
                path = self.out / "runs/feedback" / current / "status.json"
                if path.exists():
                    state = json.loads(path.read_text())
                    if state["status"] == "running":
                        state.update(status=terminal, ended_unix=time.time())
                        write_json(path, state)
            report_error = None
            try:
                report(self.out)
            except Exception as error:
                report_error = f"{type(error).__name__}: {error}"
                terminal = "report_failed"
            self.checkpoint(terminal, report=str(self.out / "index.html"), report_error=report_error)


def report(out):
    manifest = json.loads((out / "manifest.json").read_text())
    panel_dir = out / "comparisons"
    panel_dir.mkdir(exist_ok=True)
    rows = []
    for case in manifest["cases"]:
        scene = case["scene"]
        root = out / "runs/feedback" / scene
        status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {"status": "queued"}
        summary = root / scene / "summary.json"
        if status["status"] == "completed" and summary.exists():
            record = json.loads(summary.read_text())["scenes"][scene]
            status.update({key: record.get(key) for key in ("final_delivery_video", "accepted_repairs", "unresolved_shots",
                                                           "edit_attempts", "repair_plan_calls", "repair_planning_seconds")})
        videos = [case["baseline_video"], case["archived"]["original"]["video"], case["archived"]["feedback"]["video"],
                  status.get("final_delivery_video")]
        panel = panel_dir / f"{scene}.mp4"
        if all(videos) and not panel.exists():
            graph = ";".join(f"[{i}:v]scale=640:352,setsar=1,drawtext=text='{label}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]"
                             for i, label in enumerate(LABELS))
            graph += ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
            cmd = ["ffmpeg", "-nostdin", "-v", "error"]
            for video in videos:
                cmd.extend(["-i", video])
            temporary = panel.with_suffix(".tmp.mp4")
            subprocess.run([*cmd, "-filter_complex", graph, "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "20", "-y", str(temporary)], check=True, timeout=600)
            temporary.replace(panel)
        row = {"scene": scene, "reason_zh": case["reason_zh"], "status": status, "archived": case["archived"],
               "comparison_video": str(panel) if panel.exists() else None}
        if status.get("final_delivery_video"):
            hashes = [digest(video) for video in videos]
            row["v2_identical_to"] = [label for label, value in zip(LABELS[:3], hashes[:3]) if value == hashes[3]]
        ledger = root / scene / "ledger.jsonl"
        if ledger.exists():
            events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            row["planner_decisions"] = [{"shot": event["shot_index"], "attempt": event["attempt"],
                                          "plan": event["planning"]["plan"], "actual_changes": event["planning"].get("actual_changes"),
                                          "rejected_plans": event["planning"].get("rejected_plans", [])}
                                         for event in events if event.get("event") == "repair_plan"]
        rows.append(row)
    write_json(out / "comparison.json", {"quality_claim": False, "cases": rows, "limitations": manifest["limitations"]})
    page = ["<!doctype html><meta charset='utf-8'><title>Feedback v2 comparison</title>",
            "<style>body{font:16px system-ui;margin:30px;background:#111923;color:#eee}section{padding:20px;margin:20px 0;background:#233040}video{width:100%;max-width:1280px}a{color:#9ccaff}td,th{padding:8px;text-align:left}</style>",
            f"<h1>baseline / 原版 / v1 / v2</h1><p>固定 {len(manifest['cases'])} 个开发案例；力度 1.0，每镜头最多 3 次编辑。所有失败保留。验收通过数不等于人工确认的质量提升。</p>",
            "<p><a href='progress.json'>实时状态</a> · <a href='comparison.json'>结果与规划记录</a></p>"]
    for row in rows:
        page.append(f"<section><h2>{html.escape(row['scene'])}</h2><p>{html.escape(row['reason_zh'])}</p>")
        page.append("<table><tr><th>版本</th><th>状态</th><th>编辑次数</th><th>通过修复数</th><th>未解决镜头（从 0 开始）</th><th>秒</th></tr>")
        for label, state in (("original", row["archived"]["original"]), ("v1", row["archived"]["feedback"]), ("v2", row["status"])):
            edits = sum(state.get("edit_attempts", {}).values()) if "edit_attempts" in state else "—"
            accepted = len(state["accepted_repairs"]) if "accepted_repairs" in state else "—"
            page.append(f"<tr><td>{label}</td><td>{html.escape(state.get('status', 'archived'))}</td><td>{edits}</td><td>{accepted}</td><td>{html.escape(str(state.get('unresolved_shots', '—')))}</td><td>{round(state.get('wall_seconds', 0))}</td></tr>")
        page.append("</table>")
        if row["comparison_video"]:
            page.append(f"<video controls preload='metadata' src='comparisons/{html.escape(row['scene'])}.mp4'></video>")
            page.append(f"<p>v2 与哪些版本字节相同：{html.escape(str(row.get('v2_identical_to', [])))}</p>")
        page.append("</section>")
    (out / "index.html").write_text("\n".join(page))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "videos/feedback_repair_20260914")
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    args = parser.parse_args()
    args.output, args.reference = args.output.resolve(), args.reference.resolve()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "report":
        report(args.output)
    else:
        with (args.output / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = json.loads((args.output / "manifest.json").read_text())
            verify(manifest)
            Worker(args.output, manifest).run()


if __name__ == "__main__":
    main()
