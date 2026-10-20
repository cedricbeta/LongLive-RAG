#!/usr/bin/env python3
"""Run the existing NarraStream metrics with isolated caches and API provenance.

This is an iteration protocol, not the paper's 30B/27B evaluator. Metric code,
prompts, normalization and fallback policies are unchanged. The API boundary
adds seed=0 and content-addressed responses, independently of the controller.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def redact_images(value):
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_sha256": hashlib.sha256(value.encode()).hexdigest(), "encoded_bytes": len(value)}
    if isinstance(value, list):
        return [redact_images(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_images(item) for key, item in value.items()}
    return value


def evaluation_aggregated(results):
    """The API returns aggregates; saved snapshots wrap them in 'aggregated'."""
    if not isinstance(results, dict):
        raise TypeError("Evaluator must return a mapping")
    aggregate = results.get("aggregated", results)
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("metric_coverage"), dict):
        raise ValueError("Evaluator result lacks metric coverage")
    return aggregate


MAX_NUDGE = 3
NUDGE_ERROR_MARKERS = ("entity grounding payload", "Empty planner response")


def _nudge_worthy(error):
    """Parser rejections that a changed prompt can fix (empty/invalid extractor JSON)."""
    text = str(error)
    return any(marker in text for marker in NUDGE_ERROR_MARKERS)


def nudge_text(level):
    # No "empty list" escape hatch: the first wording offered one and the planner
    # took it for 39 of 55 nudged prompts (all of which name characters), turning
    # parse-error defaults into empty-entity defaults (2026-09-11).
    return (
        f"\n\nFormat reminder {level}: the previous reply was not a valid answer. Return exactly one JSON "
        "object with this shape: {\"entities\": [{\"name\": \"<entity name as written in the text>\", "
        "\"attributes\": [\"<visible attribute>\", \"...\"]}, ...]}. Every named person, animal or object in "
        "the text must appear once in \"entities\" with its visible attributes (hair, clothing, colours, held "
        "objects). Use exactly the keys \"entities\", \"name\", \"attributes\". Never return an empty "
        "object, empty keys, or an empty \"entities\" array when the text names anyone or anything."
    )


def nudged_request(kwargs, level):
    """Append the format reminder to the last user message (string or parts list)."""
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return kwargs
    messages = [dict(message) if isinstance(message, dict) else message for message in messages]
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + nudge_text(level)
        elif isinstance(content, list):
            message["content"] = list(content) + [{"type": "text", "text": nudge_text(level)}]
        else:
            return kwargs
        break
    return {**kwargs, "messages": messages}


def install_audited_client(api_clients, cache_root, events_path, api_retry=None):
    """Patch only this process's API factory; never modify the external checkout."""
    from openai.types.chat import ChatCompletion

    original = api_clients.build_openai_compatible_client
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    transactions = threading.local()

    def event(payload):
        with events_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False) + "\n")
            stream.flush()

    if api_retry is not None:
        original_retry = api_retry.run_with_retry

        def audited_retry(operation, **retry_kwargs):
            # The entity-grounding extractor is deterministic at temperature 0:
            # when the planner answers {"": ""} once, an identical retry answers
            # the same way, and after the retry budget the segment silently
            # scores 0.5 (42 of 198 segments in the 2026-09-09 A run).  Only a
            # changed request can recover, so each parser rejection of that
            # kind escalates a format reminder that audited_create appends to
            # the next attempt's prompt.  The reminder level is part of the
            # request, hence of the cache key and the audit trail.
            state = {"nudge": 0}

            def attempt():
                # A successful HTTP response is not yet a valid metric payload.
                # Publish only after the existing metric parser accepts it.
                previous = getattr(transactions, "pending", None)
                previous_nudge = getattr(transactions, "nudge", 0)
                pending = []
                transactions.pending = pending
                transactions.nudge = state["nudge"]
                try:
                    result = operation()
                    for path, payload, lock in pending:
                        if not payload.get("truncated"):
                            atomic_json(path, payload)
                    return result
                except Exception as error:
                    for path, payload, lock in pending:
                        rejected = cache_root / "rejected" / f"{path.stem}_{time.time_ns()}.json"
                        atomic_json(rejected, {**payload, "parser_error_type": type(error).__name__})
                        event({"key": path.stem, "parser_rejected": True,
                               "error_type": type(error).__name__, "nudge_level": state["nudge"]})
                    if _nudge_worthy(error):
                        state["nudge"] = min(state["nudge"] + 1, MAX_NUDGE)
                    raise
                finally:
                    for path, payload, lock in pending:
                        lock.close()
                    transactions.pending = previous
                    transactions.nudge = previous_nudge
            return original_retry(attempt, **retry_kwargs)

        api_retry.run_with_retry = audited_retry

    def factory(service):
        # Only local endpoints are permitted: no accidental billable/cloud requests.
        from urllib.parse import urlparse
        if urlparse(service.base_url or "").hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This iteration runner permits loopback API endpoints only")
        client = original(service)
        create = client.chat.completions.create

        def audited_create(**kwargs):
            kwargs = {**kwargs, "seed": 0}
            nudge = int(getattr(transactions, "nudge", 0) or 0)
            if nudge > 0:
                kwargs = nudged_request(kwargs, nudge)
            if kwargs.get("stream"):
                raise ValueError("Streaming completions are not supported by the audit cache")
            key = request_key(kwargs)
            cached_path = cache_root / f"{key}.json"
            lock = (cache_root / f"{key}.lock").open("a")
            deferred = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if cached_path.exists():
                    payload = json.loads(cached_path.read_text())
                    if payload["request_sha256"] != key:
                        raise ValueError("Request cache fingerprint mismatch")
                    event({"key": key, "cache_hit": True, "nudge_level": nudge})
                    return ChatCompletion.model_validate(payload["response"])
                started = time.monotonic()
                try:
                    response = create(**kwargs)
                except Exception as error:
                    event({"key": key, "cache_hit": False, "error_type": type(error).__name__,
                           "wall_seconds": time.monotonic() - started})
                    raise
                data = response.model_dump(mode="json")
                truncated = any(choice.get("finish_reason") == "length" for choice in data.get("choices", []))
                payload = {"request_sha256": key, "request": redact_images(kwargs),
                           "response": data, "truncated": truncated, "nudge_level": nudge}
                pending = getattr(transactions, "pending", None)
                if pending is not None:
                    pending.append((cached_path, payload, lock))
                    deferred = True
                elif not truncated:
                    atomic_json(cached_path, payload)
                event({"key": key, "cache_hit": False, "truncated": truncated,
                       "wall_seconds": time.monotonic() - started, "nudge_level": nudge})
                return response
            finally:
                if not deferred:
                    lock.close()

        client.chat.completions.create = audited_create
        return client

    api_clients.build_openai_compatible_client = factory


def isolated_config(source, base_url, cache_root):
    import copy
    config = copy.deepcopy(source)
    for service in config["services"].values():
        service.update(provider="openai", base_url=base_url, api_key="EMPTY", temperature=0.0)
        service.pop("api_key_env", None)
    for name in ("vlm_frames", "languagebind"):
        config.setdefault("cache", {}).setdefault(name, {}).update(cache_dir=str(cache_root / name))
    config["cache"]["metric_results"] = {"enabled": True}
    config["aggregation"]["llm_prompt_weighting"]["cache_dir"] = str(cache_root / "prompt_weights")
    for name in ("entity_grounding", "conditional_adjacent", "conditional_longrange"):
        config["aggregation"]["metrics"][name]["cache_dir"] = str(cache_root / name)
    return config


def count_nudged_requests(events_path):
    """How many served requests carried a format reminder (level > 0)."""
    counts = {}
    try:
        for line in Path(events_path).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            level = int(row.get("nudge_level") or 0)
            if level > 0 and "cache_hit" in row:
                counts[str(level)] = counts.get(str(level), 0) + 1
    except OSError:
        return {}
    return counts


def observed_fallbacks(output):
    """Report observed evidence, not a claim that all silent fallbacks are detected."""
    records = []
    for path in (Path(output) / "local_cache").rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            records.append({"path": str(path), "signal": "unreadable_cache"})
            continue

        def walk(value, location=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    label = f"{location}.{key}"
                    if (key in {"error", "retry_exhausted", "used_fallback"} and item) or (
                        key in {"parse_mode", "mode", "status", "reason"} and isinstance(item, str)
                        and any(word in item.lower() for word in ("fallback", "failed", "error", "empty_entities"))
                    ):
                        records.append({"path": str(path), "signal": label, "value": item})
                    walk(item, label)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{location}[{index}]")
        walk(payload)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("/home/chendong/video-gen/NarraStream-Bench"))
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--metrics", nargs="+")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cache_root = args.cache_root.resolve()
    if args.output.exists():
        raise FileExistsError("Fresh evaluation output required; do not reuse sample-ID caches")
    samples = json.loads(args.eval_data.read_text())
    if len(samples) != args.expected_count or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Evaluation sample count/identity mismatch")
    # Materialized video segments, source videos and prompt text all enter provenance.
    inputs = {str(args.eval_data.resolve()): digest(args.eval_data)}
    for sample in samples:
        if len(sample["prompts"]) != 6 or len(sample["segment_paths"]) != 6:
            raise ValueError("iteration33 requires six prompt segments")
        for path in [sample["source_video"], *sample["segment_paths"]]:
            inputs[str(Path(path).resolve())] = digest(path)
    args.output.mkdir(parents=True)
    import yaml
    source_config = args.bench / "configs/local_qwen3vl8b.yaml"
    config = isolated_config(yaml.safe_load(source_config.read_text()), args.base_url, args.cache_root)
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    code_hashes = {str(path): digest(path) for path in sorted((args.bench / "narrastream_bench").rglob("*.py"))}
    code_hashes.update({str(Path(__file__).resolve()): digest(__file__), str(source_config): digest(source_config),
                        str(args.bench / "configs/paths.yaml"): digest(args.bench / "configs/paths.yaml")})
    record = {"status": "running", "pid": os.getpid(), "expected_count": args.expected_count,
              "protocol": "iteration8b_seed0_validated_cache_v2", "official_paper_evaluation": False,
              "inputs": inputs, "code_hashes": code_hashes, "request_seed": 0,
              "device": args.device, "api_workers": args.api_workers,
              "cache_root": str(args.cache_root), "started_unix": time.time(),
              "audit_limitation": "Observed cache/log signals only; silent metric fallbacks and judge correctness are not certified."}
    atomic_json(args.output / "audit.json", record)
    sys.path.insert(0, str(args.bench))
    from narrastream_bench.utils import api_clients, api_retry
    install_audited_client(api_clients, args.response_cache or args.cache_root / "api_responses",
                           args.output / "api_events.jsonl", api_retry)
    try:
        from narrastream_bench.core.evaluator import NarraStreamBench
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Dedicated metric GPU is unavailable")
        bench = NarraStreamBench(torch.device(args.device), str(args.output), str(config_path),
                                 str(args.bench / "configs/paths.yaml"))
        results = bench.evaluate(str(args.eval_data), metric_list=args.metrics, api_workers=args.api_workers,
                                 snapshot_path=str(args.output / "results_latest.json"))
        changed = [path for path, value in {**inputs, **code_hashes}.items()
                   if not Path(path).is_file() or digest(path) != value]
        if changed:
            raise RuntimeError(f"Evaluation inputs/code changed: {changed[:5]}")
        coverage = evaluation_aggregated(results)["metric_coverage"]
        missing = {metric: row for metric, row in coverage.items()
                   if row["valid_count"] != args.expected_count}
        record.update(status="metrics_completed" if not missing else "metrics_completed_with_missing",
                      missing_coverage=missing, observed_fallbacks=observed_fallbacks(args.output))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["observed_fallbacks"] = observed_fallbacks(args.output)
        record["nudged_requests"] = count_nudged_requests(args.output / "api_events.jsonl")
        record["valid_for_quality_claims"] = False
        record["quality_claim_limitation"] = "Requires coverage/fallback/noise review and independent-model validation; completed scores alone are insufficient."
        record["finished_unix"] = time.time()
        atomic_json(args.output / "audit.json", record)


if __name__ == "__main__":
    main()
