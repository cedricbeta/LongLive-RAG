import json
from types import SimpleNamespace

import pytest

from scripts.run_audited_narrastream_eval import (
    atomic_json, install_audited_client, isolated_config, observed_fallbacks,
    redact_images, request_key, evaluation_aggregated, count_nudged_requests, nudged_request,
)
from scripts.run_iteration_eval_campaign import paired_interval, validate_samples, merge_parts, FEATURE_METRICS, NOISE_METRICS


def test_fingerprint_includes_image_prompt_model_seed_and_sampling():
    request = {"model": "model", "messages": ["prompt", "data:image/png;base64,AAAA"], "seed": 0, "temperature": 0}
    assert request_key(request) == request_key(dict(reversed(list(request.items()))))
    for key, value in [("seed", 1), ("temperature", 1), ("model", "other"), ("messages", ["different"])]:
        assert request_key(request) != request_key({**request, key: value})


def test_redaction_retains_image_fingerprint_not_pixels():
    value = redact_images({"content": ["text", "data:image/png;base64,AAAA"]})
    assert value["content"][0] == "text"
    assert len(value["content"][1]["image_sha256"]) == 64
    assert "base64" not in json.dumps(value)


def test_isolated_config_does_not_mutate_metrics_or_original(tmp_path):
    source = {"services": {"vlm": {"api_key_env": "SECRET", "model": "m"}},
              "aggregation": {"llm_prompt_weighting": {}, "metrics": {
                  name: {"some_formula_parameter": 0.5} for name in
                  ("entity_grounding", "conditional_adjacent", "conditional_longrange")}}}
    result = isolated_config(source, "http://127.0.0.1:1234/v1", tmp_path)
    assert result["aggregation"]["metrics"]["entity_grounding"]["some_formula_parameter"] == 0.5
    assert "cache_dir" not in source["aggregation"]["metrics"]["entity_grounding"]
    assert "api_key_env" not in result["services"]["vlm"]
    assert "cache" not in source


def test_cached_completion_is_identical_and_seed_is_sent(tmp_path):
    from openai.types.chat import ChatCompletion
    calls = []
    response = ChatCompletion.model_validate({"id": "test", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]})
    def create(**kwargs):
        calls.append(kwargs)
        return response
    api = SimpleNamespace(build_openai_compatible_client=lambda service:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    install_audited_client(api, tmp_path / "cache", tmp_path / "events.jsonl")
    service = SimpleNamespace(base_url="http://127.0.0.1:1234/v1")
    client = api.build_openai_compatible_client(service)
    first = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hello"}])
    second = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hello"}])
    assert first.model_dump() == second.model_dump()
    assert len(calls) == 1 and calls[0]["seed"] == 0
    assert [json.loads(line)["cache_hit"] for line in (tmp_path / "events.jsonl").read_text().splitlines()] == [False, True]
    with pytest.raises(ValueError, match="loopback"):
        api.build_openai_compatible_client(SimpleNamespace(base_url="https://example.com/v1"))


def test_observed_fallbacks_are_not_inferred_from_midpoint_scores(tmp_path):
    atomic_json(tmp_path / "local_cache/a.json", {"score": 0.5})
    atomic_json(tmp_path / "local_cache/b.json", {"nested": {"retry_exhausted": True, "parse_mode": "request_failed"}})
    result = observed_fallbacks(tmp_path)
    assert len(result) == 2
    assert all(row["path"].endswith("b.json") for row in result)


def test_paired_interval_preserves_missingness_and_is_deterministic():
    a = [{"sample_id": 0, "m": 0.1}, {"sample_id": 1, "m": None}]
    b = [{"sample_id": 0, "m": 0.3}, {"sample_id": 1, "m": 0.9}]
    result = paired_interval(a, b, "m")
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(0.2)
    assert paired_interval(a, b, "missing")["n"] == 0
    assert paired_interval(a, b, "m") == result


def test_reject_missing_or_duplicate_sample_id():
    with pytest.raises(ValueError, match="Missing/reordered"):
        validate_samples([], ["scene"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_samples([{"scene": "a", "sample_id": 0}, {"scene": "b", "sample_id": 0}], ["a", "b"])


def test_aggregate_return_and_snapshot_contract():
    aggregate = {"entity_grounding": 0.5, "metric_coverage": {"entity_grounding": {"valid_count": 6, "total_count": 6}}}
    assert evaluation_aggregated(aggregate) == aggregate
    assert evaluation_aggregated({"aggregated": aggregate, "per_sample": []}) == aggregate
    with pytest.raises(ValueError):
        evaluation_aggregated({"score": 0.5})


def test_rejected_payload_is_not_cached_and_retry_calls_server(tmp_path):
    from openai.types.chat import ChatCompletion
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        content = '{"":[]}' if len(calls) == 1 else '{"entities":[]}'
        return ChatCompletion.model_validate({"id": str(len(calls)), "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})
    api = SimpleNamespace(build_openai_compatible_client=lambda service:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    def retry(operation, **kwargs):
        try:
            return operation()
        except ValueError:
            return operation()
    retry_module = SimpleNamespace(run_with_retry=retry)
    install_audited_client(api, tmp_path / "cache", tmp_path / "events.jsonl", retry_module)
    client = api.build_openai_compatible_client(SimpleNamespace(base_url="http://127.0.0.1:1234/v1"))
    def parse():
        response = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "entities"}])
        value = json.loads(response.choices[0].message.content)
        if "entities" not in value:
            raise ValueError("Invalid entities")
        return value
    assert retry_module.run_with_retry(parse, label="test") == {"entities": []}
    assert len(calls) == 2
    assert retry_module.run_with_retry(parse, label="test") == {"entities": []}
    assert len(calls) == 2
    assert len(list((tmp_path / "cache/rejected").glob("*.json"))) == 1


def test_real_evaluator_snapshot_return(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    source = Path("/home/chendong/video-gen/NarraStream-Bench/narrastream_bench/core/evaluator.py")
    if not source.exists():
        pytest.skip("External evaluator checkout not installed")
    spec = importlib.util.spec_from_file_location("benchmark_contract_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    data = tmp_path / "data.json"
    atomic_json(data, [{"sample_id": 0}])
    bench = module.NarraStreamBench("cpu", str(tmp_path / "results"), str(config), str(config))
    monkeypatch.setattr(bench, "_compute_metric_scores", lambda metric, data, **kw: ({0: 0.75}, "test_metric"))
    result = bench.evaluate(str(data), metric_list=["subject_consistency"],
                            snapshot_path=str(tmp_path / "results/results_latest.json"))
    assert evaluation_aggregated(result)["metric_coverage"]["subject_consistency"]["valid_count"] == 1


def test_merge_disjoint_parts_uses_full_metric_groups():
    parts = []
    for metrics, value in [(FEATURE_METRICS, 0.6), (NOISE_METRICS, 0.9)]:
        parts.append({"aggregated": {**{m: value for m in metrics}, "total_score": 0.01,
            "metric_coverage": {m: {"valid_count": 1, "total_count": 1} for m in metrics}},
            "per_sample": [{"sample_id": 0, **{m: value for m in metrics}}]})
    merged = merge_parts(parts)
    assert merged["aggregated"]["quality_score"] == pytest.approx(0.6)
    assert merged["aggregated"]["temporal_score"] == pytest.approx(0.8)
    assert merged["aggregated"]["instruction_score"] == pytest.approx(0.8)
    assert merged["aggregated"]["total_score"] == pytest.approx(2.2 / 3)
    assert merged["per_sample"][0]["total_score"] == pytest.approx(2.2 / 3)
    with pytest.raises(ValueError, match="Overlapping"):
        merge_parts([parts[0], parts[0]])


def test_invalid_extractor_payload_escalates_a_format_reminder(tmp_path):
    """A deterministic {"": ""} answer only recovers when the request changes."""
    from openai.types.chat import ChatCompletion
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        content = '{"entities":[{"name":"Nora","attributes":["beige sweater"]}]}' if "Format reminder" in prompt else '{"": ""}'
        return ChatCompletion.model_validate({"id": str(len(calls)), "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})
    api = SimpleNamespace(build_openai_compatible_client=lambda service:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    def retry(operation, **kwargs):
        for _ in range(10):
            try:
                return operation()
            except ValueError:
                continue
        raise RuntimeError("exhausted")
    retry_module = SimpleNamespace(run_with_retry=retry)
    install_audited_client(api, tmp_path / "cache", tmp_path / "events.jsonl", retry_module)
    client = api.build_openai_compatible_client(SimpleNamespace(base_url="http://127.0.0.1:1234/v1"))
    def parse():
        response = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "extract entities"}])
        value = json.loads(response.choices[0].message.content)
        if not isinstance(value.get("entities"), list):
            raise ValueError("Invalid entity grounding payload")
        return value
    result = retry_module.run_with_retry(parse, label="test")
    assert result["entities"][0]["name"] == "Nora"
    assert len(calls) == 2                                   # one plain attempt, one nudged attempt
    assert "Format reminder 1" in calls[1]["messages"][-1]["content"]
    assert calls[0]["messages"][-1]["content"] == "extract entities"   # original prompt untouched
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row.get("nudge_level") for row in events if "cache_hit" in row] == [0, 1]
    assert count_nudged_requests(tmp_path / "events.jsonl") == {"1": 1}
    # a second identical operation re-asks the (uncached, rejected) plain request once,
    # then hits the nudged cache: one extra server call, no second nudged call
    assert retry_module.run_with_retry(parse, label="test")["entities"][0]["name"] == "Nora"
    assert len(calls) == 3
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    served = [(row.get("nudge_level"), row["cache_hit"]) for row in events if "cache_hit" in row]
    assert served == [(0, False), (1, False), (0, False), (1, True)], served
    # unrelated parser errors never change the prompt
    def other():
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "score"}])
        raise ValueError("entity_results must be a list")
    with pytest.raises(RuntimeError):
        retry_module.run_with_retry(other, label="test")
    assert all("Format reminder" not in c["messages"][-1]["content"] for c in calls[2:])


def test_nudged_request_handles_content_parts_and_leaves_other_roles():
    request = {"messages": [{"role": "system", "content": "sys"},
                            {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    out = nudged_request(request, 2)
    assert out["messages"][0]["content"] == "sys"
    assert out["messages"][1]["content"][-1]["type"] == "text" and "Format reminder 2" in out["messages"][1]["content"][-1]["text"]
    assert request["messages"][1]["content"] == [{"type": "text", "text": "hi"}]   # input not mutated


def test_observed_fallbacks_count_empty_entity_defaults(tmp_path):
    cache = tmp_path / "local_cache/metrics/entity_grounding__v5/items/segments"
    cache.mkdir(parents=True)
    (cache / "sample_1__item_0.json").write_text(json.dumps({"score": 0.5, "reason": "empty_entities"}))
    (cache / "sample_1__item_1.json").write_text(json.dumps({"score": 0.7, "scoring_variant": "EG-Entity"}))
    signals = observed_fallbacks(tmp_path)
    assert len(signals) == 1 and signals[0]["value"] == "empty_entities"
