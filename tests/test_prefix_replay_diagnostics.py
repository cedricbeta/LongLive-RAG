import copy
import pytest

from scripts.run_prefix_replay_diagnostics import compare_digests, eligible, FAILURE


def test_never_take_transiently_idle_generation_gpu():
    assert not eligible({"status": "running"}, False, 80000)
    assert not eligible({"status": "generation_finished_with_failures"}, True, 80000)
    assert not eligible({"status": "generation_finished_with_failures"}, False, 60000)
    assert eligible({"status": "generation_finished_with_failures"}, False, 80000)


def test_exact_prefix_not_suffix_or_tolerance():
    parent = {"shape": [1, 3, 1, 1, 1], "dtype": "torch.bfloat16", "num_chunks": 3,
              "chunk_digests": ["a", "b", "c"]}
    child = copy.deepcopy(parent)
    child["chunk_digests"][2] = "different suffix"
    assert compare_digests(parent, child, 2)["identical"]
    child["chunk_digests"][1] = "any difference"
    assert compare_digests(parent, child, 2)["mismatched_chunks"] == [1]
    child["dtype"] = "torch.float16"
    with pytest.raises(ValueError, match="layout"):
        compare_digests(parent, child, 2)


def test_invalid_digest_coverage_rejected():
    incomplete = {"shape": [1, 3, 1, 1, 1], "dtype": "torch.bfloat16", "num_chunks": 3,
                  "chunk_digests": ["a", "b"]}
    with pytest.raises(ValueError, match="coverage"):
        compare_digests(incomplete, incomplete, 2)


def test_recognizes_specific_failure_not_generic_error():
    message = "RuntimeError: edit_shot05_attempt2: replay prefix diverged from branch-0000 before chunk 40: {'identical': False}"
    assert FAILURE.findall(message) == [("edit_shot05_attempt2", "branch-0000", "40")]
    assert not FAILURE.findall("RuntimeError: CUDA out of memory")
