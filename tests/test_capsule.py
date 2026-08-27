import json
from pathlib import Path

import pytest
import torch

from utils.capsule import (
    CapsuleEditConfig,
    CapsuleProbeConfig,
    CapsuleRuntime,
    entity_mask_from_attention,
)


def test_capsule_configs_are_default_off():
    assert not CapsuleProbeConfig.from_config(None).enabled
    assert not CapsuleEditConfig.from_config(None).enabled


def test_attention_mask_has_declared_shape_and_nonempty_frames():
    soft = torch.zeros(2, 3, 4)
    soft[0, 1, 2] = 1.0
    soft[1, 0, 0] = 0.7
    mask = entity_mask_from_attention(
        soft,
        top_fraction=0.20,
        min_relative=0.50,
        largest_component=True,
        dilation=0,
    )
    assert mask.shape == soft.shape
    assert mask.dtype == torch.bool
    assert mask.flatten(1).any(dim=1).all()
    assert mask[0, 1, 2]
    assert mask[1, 0, 0]


def test_subject_offsets_and_cross_attention_ignore_padding():
    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "offset_mapping": [(0, 1), (2, 6), (0, 0), (0, 0)],
                "attention_mask": [1, 1, 1, 0],
            }

    probe = CapsuleProbeConfig.from_config(
        {
            "enabled": True,
            "mask_layers": [0],
            "subject_phrases": ["bear"],
        }
    )
    runtime = CapsuleRuntime(
        probe, CapsuleEditConfig(), frame_seqlen=4, grid_h=2, grid_w=2
    )
    runtime.start_sample("scene", 0, ["a bear"])
    runtime.set_runtime_context(chunk_index=0, shot_index=0, phase="clean_recache")
    runtime.runtime["num_frames"] = 2
    assert runtime.set_chunk_prompt("a bear", FakeTokenizer()) == [1]
    assert runtime.runtime["text_token_count"] == 3

    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 6, 2, 4)
    k[:, 3:] = 1.0e4  # padded keys must not enter the side softmax
    runtime.record_cross_attention(q, k, layer=0)
    assert runtime._mask_sums[0].shape == (2, 2, 2)
    assert torch.isfinite(runtime._mask_sums[0]).all()


def test_probe_dumps_only_masked_clean_v(tmp_path: Path):
    probe = CapsuleProbeConfig.from_config(
        {
            "enabled": True,
            "output_dir": str(tmp_path),
            "layers": [0],
            "frame_stride": 1,
            "mask_layers": [14],
            "subject_phrases": ["bear"],
            "mask_dilation": 0,
        }
    )
    runtime = CapsuleRuntime(
        probe,
        CapsuleEditConfig(),
        frame_seqlen=4,
        grid_h=2,
        grid_w=2,
    )
    runtime.start_sample("brown_bear_river", 0, ["a bear"])
    runtime.set_runtime_context(chunk_index=0, shot_index=0, phase="clean_recache")
    runtime.runtime.update(
        {
            "num_frames": 2,
            "start_token": 0,
            "subject_token_indices": [1],
            "prompt": "a bear",
        }
    )
    soft = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    runtime._mask_sums = {14: soft}
    new_v = torch.arange(1 * 8 * 2 * 128, dtype=torch.float32).reshape(1, 8, 2, 128)
    runtime.capture_clean_v(
        layer=0,
        new_v=new_v,
        update_info={
            "current_start": 0,
            "num_new_frames": 2,
            "new_v": new_v,
        },
    )
    runtime.finalize_sample()
    dump_path = next((tmp_path / "brown_bear_river" / "seed0" / "v" / "layer_00").glob("*.pt"))
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    assert payload["v"].shape == (2, 2, 128)
    assert payload["frame_indices"].tolist() == [0, 1]
    assert payload["spatial_indices"].tolist() == [0, 3]
    assert (tmp_path / "brown_bear_river" / "seed0" / "manifest.json").exists()


@pytest.mark.parametrize(
    "mask_method",
    [
        "clipseg_precomputed",
        "clipseg_precomputed_with_manual_visibility_overrides",
    ],
)
def test_precomputed_clipseg_mask_slices_global_clean_recache_frames(
    tmp_path: Path, mask_method: str
):
    mask_path = tmp_path / "masks.pt"
    binary = torch.zeros(12, 2, 2, dtype=torch.bool)
    binary[8, 0, 1] = True
    binary[9, 1, 0] = True
    torch.save(
        {
            "binary": binary,
            "soft": binary.to(torch.float16),
            "metadata": {"method": "test_clipseg"},
        },
        mask_path,
    )
    probe = CapsuleProbeConfig.from_config(
        {
            "enabled": True,
            "output_dir": str(tmp_path / "dump"),
            "layers": [0],
            "frame_stride": 1,
            "mask_method": mask_method,
            "mask_path": str(mask_path),
            "subject_phrases": ["bear"],
        }
    )
    runtime = CapsuleRuntime(
        probe, CapsuleEditConfig(), frame_seqlen=4, grid_h=2, grid_w=2
    )
    runtime.start_sample("scene", 0, ["a bear"])
    runtime.set_runtime_context(chunk_index=1, shot_index=0, phase="clean_recache")
    runtime.runtime.update({"num_frames": 2, "start_token": 8 * 4})
    soft, selected = runtime._ensure_current_mask()
    assert torch.equal(selected, binary[8:10])
    assert torch.equal(soft.bool(), binary[8:10])
    assert runtime.stats["cross_attention_calls"] == 0


def _artifact(path: Path, *, heads: int = 2):
    eye = torch.eye(128).repeat(heads, 1, 1)
    basis = torch.zeros(heads, 128, 1)
    basis[:, 0, 0] = 1.0
    torch.save(
        {
            "layer": 0,
            "rank": 1,
            "donor": torch.ones(heads, 128),
            "W": eye,
            "W_inv": eye,
            "U": basis,
            "random_R": basis,
        },
        path,
    )


def test_capsule_edit_rejects_malformed_or_wrong_scene_artifact(tmp_path: Path):
    artifact_path = tmp_path / "artifact.pt"
    _artifact(artifact_path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    payload["scene"] = "expected_scene"
    payload["U"] = payload["U"][:, :, :0]
    torch.save(payload, artifact_path)
    with pytest.raises(ValueError, match="artifact U must be"):
        CapsuleRuntime(
            CapsuleProbeConfig(),
            CapsuleEditConfig.from_config(
                {
                    "enabled": True,
                    "artifact_path": str(artifact_path),
                    "layer": 0,
                    "rank": 1,
                    "subject_phrases": ["subject"],
                }
            ),
            frame_seqlen=2,
            grid_h=1,
            grid_w=2,
        )

    _artifact(artifact_path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    payload["scene"] = "expected_scene"
    torch.save(payload, artifact_path)
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(),
        CapsuleEditConfig.from_config(
            {
                "enabled": True,
                "artifact_path": str(artifact_path),
                "layer": 0,
                "rank": 1,
                "subject_phrases": ["subject"],
            }
        ),
        frame_seqlen=2,
        grid_h=1,
        grid_w=2,
    )
    with pytest.raises(ValueError, match="artifact scene"):
        runtime.start_sample("wrong_scene", 0, ["subject"])


def test_capsule_edit_changes_only_selected_v_component_and_never_k(tmp_path: Path):
    artifact_path = tmp_path / "artifact.pt"
    _artifact(artifact_path)
    edit = CapsuleEditConfig.from_config(
        {
            "enabled": True,
            "arm": "capsule",
            "artifact_path": str(artifact_path),
            "layer": 0,
            "rank": 1,
            "alpha": 0.5,
            "boundaries": [1],
            "subject_phrases": ["subject"],
        }
    )
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(), edit, frame_seqlen=2, grid_h=1, grid_w=2
    )
    runtime.start_sample("scene", 0, ["subject"])
    runtime._cache_masks[0] = torch.tensor(
        [False, False, True, True, False, False, False, False, False, False]
    )
    k = torch.randn(1, 10, 2, 128)
    v = torch.zeros(1, 10, 2, 128)
    k_before = k.clone()
    cache = [
        {
            "k": k,
            "v": v,
            "global_end_index": torch.tensor([10]),
            "local_end_index": torch.tensor([10]),
            "pinned_start": torch.tensor([-1]),
            "pinned_len": torch.tensor([0]),
            "quantized": False,
        }
    ]
    record = runtime.edit_active_window(
        cache,
        boundary_shot=1,
        local_attn_frames=5,
        global_sink_frames=0,
    )
    assert record["edited_tokens"] == 2
    assert torch.equal(cache[0]["k"], k_before)
    assert torch.allclose(cache[0]["v"][0, 2:4, :, 0], torch.full((2, 2), 0.5))
    assert torch.count_nonzero(cache[0]["v"][0, 2:4, :, 1:]) == 0
    assert torch.count_nonzero(cache[0]["v"][0, :2]) == 0
    assert torch.count_nonzero(cache[0]["v"][0, 4:]) == 0


def test_edit_only_run_writes_runtime_manifest(tmp_path: Path):
    artifact_path = tmp_path / "artifact.pt"
    _artifact(artifact_path)
    edit = CapsuleEditConfig.from_config(
        {
            "enabled": True,
            "output_dir": str(tmp_path / "runtime"),
            "artifact_path": str(artifact_path),
            "layer": 0,
            "rank": 1,
            "subject_phrases": ["subject"],
        }
    )
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(), edit, frame_seqlen=2, grid_h=1, grid_w=2
    )
    runtime.start_sample("scene", 0, ["subject"])
    runtime.finalize_sample()
    manifest_path = tmp_path / "runtime/scene/seed0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["edit_config"]["enabled"] is True
    assert manifest["probe_config"]["enabled"] is False


def test_full_vector_arm_uses_same_in_place_path(tmp_path: Path):
    artifact_path = tmp_path / "artifact.pt"
    _artifact(artifact_path, heads=1)
    edit = CapsuleEditConfig.from_config(
        {
            "enabled": True,
            "arm": "full-vector",
            "artifact_path": str(artifact_path),
            "layer": 0,
            "rank": 1,
            "alpha": 1.0,
            "boundaries": [2],
            "subject_phrases": ["subject"],
        }
    )
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(), edit, frame_seqlen=1, grid_h=1, grid_w=1
    )
    runtime.start_sample("scene", 0, ["subject"])
    runtime._cache_masks[0] = torch.tensor([False, True, False])
    cache = [
        {
            "k": torch.randn(1, 3, 1, 128),
            "v": torch.zeros(1, 3, 1, 128),
            "global_end_index": torch.tensor([3]),
            "local_end_index": torch.tensor([3]),
            "pinned_start": torch.tensor([-1]),
            "pinned_len": torch.tensor([0]),
            "quantized": False,
        }
    ]
    runtime.edit_active_window(
        cache,
        boundary_shot=2,
        local_attn_frames=3,
        global_sink_frames=0,
    )
    assert torch.allclose(cache[0]["v"][0, 1], torch.ones(1, 128))
    assert torch.count_nonzero(cache[0]["v"][0, [0, 2]]) == 0


def test_full_vector_multi_layer_edit_changes_only_v(tmp_path: Path):
    artifact_path = tmp_path / "multi.pt"
    donor0 = torch.ones(1, 128)
    donor1 = torch.full((1, 128), 2.0)
    torch.save(
        {
            "scene": "scene",
            "per_layer": {
                "0": {"layer": 0, "rank": 1, "donor": donor0},
                "1": {"layer": 1, "rank": 1, "donor": donor1},
            },
        },
        artifact_path,
    )
    edit = CapsuleEditConfig.from_config(
        {
            "enabled": True,
            "arm": "full_vector",
            "artifact_path": str(artifact_path),
            "layer": 0,
            "layers": [0, 1],
            "rank": 1,
            "alpha": 1.0,
            "boundaries": [1],
            "subject_phrases": ["subject"],
        }
    )
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(), edit, frame_seqlen=1, grid_h=1, grid_w=1
    )
    runtime.start_sample("scene", 0, ["subject"])
    runtime._cache_masks[0] = torch.tensor([False, True, False])
    runtime._cache_masks[1] = torch.tensor([False, True, False])
    caches = []
    originals = []
    for _ in range(2):
        key = torch.randn(1, 3, 1, 128)
        originals.append(key.clone())
        caches.append(
            {
                "k": key,
                "v": torch.zeros(1, 3, 1, 128),
                "local_end_index": torch.tensor([3]),
                "pinned_start": torch.tensor([-1]),
                "pinned_len": torch.tensor([0]),
                "quantized": False,
            }
        )
    record = runtime.edit_active_window(
        caches, boundary_shot=1, local_attn_frames=3, global_sink_frames=0
    )
    assert record["layers"] == [0, 1]
    assert torch.equal(caches[0]["k"], originals[0])
    assert torch.equal(caches[1]["k"], originals[1])
    assert torch.allclose(caches[0]["v"][0, 1], donor0)
    assert torch.allclose(caches[1]["v"][0, 1], donor1)
    assert [(item["boundary_shot"], item["layer"]) for item in runtime.stats["edits"]] == [
        (1, 0),
        (1, 1),
    ]


def test_precomputed_mask_mirror_selects_only_active_local_sink_and_pinned_v(tmp_path: Path):
    artifact_path = tmp_path / "artifact.pt"
    _artifact(artifact_path, heads=1)
    mask_path = tmp_path / "mask.pt"
    binary = torch.tensor(
        [
            [[True, False]],
            [[False, True]],
            [[True, True]],
            [[True, False]],
        ],
        dtype=torch.bool,
    )
    torch.save({"binary": binary, "soft": binary.float(), "metadata": {}}, mask_path)
    edit = CapsuleEditConfig.from_config(
        {
            "enabled": True,
            "output_dir": str(tmp_path / "runtime"),
            "arm": "full_vector",
            "artifact_path": str(artifact_path),
            "layer": 0,
            "rank": 1,
            "alpha": 1.0,
            "boundaries": [1],
            "mask_method": "clipseg_precomputed",
            "mask_path": str(mask_path),
            "subject_phrases": ["subject"],
        }
    )
    runtime = CapsuleRuntime(
        CapsuleProbeConfig(), edit, frame_seqlen=2, grid_h=1, grid_w=2
    )
    runtime.start_sample("scene", 0, ["subject"])
    for chunk, start in ((0, 0), (1, 4)):
        runtime.set_runtime_context(
            chunk_index=chunk, shot_index=0, phase="clean_recache"
        )
        runtime.runtime.update({"num_frames": 2, "start_token": start})
        new_v = torch.zeros(1, 4, 1, 128)
        runtime.mirror_cache_update(
            layer=0,
            update_info={
                "action": "direct_insert",
                "new_v": new_v,
                "local_start_index": start,
                "local_end_index": start + 4,
            },
            cache_tokens=8,
        )
    assert runtime._cache_masks[0].tolist() == [True, False, False, True, True, True, True, False]

    k = torch.randn(1, 8, 1, 128)
    k_before = k.clone()
    cache = [
        {
            "k": k,
            "v": torch.zeros(1, 8, 1, 128),
            "local_end_index": torch.tensor([8]),
            "pinned_start": torch.tensor([4]),
            "pinned_len": torch.tensor([2]),
            "quantized": False,
        }
    ]
    record = runtime.edit_active_window(
        cache,
        boundary_shot=1,
        local_attn_frames=1,
        global_sink_frames=1,
    )
    assert record["edited_tokens"] == 4
    assert torch.equal(cache[0]["k"], k_before)
    changed = cache[0]["v"][0, :, 0].abs().sum(dim=-1).nonzero().flatten().tolist()
    assert changed == [0, 4, 5, 6]
