"""Contract tests; the real pinned-checkpoint run is a separate validation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location('prompt_adapter', Path(__file__).with_name('prompt_schedule_adapter.py'))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_schedule_rejects_bad_timing_and_changed_initial_prompt():
    assert adapter.validate_schedule(None, ['a'], 120, 3) == {}
    schedule = [{'start_latent_frame': 0, 'prompts': ['a']}, {'start_latent_frame': 60, 'prompts': ['b']}]
    assert adapter.validate_schedule(schedule, ['a'], 120, 3) == {0: ['a'], 60: ['b']}
    for start in [1, -3, 120, True]:
        with pytest.raises(ValueError):
            adapter.validate_schedule([schedule[0], {'start_latent_frame': start, 'prompts': ['b']}], ['a'], 120, 3)
    with pytest.raises(ValueError):
        adapter.validate_schedule(schedule, ['different'], 120, 3)


def test_native_decoder_timing_does_not_pretend_120_latents_is_30_seconds():
    assert adapter.native_decoded_prefix(0) == 0
    assert adapter.native_decoded_prefix(60) == 237
    assert adapter.native_decoded_prefix(120) == 474
    assert adapter.native_decoded_prefix(240) == 948
    assert adapter.native_decoded_prefix(246) == 969


@pytest.mark.parametrize('end,resident,indices', [(3,3,[0,1,2]), (9,4,[0,6,7,8])])
def test_refresh_packs_sink_and_recent_without_replacing_history(end, resident, indices):
    pipeline = SimpleNamespace(frame_seq_length=2, args=SimpleNamespace(
        model_kwargs=SimpleNamespace(sink_size=1), context_noise=0),
        latent_descriptors=[torch.tensor([float(i)]) for i in range(end)])
    pipeline.kv_cache1 = [dict(k=torch.zeros(1,resident*2,1,1),v=torch.zeros(1,resident*2,1,1),
        global_end_index=torch.tensor([end*2]),local_end_index=torch.tensor([resident*2]),
        cpu_k_frames=[torch.tensor([7.])],cpu_v_frames=[torch.tensor([8.])])]
    pipeline.text_encoder = lambda text_prompts: {'prompt_embeds': text_prompts}
    def initialize(**kwargs):
        pipeline.crossattn_cache = [{'is_init': False}]
    pipeline._initialize_crossattn_cache = initialize
    output = torch.arange(float(end)).reshape(1,end,1,1,1)
    def generator(**kwargs):
        assert kwargs['noisy_image_or_video'].flatten().tolist() == indices
        assert kwargs['current_start'] == (end-resident)*2
        assert kwargs['sink_recache_after_switch']
        pipeline.kv_cache1[0]['k'].fill_(2)
        pipeline.kv_cache1[0]['v'].fill_(3)
        pipeline.crossattn_cache[0]['is_init'] = True
    pipeline.generator = generator
    conditional, report = adapter.refresh_prompt(pipeline, output, end, ['next'], None, audit=True)
    assert conditional == {'prompt_embeds': ['next']}
    assert report['resident_source_frames'] == indices
    assert report['history_sha256_before'] == report['history_sha256_after']
    assert report['resident_k_sha256_before'] != report['resident_k_sha256_after']
    assert report['cross_attention_initialized'] == [True]
    def corrupt(**kwargs):
        generator(**kwargs)
        pipeline.kv_cache1[0]['cpu_k_frames'][0].add_(1)
    pipeline.generator = corrupt
    with pytest.raises(RuntimeError, match='historical retrieval'):
        adapter.refresh_prompt(pipeline, output, end, ['bad'], None, audit=True)
