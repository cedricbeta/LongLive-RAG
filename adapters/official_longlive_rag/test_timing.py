import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('timing_adapter',Path(__file__).with_name('timing_adapter.py'))
timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)


def test_sixty_seconds_has_exact_boundaries_and_no_block_crosses_prompts():
    segments=[{'seconds':10,'prompts':[f'shot{i}']} for i in range(6)]
    result=timing.compile_timing(segments,['shot0'])
    assert result['latent_frames']==241
    assert result['decoded_frames_before_crop']==961
    assert result['export_frames']==960 and result['crop_start_frames']==1
    assert [r['start_export_frame'] for r in result['segments']]==[0,160,320,480,640,800]
    assert [r['start_latent_frame'] for r in result['segments']]==[0,41,81,121,161,201]
    cursor=0
    starts=set(result['schedule'])
    for block in result['blocks']:
        assert 1<=block<=3
        assert not any(cursor<boundary<cursor+block for boundary in starts)
        cursor+=block
    assert cursor==241


@pytest.mark.parametrize('seconds',[0,0.1,10.0625])
def test_fractional_latent_timing_is_not_silently_rounded(seconds):
    with pytest.raises(ValueError):
        timing.compile_timing([{'seconds':seconds,'prompts':['p']}],['p'])
