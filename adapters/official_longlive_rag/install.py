#!/usr/bin/env python3
"""Install a minimal opt-in adapter into a clean, pinned official checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = '973884a3cd3ad4b314c3d4ab42274c52e7a0b22a'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned source anchor did not match exactly once: ' + old[:70])
    return source.replace(old, new, 1)


def install(root, timed=False):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != REVISION:
        raise ValueError('This adapter supports only the pinned official revision')
    names = ['pipeline/causal_inference.py', 'inference.py']
    original = {name: (root / name).read_bytes() for name in names}
    for name in names:
        clean = subprocess.check_output(['git', 'show', f'{REVISION}:{name}'], cwd=root)
        if original[name] != clean:
            raise ValueError('Refusing to replace modified official file: ' + name)
    helper = root / 'prompt_schedule_adapter.py'
    if helper.exists():
        raise FileExistsError(helper)
    if timed and (root/'timing_adapter.py').exists():
        raise FileExistsError(root/'timing_adapter.py')
    source = original[names[0]].decode()
    source = once(source, '        skip_vae_decode: bool = False,',
        '        skip_vae_decode: bool = False,\n        prompt_schedule=None,\n        audit_prompt_switch=False,')
    source = once(source, '        num_blocks = num_output_frames // self.num_frame_per_block',
        '        num_blocks = num_output_frames // self.num_frame_per_block\n'
        '        from prompt_schedule_adapter import validate_schedule, refresh_prompt\n'
        '        schedule = validate_schedule(prompt_schedule, text_prompts, num_output_frames, self.num_frame_per_block)\n'
        '        self.prompt_switch_log = []')
    source = once(source, '            # Step 2.1: Spatial denoising loop',
        '            if current_start_frame > 0 and current_start_frame in schedule:\n'
        '                conditional_dict, switch_report = refresh_prompt(\n'
        '                    self, output, current_start_frame, schedule[current_start_frame],\n'
        '                    memory_indices, audit=audit_prompt_switch)\n'
        '                self.prompt_switch_log.append(switch_report)\n\n'
        '            # Step 2.1: Spatial denoising loop')
    runner = original[names[1]].decode()
    runner = once(runner, '        profile=False,',
        '        profile=False,\n'
        '        prompt_schedule=OmegaConf.to_container(config.prompt_schedule, resolve=True) if getattr(config, "prompt_schedule", None) else None,\n'
        '        audit_prompt_switch=bool(getattr(config, "audit_prompt_switch", False)),')
    runner = once(runner, '    current_video = rearrange(video,',
        '    if getattr(config, "adapter_diagnostics_dir", None):\n'
        '        import json\n'
        '        from pathlib import Path\n'
        '        diagnostic_dir = Path(config.adapter_diagnostics_dir)\n'
        '        diagnostic_dir.mkdir(parents=True, exist_ok=True)\n'
        '        torch.save(latents.cpu(), diagnostic_dir / f"latents_{idx:04d}.pt")\n'
        '        (diagnostic_dir / f"switches_{idx:04d}.json").write_text(json.dumps(pipeline.prompt_switch_log, indent=2) + "\\n")\n'
        '    current_video = rearrange(video,')
    if timed:
        source = once(source, '        audit_prompt_switch=False,',
            '        audit_prompt_switch=False,\n        timed_segments=None,')
        source = once(source, '        assert num_output_frames % self.num_frame_per_block == 0',
            '        if timed_segments is None:\n            assert num_output_frames % self.num_frame_per_block == 0')
        source = once(source, '        self.prompt_switch_log = []',
            '        self.prompt_switch_log = []\n'
            '        self.timing_record = None\n'
            '        if timed_segments is not None:\n'
            '            if prompt_schedule is not None:\n'
            '                raise ValueError("Choose timed segments or latent schedule, not both")\n'
            '            from timing_adapter import compile_timing\n'
            '            self.timing_record = compile_timing(timed_segments, text_prompts)\n'
            '            if num_output_frames != self.timing_record["latent_frames"]:\n'
            '                raise ValueError("Noise length does not match exact timing plan")\n'
            '            schedule = self.timing_record["schedule"]')
        source = once(source, '        all_num_frames = [self.num_frame_per_block] * num_blocks',
            '        all_num_frames = self.timing_record["blocks"] if self.timing_record else [self.num_frame_per_block] * num_blocks')
        source = once(source, '                self.prompt_switch_log.append(switch_report)',
            '                if self.timing_record:\n'
            '                    switch_report["independent_decoder_prefix_frames"] = switch_report.pop("native_decoded_prefix_frames")\n'
            '                    switch_report["export_prefix_frames"] = 4 * current_start_frame - 4\n'
            '                    switch_report["timing_protocol"] = self.timing_record["protocol"]\n'
            '                self.prompt_switch_log.append(switch_report)')
        source = once(source, '            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)',
            '            if self.timing_record:\n'
            '                from timing_adapter import decode_continuous\n'
            '                video = decode_continuous(self.vae, output.to(noise.device))\n'
            '                if video.shape[1] != self.timing_record["decoded_frames_before_crop"]:\n'
            '                    raise RuntimeError("Actual decoder duration disagrees with exact timing plan")\n'
            '                video = video[:, 1:]\n'
            '                if video.shape[1] != self.timing_record["export_frames"]:\n'
            '                    raise RuntimeError("Export frame count disagrees with exact timing plan")\n'
            '            else:\n'
            '                video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)')
        runner = once(runner, '        audit_prompt_switch=bool(getattr(config, "audit_prompt_switch", False)),',
            '        audit_prompt_switch=bool(getattr(config, "audit_prompt_switch", False)),\n'
            '        timed_segments=OmegaConf.to_container(config.timed_segments, resolve=True) if getattr(config, "timed_segments", None) else None,')
        runner = once(runner, '        torch.save(latents.cpu(), diagnostic_dir / f"latents_{idx:04d}.pt")',
            '        torch.save(latents.cpu(), diagnostic_dir / f"latents_{idx:04d}.pt")\n'
            '        (diagnostic_dir / f"timing_{idx:04d}.json").write_text(json.dumps(pipeline.timing_record, indent=2) + "\\n")')
    # All validation/transformation succeeds before changing the checkout.
    (root / names[0]).write_text(source)
    (root / names[1]).write_text(runner)
    shutil.copy2(Path(__file__).with_name('prompt_schedule_adapter.py'), helper)
    if timed:
        shutil.copy2(Path(__file__).with_name('timing_adapter.py'), root/'timing_adapter.py')
    sha = lambda data: hashlib.sha256(data).hexdigest()
    record = {'official_revision': REVISION, 'original_sha256': {n:sha(b) for n,b in original.items()},
              'adapted_sha256': {n:sha((root/n).read_bytes()) for n in names + [helper.name] + (['timing_adapter.py'] if timed else [])},
              'timed_adapter_installed': timed,
              'status': 'Experimental prompt-switch adapter; correction not implemented'}
    (root/'prompt_adapter_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--timed', action='store_true')
    args=parser.parse_args()
    print(json.dumps(install(args.checkout.resolve(), timed=args.timed), indent=2))
