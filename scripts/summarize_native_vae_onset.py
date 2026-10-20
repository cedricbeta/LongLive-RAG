#!/usr/bin/env python3
"""Audit bounded VAE interventions and display raw evidence without quality labels."""
import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(folder, telemetry, raw_windows=None, mask_path=None):
    import numpy as np
    from PIL import Image
    inputs = {}
    def load(path):
        inputs[str(path.relative_to(folder.parent))] = sha(path)
        return read(path)
    plan, status, result = [load(folder / name) for name in ('plan.json', 'status.json', 'result.json')]
    assert status['status'] == 'completed' and status['returncode'] == 0
    assert sha(folder / 'runner.py') == plan['source_sha256']
    assert sha(folder / 'admission_source.py') == plan['admission_source_sha256']
    for name, expected in plan['native_source_sha256'].items():
        assert sha(folder / 'native_source' / name) == expected
    assert result['actual_vae_decodes'] == len(result['rows']) == len(plan['cases']) == 5
    assert result['actual_cached_decode_calls'] == 10
    assert result['new_diffusion_calls'] == result['new_vlm_calls'] == result['human_labels'] == 0
    assert not result['candidate_adopted']
    rows = {row['id']: row for row in result['rows']}
    hashes, traces = {}, {}
    for case in plan['cases']:
        row = rows[case['id']]
        assert row['case'] == case and row['decoded_shape'] == [476, 3, 480, 832]
        assert [r['frame'] for r in row['frames']] == list(range(476))
        assert row['first_changed_frame'] == next((r['frame'] for r in row['frames'] if r['changed_values']), None)
        for frame in row['frames']:
            assert 0 <= frame['mae'] <= frame['max_abs'] <= 1 and frame['rms'] >= frame['mae'] - 1e-7
        for frame, expected in row['saved_png_sha256'].items():
            path = folder / 'frames' / f"{case['id']}_{frame}.png"
            assert sha(path) == expected
            inputs[str(path.relative_to(folder.parent))] = expected
            with Image.open(path) as image:
                assert image.size == (832, 480) and image.mode == 'RGB'
        hashes[case['id']] = row['raw_prefix_320_sha256']
        traces[case['id']] = {str(frame['frame']): {k: frame[k] for k in ('mae', 'max_abs', 'mask_projected_mae') if k in frame}
            for frame in row['frames'] if frame['frame'] in plan['saved_frames']}
    assert len(set(hashes.values())) == 1
    repeat_exact = rows['parent']['raw_video_sha256'] == rows['parent_repeat']['raw_video_sha256']
    assert repeat_exact and rows['parent_repeat']['first_changed_frame'] is None
    pairs = result['pairwise_raw_window']
    for name, values in pairs.items():
        a, b = name.split('_vs_')
        for row in values:
            frame = row['frame']
            assert row['exact'] == (rows[a]['frames'][frame]['raw_frame_sha256'] == rows[b]['frames'][frame]['raw_frame_sha256'])
    runtime = None
    if raw_windows is not None:
        import torch
        torch.set_num_threads(4)
        assert mask_path is not None and sha(mask_path) == plan['input_files']['mask']['sha256']
        mask = torch.load(mask_path, map_location='cpu', weights_only=True)['mask']
        windows = {}
        tensor_sha = lambda value: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, row in rows.items():
            path = raw_windows / (name + '.pt')
            assert sha(path) == row['local_raw_window_file_sha256']
            window = torch.load(path, map_location='cpu', weights_only=True)
            assert tensor_sha(window) == row['raw_window_tensor_sha256']
            windows[name] = window
        for name, window in windows.items():
            for offset, pixels in enumerate(window):
                frame = offset + plan['raw_window'][0]
                expected = rows[name]['frames'][frame]
                assert tensor_sha(pixels) == expected['raw_frame_sha256']
                d = pixels - windows['parent'][offset]
                assert abs(d.abs().mean().item() - expected['mae']) <= 1e-8
                assert d.abs().max().item() == expected['max_abs'] and int(d.count_nonzero()) == expected['changed_values']
                if frame >= 320:
                    region = mask[frame // 4 + 1 - 81].bool().repeat_interleave(8, 0).repeat_interleave(8, 1)
                    assert abs(d[:, region].abs().mean().item() - expected['mask_projected_mae']) <= 1e-8
                if frame in plan['saved_frames']:
                    actual = (pixels * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    with Image.open(folder / 'frames' / f'{name}_{frame}.png') as image:
                        assert np.array_equal(actual, np.asarray(image))
        for name, values in pairs.items():
            a, b = name.split('_vs_')
            for row in values:
                offset = row['frame'] - plan['raw_window'][0]
                assert torch.equal(windows[a][offset], windows[b][offset]) == row['exact']
        runtime = {'actual_raw_frames_recomputed': 5 * 72, 'pngs_reconstructed': 5 * 16,
            'scalar_tolerance': 1e-8, 'tensor_and_png_equality': 'exact', 'scope': 'Only stored raw window312:384 is available for recomputation; full476-frame hashes bind the recorded GPU decode.'}
    monitor = load(telemetry / 'plan.json')
    measured = load(telemetry / 'summary.json')
    assert sha(folder / 'plan.json') == monitor['target_plan_sha256']
    assert sha(folder / 'manifest.json') == monitor['target_manifest_sha256']
    assert sha(telemetry / 'monitor_source.py') == monitor['source_sha256']
    assert load(telemetry / 'target_terminal_status.json') == status
    assert sha(telemetry / 'samples.jsonl') == measured['samples_sha256']
    assert sha(telemetry / 'target_terminal_status.json') == measured['target_status_sha256']
    spec = importlib.util.spec_from_file_location('onset_monitor', telemetry / 'monitor_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [json.loads(line) for line in (telemetry / 'samples.jsonl').read_text().splitlines()]
    assert module.summarize(samples, status) == {k: v for k, v in measured.items() if k not in ('samples_sha256', 'target_status_sha256')}
    inputs[str((telemetry / 'samples.jsonl').relative_to(folder.parent))] = sha(telemetry / 'samples.jsonl')
    return {'protocol': plan['protocol'], 'first_changed_raw_frames': {name: row['first_changed_frame'] for name, row in rows.items()},
        'all_five_raw_prefixes_320_exact': True, 'raw_parent_repeat_all_476_frames_exact': repeat_exact,
        'early_only_equals_candidate_through_331': all(r['exact'] for r in pairs['early_only_vs_candidate'] if r['frame'] <= 331),
        'later_only_equals_parent_through_331': all(r['exact'] for r in pairs['parent_vs_later_only'] if r['frame'] <= 331),
        'fixed_frame_deltas': traces, 'supervisor_seconds': status['wall_seconds'],
        'worker_seconds_nested': result['worker_seconds'], 'loading_seconds_nested': result['loading_seconds_nested'],
        'device_energy_bounds': measured['device_energy_bounds'], 'observed_foreign_pids': measured['observed_foreign_compute_pids_inside'],
        'actual_vae_decodes': 5, 'new_diffusion_calls': 0, 'new_vlm_calls': 0,
        'raw_window_recomputation': runtime, 'inputs_sha256': inputs,
        'candidate_adopted': False, 'actual_human_labels': 0, 'held_out': False,
        'scope': 'Numerical causal influence from a frozen latent intervention on one exposed script. Pixel change is not garment accuracy, and hybrid decodes are not usable repair outputs. No policy, sampling, acceptance or historical verdict change.'}


def plot(folder, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = read(folder / 'result.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    labels = {'early_only': 'First edited block only', 'later_only': 'Later blocks only', 'candidate': 'Full candidate prefix'}
    for row in result['rows']:
        if row['id'] not in labels:
            continue
        frames = [r for r in row['frames'] if 312 <= r['frame'] <= 383]
        axes[0].plot([r['frame'] for r in frames], [r['mae'] for r in frames], label=labels[row['id']])
        frames = [r for r in frames if r['frame'] >= 320]
        axes[1].plot([r['frame'] for r in frames], [r['mask_projected_mae'] for r in frames], label=labels[row['id']])
    for ax, title in zip(axes, ['All RGB pixels', 'Automatic-mask projection (diagnostic region)']):
        ax.axvline(320, color='#444', linestyle=':', linewidth=1)
        ax.axvline(332, color='#888', linestyle='--', linewidth=1)
        ax.set(xlabel='Export frame index', ylabel='Mean absolute change from raw parent [0, 1]', title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Causal VAE block interventions — pixel difference, not repair accuracy')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def gallery(folder, output):
    plan = read(folder / 'plan.json')
    text = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Native VAE onset interventions</title><style>body{font:16px system-ui;margin:25px auto;padding:0 20px;max-width:1400px;color:#17232e}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%;height:auto}section{border-top:1px solid #aaa;margin:30px 0;padding-top:10px}@media(max-width:750px){.grid{grid-template-columns:1fr 1fr}}</style>',
        '<h1>Existing latents, four causal decode conditions</h1>',
        '<p>Raw VAE frames, before lossy video encoding. First block = latent81:84; later blocks =84:120. Two hybrid conditions are diagnostic interventions, not generated repair candidates. Pixel change does not establish clothing correctness, identity, actions or protection.</p>',
        '<img src="onset_deltas.png" alt="All-frame pixel difference traces">']
    for frame in plan['saved_frames']:
        text.append(f'<section><h2>Frame {frame} · {frame / 16:g} seconds</h2><div class="grid">')
        for name in ('parent', 'early_only', 'later_only', 'candidate'):
            src = f'../../results/{folder.name}/frames/{name}_{frame}.png'
            text.append(f'<div><h3>{html.escape(name)}</h3><a href="{src}"><img width="832" height="480" src="{src}" alt="{name}, frame {frame}"></a></div>')
        text.append('</div></section>')
    text.append('<p>Parent repetition frames and all 476-frame numerical traces remain in the result folder. No candidate is adopted.</p></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(text) + '\n')


def mask_plot(folder, mask_path, output):
    """Post hoc diagnostic view of the actual mask; not a manual region label."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from PIL import Image
    mask = torch.load(mask_path, map_location='cpu', weights_only=True)['mask']
    assert sha(mask_path) == read(folder / 'plan.json')['input_files']['mask']['sha256']
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for row, frame in enumerate([320, 327, 331, 352]):
        parent = np.asarray(Image.open(folder / 'frames' / f'parent_{frame}.png'))
        candidate = np.asarray(Image.open(folder / 'frames' / f'candidate_{frame}.png'))
        region = mask[frame // 4 + 1 - 81].numpy().repeat(8, 0).repeat(8, 1)
        for ax, image, title in zip(axes[row], [parent, parent, candidate], ['Parent', 'Actual mask boundary', 'Candidate']):
            ax.imshow(image)
            ax.set_title(f'{title} · {frame}')
            ax.axis('off')
        axes[row, 1].contour(region, levels=[.5], colors=['#ffdf00'], linewidths=1)
    fig.savefig(output, dpi=110, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'telemetry', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('raw-windows', 'mask', 'plot', 'gallery', 'mask-plot'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    result = summarize(args.folder.resolve(), args.telemetry.resolve(), args.raw_windows, args.mask)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.plot:
        plot(args.folder, args.plot)
    if args.gallery:
        gallery(args.folder, args.gallery)
    if args.mask_plot:
        mask_plot(args.folder, args.mask, args.mask_plot)
    print(json.dumps({k: result[k] for k in ('first_changed_raw_frames', 'supervisor_seconds', 'raw_window_recomputation')}))
