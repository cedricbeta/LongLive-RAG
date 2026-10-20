#!/usr/bin/env python3
"""Reconstruct wording-control costs and raw observations; make no quality verdict."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def summarize(root):
    inputs = {}
    def read(name):
        path = root / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    prefix = 'results/native_iris_natural_expression_v1/'
    plan = read(prefix + 'plan.json')
    binding = read(prefix + 'binding.json')
    manifest = read(prefix + 'candidate/manifest.json')
    native = read(prefix + 'candidate/audit.json')
    genenergy = read('results/native_iris_natural_expression_v1_telemetry/summary.json')
    observations = read('results/native_iris_natural_expression_observations_v1/audit.json')
    older = read('results/native_iris_molmo_observations_v1/audit.json')
    assert binding['only_action_difference'] == ['prompt_sha256']
    assert observations['actual_calls'] == observations['completed'] == 10
    assert observations['actual_frames_reproduced'] == 8
    rows = sorted([r for r in observations['descriptions'] if r['control'] is None], key=lambda r: r['frame'])
    assert [r['frame'] for r in rows] == plan['review_frames']
    previous = [r for r in older['descriptions'] if r['control'] is None]
    return {'phase': 'post_failure_natural_expression_development_control',
        'appended_sentence': plan['appended_sentence'], 'actual_generation_attempts': 1,
        'comparison': plan['comparison'], 'observations': rows,
        'posthoc_literal_clothing_counts': {word: sum(word in r['parsed']['clothing'].lower() for r in rows)
            for word in ('coat', 'jacket', 'apron')},
        'previous_32_descriptions_containing_coat': sum('coat' in r['parsed']['clothing'].lower() for r in previous),
        'count_scope': 'Literal substring inventory of model clothing text, not ground truth, repair rate or model ranking.',
        'costs': {'native_preparation_seconds': manifest['preparation_wall_seconds'],
            'render_supervisor_seconds': native['status']['wall_seconds'],
            'repair_stage_seconds_nested_in_render': native['repair_stage_seconds'],
            'observation_supervisor_seconds': observations['supervisor_seconds'],
            'observation_worker_seconds_nested': observations['worker_seconds'],
            'observation_model_load_seconds_nested': observations['model_load_seconds'],
            'render_plus_observation_stage_seconds': native['status']['wall_seconds'] + observations['supervisor_seconds'],
            'preparation_render_observation_stage_seconds': manifest['preparation_wall_seconds'] + native['status']['wall_seconds'] + observations['supervisor_seconds'],
            'prior_35_call_diagnostic_supervisor_seconds_separate': older['supervisor_seconds'],
            'scope': 'Separately measured stages, not one newly timed contiguous pipeline. No double counting of nested intervals. Existing original generation, earlier pair, compiler/localization, initial CPU file preparation, provisioning and export are separate.'},
        'render_device_energy_bounds': genenergy['device_energy_bounds'],
        'observation_device_energy_bounds': observations['telemetry']['device_energy_bounds'],
        'energy_scope': 'Device counters include idle consumption; sampled utilization is not kernel GPU time.',
        'engineering': {'prefix_latents_exact': native['prefix_latents_exact'],
            'outside_mask_latents_exact': native['outside_mask_latents_exact'],
            'changed_edited_latents': len(native['changed_edited_latents']),
            'changed_suffix_latents': len(native['changed_suffix_latents']),
            'additional_denoiser_calls': native['additional_denoiser_calls']},
        'candidate_adopted': False, 'controller_acceptance_executed': False,
        'paired_protection_executed': False, 'production_future_guard_executed': False,
        'actual_human_labels': 0, 'independent_script_count': 1, 'held_out': False,
        'quality_benefit_established': False, 'inputs_sha256': inputs,
        'limits': ['One exposed script and seed; not a formal ablation or a new arm of the completed budget pair.',
            'Descriptions of later samples are not a completed controller future guard.',
            'The start-frame description remains apron and colors/type words vary across later views.',
            'All eight fixed samples and full videos are retained; sparse images cannot validate motion or identity.',
            'No retrospective change to the zero-retention outcomes of the earlier pair.']}


def gallery(root, result, output):
    esc = html.escape
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Iris natural-expression development control</title>',
        '<style>body{font:17px system-ui;max-width:1200px;margin:28px auto;padding:0 18px;color:#17232e}img{width:100%;height:auto}video{width:100%;max-width:832px}section{margin:36px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}pre{white-space:pre-wrap;background:#f1f4f6;padding:12px}a{color:#165fb1}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>',
        '<h1>One wording-only development replay</h1>',
        '<p>Same original parent, automatic mask, seed 4244 and sigma 1.0; adds <strong>Iris wears a moss green coat.</strong> One exposed script. No acceptance or human ratings. All eight pre-frozen views are shown, including the unrepaired start.</p>',
        '<p>The four-way video shows original / feedback second / fixed-noise second / natural expression in reading order. It retains all 960 frames at 16 fps. <a href="candidate.mp4">Full-resolution new candidate</a> · <a href="../../results/native_iris_natural_expression_v1/summary.json">Machine-readable result</a></p>',
        '<video controls preload="metadata" src="four_way_comparison.mp4"></video>',
        '<p>Captions below are raw target-blind Molmo outputs, not verified labels. Gray control and repeat transcripts are in the result folder. Still inspection found a block-like distortion at frame 479 in all four videos, including the original; it is not evidence of a newly introduced defect.</p>']
    old = json.loads((root / 'results/native_iris_molmo_observations_v1/audit.json').read_text())
    for row in result['observations']:
        frame = row['frame']
        original, = [r for r in old['descriptions'] if r['branch'] == 'original' and r['frame'] == frame and r['control'] is None]
        body += [f'<section><h2>Frame {frame} · {frame / 16:g} seconds · {esc(row["scope"])}</h2><div class="pair">']
        for label, name, record in [('Original', 'native_iris_molmo_observations_v1', original),
                                     ('Natural expression', 'native_iris_natural_expression_observations_v1', row)]:
            body.append(f'<div><h3>{label}</h3><img width="832" height="480" loading="lazy" src="../../results/{name}/{esc(record["image"])}" alt="{label}, frame {frame}"><pre>{esc(json.dumps(record["parsed"], indent=2))}</pre></div>')
        body.append('</div></section>')
    body.append('</html>')
    output.write_text('\n'.join(body) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gallery', type=Path)
    args = p.parse_args()
    result = summarize(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.gallery:
        gallery(args.root.resolve(), result, args.gallery)
    print(json.dumps({'costs': result['costs'], 'literal_counts': result['posthoc_literal_clothing_counts']}))
