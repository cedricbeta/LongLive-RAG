#!/usr/bin/env python3
"""Publish complete paired-observation records without converting them to labels."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def summarize(root):
    inputs = {}
    def read(path):
        inputs[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
        return json.loads((root / path).read_text())
    joint = read('results/native_iris_paired_review_v1/audit.json')
    short = read('results/native_iris_short_pair_descriptions_v1/audit.json')
    assert joint['actual_calls'] == 3 and joint['valid_replies'] == 0
    assert len(joint['unattempted_case_ids']) == 9
    assert short['actual_calls'] == short['valid_replies'] == 36
    assert len(short['comparison_groups']) == 16
    for rows in short['comparison_groups'].values():
        assert len(rows) == 2 and rows[0]['branches']['A'] == rows[1]['branches']['B']
        assert rows[0]['branches']['B'] == rows[1]['branches']['A']
    return {'phase': 'paired_review_development_diagnostics', 'joint_actual_calls': 3,
        'joint_valid_replies': 0, 'joint_unattempted_cases': joint['unattempted_case_ids'],
        'short_actual_calls': 36, 'short_valid_replies': 36,
        'short_controls': short['controls'], 'ordered_comparisons': short['comparison_groups'],
        'costs': {'joint_supervisor_seconds': joint['supervisor_seconds'],
            'short_supervisor_seconds': short['supervisor_seconds'],
            'all_gpu_probe_supervisor_stage_seconds': joint['supervisor_seconds'] + short['supervisor_seconds'],
            'joint_device_energy_bounds': joint['telemetry']['device_energy_bounds'],
            'short_device_energy_bounds': short['telemetry']['device_energy_bounds'],
            'scope': 'Separately measured probe supervisors include file checks, loading and actual calls. Preparation, CPU preprocessing and policy bridge runs are separate; worker/model-load intervals are nested. Energy includes device idle power.'},
        'candidate_adopted': False, 'new_generated_videos': 0, 'actual_human_labels': 0,
        'independent_script_count': 1, 'held_out': False, 'inputs_sha256': inputs,
        'interpretation': 'The joint review fails its output contract and its raw same-sequence answer describes different outfits across identical aligned inputs. The shorter task parses and real-image self controls describe no differences, but the uniform-gray pair invents two outfits. Neither protocol provides a reliable acceptance decision.',
        'limits': ['Raw text variation under swapped order is not automatically a semantic disagreement score.',
            'The shorter task changes input count, prompt, schema and output budget together; not a single-factor causal test.',
            'Descriptions do not establish identity, action, full-motion or whole-suffix protection.',
            'All outputs and the nine unattempted joint cases remain recorded; no malformed result is repaired into a pass.']}


def gallery(result, output):
    esc = html.escape
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Bounded paired clothing observations</title>',
        '<style>body{font:16px system-ui;color:#16222c;max-width:1150px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;background:#f0f3f5;padding:12px}section{border-top:1px solid #aaa;margin:35px 0;padding-top:15px}@media(max-width:640px){.pair{grid-template-columns:1fr}}</style>',
        '<h1>Paired model observations: development only</h1>',
        '<p>All 36 shorter two-image calls are shown below, including four controls and both orders of every comparison. No target outfit, script or branch name was supplied to this shorter task. These are unvalidated model descriptions, not acceptance labels or human ratings.</p>',
        '<p>The earlier multi-frame joint review made three malformed calls and stopped before nine remaining cases. Its identical-sequence control described different outfits on the two sides. Full failures: <a href="../../results/native_iris_paired_review_v1/audit.json">joint audit</a>. The shorter task still invents two outfits on uniform gray input. Neither protocol is adopted.</p>']
    groups = [(f'Control: {r["pair_group"]}', [r]) for r in result['short_controls']]
    groups += [(key, rows) for key, rows in sorted(result['ordered_comparisons'].items())]
    for label, rows in groups:
        body.append('<section><h2>' + esc(label) + '</h2><div class="pair">')
        first = rows[0]
        for slot in ('A', 'B'):
            branch, frame = first['branches'][slot], first['frames'][0]
            file = 'gray.png' if branch == 'gray' else f'{branch}_{frame}.png'
            src = '../../results/native_iris_short_pair_descriptions_v1/inputs/' + file
            body.append(f'<div><h3>{esc(branch)} · frame {frame}</h3><a href="{src}"><img width="832" height="480" src="{src}" alt="Exact model input"></a></div>')
        body.append('</div>')
        for row in rows:
            mapping = ', '.join(f'{k}: {v}' for k, v in row['branches'].items())
            body.append(f'<h3>{esc(row["id"])} · {esc(mapping)}</h3><pre>{esc(row["reply"] or row["error"] or "No reply")}</pre>')
        body.append('</section>')
    body.append('</html>')
    output.parent.mkdir(parents=True, exist_ok=True)
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
        gallery(result, args.gallery)
    print(json.dumps(result['costs']))
