#!/usr/bin/env python3
"""Publish literal output inventory and all fixed frames; never assign truth labels."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path


def summarize(audit_path):
    audit = json.loads(audit_path.read_text())
    rows = [v for v in audit['descriptions'] if v['control'] is None]
    result = {'phase': 'posthoc_literal_description_inventory',
        'source_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'coding_rule': 'Case-insensitive literal substring coat/apron in the model clothing field; post hoc descriptive inventory, not truth labels or accuracy. Controls excluded.',
        'rows': [], 'actual_human_labels': 0, 'repair_benefit_established': False}
    for branch in ['original', 'shared_first', 'feedback_second', 'fixed_second']:
        values = [v for v in rows if v['branch'] == branch]
        result['rows'].append({'branch': branch, 'descriptions': len(values),
            'contains_coat': sum('coat' in v['parsed']['clothing'].lower() for v in values if v['parsed']),
            'contains_apron': sum('apron' in v['parsed']['clothing'].lower() for v in values if v['parsed'])})
    return result


def gallery(folder, output):
    data = json.loads((folder / 'audit.json').read_text())
    branches = ['original', 'shared_first', 'feedback_second', 'fixed_second']
    labels = ['Original', 'Shared first candidate', 'Feedback second candidate', 'Fixed-noise second candidate']
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Iris: target-blind development observations</title>',
        '<style>body{font:16px system-ui;margin:24px;color:#17202a;background:#fafafa}main{max-width:1500px;margin:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:850px}th,td{vertical-align:top;border:1px solid #bbb;padding:10px;width:25%}th{background:#e7edf4}img{display:block;width:100%;height:auto}p{line-height:1.45}small{color:#465568}.frame{font-weight:bold;margin-bottom:8px}</style><main>',
        '<h1>Target-blind model observations on exposed Iris branches</h1>',
        '<p>Molmo2-O-7B describes each full image without script targets, branch names or planner/checker output. These are raw development model descriptions, not human labels, calibrated truth or accepted repairs. All four branches and all eight fixed times are shown. Later frames are a diagnostic; the production future guard was not reached.</p>',
        '<p>One script; 32 correlated branch/time views, plus a gray control and two exact-image repeats. No candidate decision changed. Click any image for the exact input PNG.</p>',
        '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + label + '</th>' for label in labels) + '</tr></thead><tbody>']
    for frame in [320, 352, 400, 448, 479, 672, 720, 768]:
        lines.append('<tr>')
        for branch in branches:
            row, = [r for r in data['descriptions'] if r['branch'] == branch and r['frame'] == frame and r['control'] is None]
            relative = html.escape(os.path.relpath(folder / row['image'], output.parent), quote=True)
            lines.append(f'<td><div class="frame">{frame / 16:g} s · frame {frame} · {row["scope"]}</div><a href="{relative}"><img src="{relative}" alt="Exact observation input"></a>')
            if row['parsed']:
                for key in ['clothing', 'hair', 'uncertainty']:
                    lines.append('<p><strong>' + key.title() + ':</strong> ' + html.escape(row['parsed'][key]) + '</p>')
            else:
                lines.append('<p>' + html.escape(row['reply'] or row['error'] or 'No response') + '</p>')
            lines.append('<small>' + html.escape(row['id']) + '</small></td>')
        lines.append('</tr>')
    lines += ['</tbody></table></div><h2>Constructed controls</h2>',
        '<p>Gray input: zero people reported. The two repeated images reproduce processed tensors and generated tokens exactly. These checks establish neither semantic accuracy nor identity reliability. Full transcripts and controls are in the linked audit.</p>',
        '<p><a href="' + html.escape(os.path.relpath(folder / 'audit.json', output.parent), quote=True) + '">Full machine-readable audit</a></p></main></html>']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gallery', type=Path)
    args = p.parse_args()
    result = summarize(args.folder / 'audit.json')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.gallery:
        gallery(args.folder.resolve(), args.gallery.resolve())
    print(json.dumps(result))
