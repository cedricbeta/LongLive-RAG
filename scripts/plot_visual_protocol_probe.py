#!/usr/bin/env python3
"""Show all six frozen crops and raw model labels; no accuracy annotation."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def plot(folder, output):
    summary = json.loads((folder / 'summary.json').read_text())
    variant = summary['comparison_arm']
    frames = [p for p in summary['packets'] if len(p['images']) == 1]
    assert len(frames) == 6
    canvas = Image.new('RGB', (1050, 100 + 460 * 2), '#16191e')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'Frozen exposed Iris crops: RAW MODEL LABELS, not accuracy or human annotations', fill='white')
    draw.text((12, 30), 'A: original status-first schema; B: ' + variant, fill='white')
    draw.text((12, 50), 'P=pass  F=fail  U=unknown; joint packet and every full answer remain in the result files.', fill='white')
    aliases = {'pass': 'P', 'fail': 'F', 'unknown': 'U'}
    for index, packet in enumerate(frames):
        x, y = (index % 3) * 350, 90 + (index // 3) * 460
        draw.text((x + 8, y), packet['packet'], fill='white')
        image = Image.open(folder / packet['images'][0]).convert('RGB')
        image = ImageOps.contain(image, (330, 330))
        canvas.paste(image, (x + (350 - image.width) // 2, y + 20))
        for line, requirement in enumerate(packet['requirements']):
            key = requirement['id']
            labels = []
            for arm in ['sampling_only', variant]:
                record = packet['arms'][arm]
                labels.append(aliases[record['parsed'][key]['status']] if record['status'] == 'completed' else 'ERROR')
            label = f"{requirement['part']}/{requirement['aspect']} ({requirement['expected']}): A={labels[0]} B={labels[1]}"
            draw.text((x + 8, y + 355 + line * 19), label, fill='#ecedf0')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    manifest = {'summary_sha256': hashlib.sha256((folder / 'summary.json').read_bytes()).hexdigest(),
                'figure_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'packets': [p['packet'] for p in frames], 'comparison_arm': variant,
                'actual_human_labels': 0, 'new_model_calls': 0, 'accuracy_claim': False}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.folder.resolve(), args.output.resolve())
