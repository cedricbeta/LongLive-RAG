#!/usr/bin/env python3
"""Conservative exposure inventory; context overlap is not a full template classifier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

CONTEXTS = (
    'library reading room', 'cafe corner table', 'studio equipment desk', 'workshop workbench',
    'rooftop bar high table', 'market produce table', 'office meeting table', 'classroom front table',
    'restaurant prep counter', 'hotel lobby sofa area', 'flower shop wrapping table', 'lecture hall front row',
)


def audit(root):
    corpus = root / 'example/narrastream_bench/streambench_324.jsonl'
    subset = root / 'example/narrastream_bench/multitext_subset33_b8/manifest.json'
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    exposed = {row['index'] for row in json.loads(subset.read_text())['samples']}
    evidence = {i: ['development_subset33_manifest'] for i in exposed}
    # Look for actual video/ledger/summary artifacts. Archived source trees,
    # copied scripts and dataset directories do not establish output exposure.
    ignored = {'data', 'frozen', 'checkpoints', 'wan_models', 'longlive_models', '.git',
               'remote_verified_checkout', 'vlm_cache', '__pycache__', 'site-packages'}
    for folder, dirs, files in os.walk(root / 'videos', followlinks=False):
        dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(('source', 'env_', 'official_'))]
        for name in files:
            if not (name.endswith('.mp4') or name in ('ledger.jsonl', 'summary.json')):
                continue
            path = Path(folder) / name
            matches = {int(value) for value in re.findall(r'ns(\d{3})(?=[_\W])', str(path.relative_to(root)))}
            for index in matches:
                if index < len(rows):
                    exposed.add(index)
                    evidence.setdefault(index, []).append(str(path.relative_to(root)))
    assigned = []
    for index, row in enumerate(rows):
        text = ' '.join(row['prompts']).lower()
        contexts = [context for context in CONTEXTS if context in text]
        assigned.append({'index': index, 'script_sha256': hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()).hexdigest(), 'direct_exposure': index in exposed,
            'context_groups': contexts, 'exposure_evidence': sorted(set(evidence.get(index, [])))})
    seen_contexts = {c for row in assigned if row['direct_exposure'] for c in row['context_groups']}
    for row in assigned:
        row['overlaps_development_context'] = bool(set(row['context_groups']) & seen_contexts)
        row['held_out_certified'] = False
    return {'schema_version': 1, 'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'subset_manifest_sha256': hashlib.sha256(subset.read_bytes()).hexdigest(),
        'rows': assigned, 'summary': {'scripts': len(rows), 'direct_exposure_count': len(exposed),
            'direct_exposure_indices': sorted(exposed), 'development_contexts': sorted(seen_contexts),
            'overlapping_context_count': sum(row['overlaps_development_context'] for row in assigned),
            'unclassified_context_count': sum(not row['context_groups'] for row in assigned)},
        'interpretation': 'No independent held-out split is certified by this inventory. Public corpus results must disclose overlap.',
        'limitations': ['Conservative subset inclusion even if a particular output was not viewed.',
            'Artifact path discovery may miss external runs or differently named files.',
            'Exact scene-context strings are a lower-bound overlap audit, not a complete narrative-template classifier.',
            'Shared character and action archetypes need separate grouping before any generalization claim.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
