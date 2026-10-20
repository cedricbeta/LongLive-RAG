#!/usr/bin/env python3
"""Extend exposure audit with character and named-cast structure, not split certification."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(corpus, registry, exposure):
    scripts = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    known = json.loads(registry.read_text())
    names = sorted({row['name'] for values in known.values() if isinstance(values, list) for row in values})
    inventory = json.loads(exposure.read_text())
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != inventory['corpus_sha256']:
        raise ValueError('Exposure audit and corpus hashes differ')
    exposed = set(inventory['summary']['direct_exposure_indices'])
    name_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(n) for n in names) + r')\b')
    rows = []
    for index, script in enumerate(scripts):
        seen = []
        cast = []
        normalized = []
        for prompt in script['prompts']:
            matches = name_pattern.findall(prompt)
            for name in matches:
                if name not in seen:
                    seen.append(name)
            cast.append(sorted({seen.index(n) for n in matches}))
            normalized.append(name_pattern.sub(lambda m: f'PERSON_{seen.index(m.group())}', prompt))
        rows.append({'index': index, 'direct_exposure': index in exposed,
            'characters_in_first_mention_order': seen,
            'named_cast_by_shot': cast, 'named_cast_signature': digest(cast),
            'person_normalized_script_sha256': digest(normalized), 'held_out_certified': False})
    exposed_characters = {n for row in rows if row['direct_exposure'] for n in row['characters_in_first_mention_order']}
    exposed_cast = {row['named_cast_signature'] for row in rows if row['direct_exposure']}
    groups = collections.defaultdict(list)
    for row in rows:
        row['overlaps_development_characters'] = bool(set(row['characters_in_first_mention_order']) & exposed_characters)
        row['all_characters_development_exposed'] = bool(row['characters_in_first_mention_order']) and set(row['characters_in_first_mention_order']) <= exposed_characters
        row['overlaps_development_named_cast_signature'] = row['named_cast_signature'] in exposed_cast
        groups[row['named_cast_signature']].append(row['index'])
    capital_words = collections.Counter(w for script in scripts for p in script['prompts']
        for w in re.findall(r'\b[A-Z][a-z]+\b', p) if w not in names)
    return {'schema_version': 1, 'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'registry_sha256': hashlib.sha256(registry.read_bytes()).hexdigest(),
        'exposure_audit_sha256': hashlib.sha256(exposure.read_bytes()).hexdigest(),
        'registry_use': 'Audit only; this metadata is not passed to any generator or repair planner.',
        'rows': rows, 'named_cast_groups': dict(groups),
        'summary': {'scripts':len(rows), 'known_characters':names, 'development_characters':sorted(exposed_characters),
            'all_characters_exposed_count':sum(r['all_characters_development_exposed'] for r in rows),
            'empty_named_cast_scripts':[r['index'] for r in rows if not r['characters_in_first_mention_order']],
            'named_cast_signature_groups':len(groups),
            'named_cast_overlap_count':sum(r['overlaps_development_named_cast_signature'] for r in rows),
            'person_normalized_exact_script_groups':len({r['person_normalized_script_sha256'] for r in rows})},
        'capital_words_outside_registry':dict(capital_words.most_common()),
        'limits':['A named mention is not a visual presence label; pronouns and implicit persistence are not resolved.',
            'Equal cast signatures do not imply equal narratives; unequal wording does not establish independent templates.',
            'These hashes and character overlaps cannot certify a held-out split.',
            'Need original benchmark narrative-blueprint metadata or separately reviewed grouping before formal generalization claims.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--exposure',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = audit(args.corpus,args.registry,args.exposure)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
