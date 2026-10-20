"""Derive one bounded garment sentence from active, quoted script requirements."""
from __future__ import annotations

import re


def authored_garment_prompt(base, requirements, compiled, shot):
    """Return a joint color/type expression, or no action if binding is unclear.

    The phrase must already occur contiguously in each active source quote and
    the current authored prompt. This verifies text provenance, not the semantic
    accuracy of the compiler's entity attribution or the generator's output.
    """
    if type(shot) is not int or not isinstance(compiled, dict):
        return None
    prompts = compiled.get('prompts', [])
    if not 0 <= shot < len(prompts) or not base.startswith(prompts[shot]):
        return None
    if len(requirements) != 2:
        return None
    values, names = {}, set()
    for row in requirements:
        params = row.get('parameters', {})
        subjects = row.get('subjects', [])
        aspect, value = params.get('aspect'), row.get('expected')
        if (params.get('basis') != 'fixed_script_quote' or params.get('part') != 'garment'
                or aspect not in ('color', 'type') or aspect in values or len(subjects) != 1
                or not isinstance(value, str) or not value.strip() or value != value.strip()):
            return None
        names.add(subjects[0])
        values[aspect] = value
    if len(names) != 1 or set(values) != {'color', 'type'}:
        return None
    name = next(iter(names))
    entities = [e for e in compiled.get('entities', []) if e.get('name') == name]
    if len(entities) != 1 or not isinstance(name, str) or not name.strip():
        return None
    phrase = values['color'] + ' ' + values['type']
    literal = re.compile(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)')
    if not literal.search(prompts[shot]):
        return None
    for aspect, value in values.items():
        active = []
        for row in entities[0].get('appearance', []):
            if row.get('part') != 'garment' or row.get('aspect') != aspect:
                continue
            start, end = row.get('source', {}).get('shot'), row.get('last_active_shot')
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(prompts):
                return None
            if start <= shot <= end:
                active.append(row)
        if len(active) != 1 or active[0].get('expected') != value:
            return None
        source = active[0]['source']
        quote = source.get('quote')
        if not isinstance(quote, str) or quote not in prompts[source['shot']] or not literal.search(quote):
            return None
        # Reject an explicitly mixed-character quote; pronoun attribution remains
        # the upstream compiler's documented semantic responsibility.
        if any(e.get('name') != name and re.search(r'(?<!\w)' + re.escape(e['name']) + r'(?!\w)', quote)
               for e in compiled.get('entities', []) if isinstance(e.get('name'), str)):
            return None
    # Prefer an authored determiner. Otherwise use the bounded grammatical rule
    # needed for a singular garment; plural or unknown garment heads are refused.
    from utils.agent_typed_script import GARMENTS
    kind = values['type'].lower()
    if kind not in GARMENTS or kind in {'trousers', 'pants', 'shorts', 'jeans', 'shoes', 'boots', 'gloves', 'socks'}:
        return None
    determiner = re.search(r'\b(a|an)\s+' + re.escape(phrase) + r'(?!\w)', prompts[shot], re.I)
    article = determiner.group(1).lower() if determiner else ('an' if phrase[0].lower() in 'aeiou' else 'a')
    sentence = f'{name} wears {article} {phrase}.'
    if sentence in base:
        return None
    return base.rstrip() + ' ' + sentence
