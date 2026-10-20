"""Opt-in bounded name-ID classification before the unchanged typed owner stage.

Capitalization proposes spans, not character identities. Every span requires a
model decision. Neither this finite English inventory nor successful schema
validation proves semantic completeness or coreference correctness.
"""
from copy import deepcopy
import json

from utils.agent_typed_script import TOKEN
from utils.agent_script_candidates import compile_candidate_owners
from utils.agent_runtime import TemporalInput
from utils.agent_vlm import _validate_json

# Keep these tokens as candidates, but do not merge them into adjacent names.
# They are not a list of benchmark characters and are never automatically
# declared noncharacters (a script might use an unusual name).
SEPARATE = set('a an the at in on by with from to for he she it they his her its their this that these those'.split())


def name_candidates(prompts):
    candidates = []
    by_name = {}
    for shot, text in enumerate(prompts):
        tokens = list(TOKEN.finditer(text))
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if not token.group()[0].isupper():
                i += 1
                continue
            start, end = token.start(), token.end()
            previous = token.group()
            i += 1
            while (i < len(tokens) and tokens[i].group()[0].isupper()
                   and text[end:tokens[i].start()].isspace()
                   and previous.lower() not in SEPARATE
                   and tokens[i].group().lower() not in SEPARATE
                   and not previous.lower().endswith("'s")):
                previous = tokens[i].group()
                end = tokens[i].end()
                i += 1
            if text[start:end].lower().endswith("'s"):
                end -= 2
            name = text[start:end]
            if len(name) > 80:
                raise ValueError('Name candidate exceeds bounded length; no truncation')
            span = {'shot': shot, 'start_char': start, 'end_char': end, 'quote': name}
            if name not in by_name:
                if len(candidates) >= 32:
                    raise ValueError('Too many name candidates; no silent truncation')
                row = {'id': f'n{len(candidates):03d}', 'name': name, 'sources': []}
                by_name[name] = row
                candidates.append(row)
            by_name[name]['sources'].append(span)
    return candidates


def classified_cast(prompts, candidates, decisions):
    if candidates != name_candidates(prompts):
        raise ValueError('Name inventory differs from authored script')
    if set(decisions) != {r['id'] for r in candidates}:
        raise ValueError('Every name candidate needs exactly one decision')
    cast, unresolved, audit = [], [], []
    for row in candidates:
        decision = decisions[row['id']]
        kind, presence = decision['kind'], decision['presence']
        if kind not in ('character', 'noncharacter', 'ambiguous'):
            raise ValueError('Unknown name classification')
        if set(presence) != {str(k) for k in range(len(prompts))} or any(v not in ('required', 'ambiguous', 'not_required') for v in presence.values()):
            raise ValueError('Every authored shot needs a valid presence decision')
        if kind == 'noncharacter' and any(v != 'not_required' for v in presence.values()):
            raise ValueError('A noncharacter cannot have depicted character presence')
        first = row['sources'][0]
        source = {'shot': first['shot'], 'quote': first['quote']}
        audit.append({'id': row['id'], 'name': row['name'], **deepcopy(decision)})
        if kind == 'character':
            cast.append({'name': row['name'], 'name_source': source,
                         'presence': {str(k): presence[str(k)] for k in range(len(prompts))}})
        elif kind == 'ambiguous':
            unresolved.append({'entity': None, 'source': source, 'reason': 'Ambiguous name classification', 'name_candidate_id': row['id']})
    if len(cast) > 8:
        raise ValueError('Too many named characters; no silent truncation')
    if not cast:
        unresolved.append({'entity': None, 'source': None,
                           'reason': 'No named character classified; this finite protocol cannot certify a character-free script'})
    return cast, unresolved, audit


def compile_name_candidate_entities(client, prompts):
    candidates = name_candidates(prompts)
    presence = {'type': 'object', 'additionalProperties': False,
                'properties': {str(k): {'type': 'string', 'enum': ['required', 'ambiguous', 'not_required']} for k in range(len(prompts))},
                'required': [str(k) for k in range(len(prompts))]}
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {row['id']: {'type': 'object', 'additionalProperties': False,
                  'properties': {'kind': {'type': 'string', 'enum': ['character', 'noncharacter', 'ambiguous']},
                                 'presence': presence, 'reason': {'type': 'string', 'minLength': 1, 'maxLength': 180}},
                  'required': ['kind', 'presence', 'reason']} for row in candidates},
              'required': [row['id'] for row in candidates]}
    inputs = (TemporalInput.from_value(kind='fixed_authored_script', identifier='native_script', value=prompts,
                                      available_after_shot=-1, authored=True),)
    if candidates:
        response = client.call(prompt=json.dumps({
            'task': 'Classify EVERY immutable name-candidate ID from the original script. Capitalization alone is not evidence of a character. '
                    'Use character for a named person, animal or depicted agent; noncharacter for pronouns, sentence-initial ordinary words, places or other non-agent text; '
                    'use ambiguous when the script does not resolve this. Do not omit, rename or add IDs. '
                    'For a character, report depicted presence in every shot: required, ambiguous, or not_required. Mention in memory does not imply visible presence. '
                    'For noncharacter use not_required in every shot. Do not invent any appearance attributes. Strict JSON.',
            'name_candidates': candidates, 'prompts': {str(k): p for k, p in enumerate(prompts)}}, ensure_ascii=False),
            image_paths=[], schema=schema, call_kind='native_name_candidate_cast_v6', shot_index=0, temporal_inputs=inputs)
        decisions = response['parsed']
        _validate_json(decisions, schema)
    else:
        decisions = {}
    cast, unresolved, audit = classified_cast(prompts, candidates, decisions)
    compiled = compile_candidate_owners(client, prompts, cast)
    compiled['unresolved'].extend(unresolved)
    compiled.update(compiler_protocol='typed_name_candidates_v6', name_candidate_inventory=candidates,
                    name_classification_audit=audit, ready_for_development=not compiled['unresolved'])
    compiled['limits'] += ['Finite capitalization-based English name inventory; lower-case names, aliases and nonlocal reference are not certified.',
                           'Mandatory name-ID decisions remove an empty-list shortcut, not the possibility of wrong classification. No semantic-accuracy claim.']
    return compiled
