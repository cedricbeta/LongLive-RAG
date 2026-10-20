"""Narrow, opt-in literal hair-color reminder at existing later boundaries.

This prepares a disclosed text-conditioning intervention, not an acceptance
decision or an identity reference. Authored timing and action text are retained.
Unsupported presence grammar fails closed rather than adding an absent actor.
"""
import copy
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compile_cue(schedule, cue, *, earliest_latent):
    if cue is None:
        return schedule, {'enabled': False, 'changes': []}
    if type(earliest_latent) is not int or earliest_latent < 1:
        raise ValueError('A positive later conditioning boundary is required')
    fields = {'schema_version', 'entity', 'hair_color', 'evidence_start_latent', 'apply_at'}
    if (not isinstance(cue, dict) or set(cue) != fields
            or type(cue['schema_version']) is not int or cue['schema_version'] != 1):
        raise ValueError('Unsupported continuation cue schema')
    if not isinstance(schedule, dict) or not schedule or 0 not in schedule:
        raise ValueError('A full authored schedule is required')
    for boundary, prompts in schedule.items():
        if type(boundary) is not int or boundary < 0 or not isinstance(prompts, list) or len(prompts) != 1:
            raise ValueError('Use integer boundaries and one prompt per boundary')
        if not isinstance(prompts[0], str) or not prompts[0].strip():
            raise ValueError('Empty authored prompt')
    entity, color = cue['entity'], cue['hair_color']
    if not isinstance(entity, str) or not re.fullmatch(r'[A-Z][a-z]+', entity):
        raise ValueError('This narrow protocol supports a single explicit given name')
    if not isinstance(color, str) or color not in {'silver', 'black', 'brown', 'blonde', 'red', 'white', 'gray', 'grey'}:
        raise ValueError('Unsupported literal hair color')
    evidence = cue['evidence_start_latent']
    if type(evidence) is not int or evidence not in schedule or evidence >= earliest_latent:
        raise ValueError('Appearance evidence must precede the conditioning intervention')
    definition = re.search(r'\b' + re.escape(entity) + r', (?:an?|the) [^,.;!?]{1,80}\bwith '
                           + re.escape(color) + r' hair\b', schedule[evidence][0])
    if definition is None:
        raise ValueError('Color is not explicitly attached to this named actor in the authored definition')
    boundaries = cue['apply_at']
    if not isinstance(boundaries, list) or not boundaries or any(type(x) is not int for x in boundaries):
        raise ValueError('Explicit later boundaries are required')
    if boundaries != sorted(set(boundaries)) or any(x < earliest_latent or x not in schedule for x in boundaries):
        raise ValueError('Cue cannot alter the prefix, timing or duplicate a boundary')
    result = copy.deepcopy(schedule)
    sentence = entity + ' has ' + color + ' hair.'
    changes = []
    for boundary in boundaries:
        authored = schedule[boundary][0]
        # This intentionally accepts only a narrow explicit present-action clause.
        # A name mentioned in another actor's recollection is not enough.
        subject = re.search(r'(?:^|,\s*)' + re.escape(entity)
                            + r' (?:stands|sits|leans|continues|finishes|reaches|picks|makes|steps|settles)\b', authored)
        departure = re.search(r'\b(?:after|without) ' + re.escape(entity) + r'\b|\b'
                              + re.escape(entity) + r' (?:leaves|left|exits|has left|is absent|is not present)\b', authored)
        if subject is None or departure is not None:
            raise ValueError('The target is not unambiguously present under the supported grammar')
        if sentence in authored:
            raise ValueError('Do not duplicate an already appended cue')
        effective = authored + ' ' + sentence
        result[boundary] = [effective]
        changes.append({'start_latent_frame': boundary, 'authored_prompt': authored,
                        'effective_prompt': effective, 'authored_sha256': digest([authored]),
                        'effective_sha256': digest([effective])})
    record = {'enabled': True, 'protocol': 'literal_hair_color_continuation_v1',
        'cue': copy.deepcopy(cue), 'earliest_latent': earliest_latent,
        'source_quote': definition.group(0), 'sentence': sentence, 'changes': changes,
        'authored_schedule_sha256': digest(schedule), 'effective_schedule_sha256': digest(result),
        'authored_text_preserved_verbatim': True, 'new_prompt_boundaries': 0,
        'new_attribute_or_reference_image': False, 'automatic_acceptance': False}
    return result, record
