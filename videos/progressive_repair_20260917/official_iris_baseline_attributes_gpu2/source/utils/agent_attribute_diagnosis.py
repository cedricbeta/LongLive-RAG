"""Opt-in per-requirement appearance evidence, with one bounded visual recheck.

Requirements are compiled once from the fixed description, before images are
seen. Exact source spans prohibit adding hidden attributes. Extraction can still
omit requirements and needs independent auditing; this is a development judge,
not an independent quality evaluator. Identity/action/quality guards are separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.agent_entity_judge import VLMComparer
from utils.agent_runtime import file_sha256

PARTS = ('hair', 'garment', 'face', 'body', 'other')
ASPECTS = ('color', 'style', 'type', 'age', 'build', 'detail')
STATUSES = ('pass', 'fail', 'unknown')
REQUIREMENTS_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'requirements': {'type': 'array', 'maxItems': 24, 'items': {
        'type': 'object', 'additionalProperties': False,
        'properties': {'part': {'type': 'string', 'enum': list(PARTS)},
                       'aspect': {'type': 'string', 'enum': list(ASPECTS)},
                       'expected': {'type': 'string', 'minLength': 1, 'maxLength': 240}},
        'required': ['part', 'aspect', 'expected']}}},
    'required': ['requirements'],
}


def freeze_requirements(description, rows):
    """Stable IDs include the exact normative quote; never the observed image."""
    result = []
    seen = set()
    for row in rows:
        part, aspect, expected = row['part'], row['aspect'], row['expected']
        if part not in PARTS or aspect not in ASPECTS or not expected.strip() or expected not in description:
            raise ValueError('Requirement must quote an exact nonempty script span and valid part')
        key = (part, aspect, expected)
        if key in seen:
            raise ValueError('Duplicate requirement')
        seen.add(key)
        identifier = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        result.append({'id': identifier, 'part': part, 'aspect': aspect, 'expected': expected})
    if not result:
        # No appearance claim can be proved by an empty extraction.
        raise ValueError('Empty requirement extraction is not a passing assessment')
    return sorted(result, key=lambda row: row['id'])


def validate_required_aspects(description, requirements):
    """Catch known omissions without pretending the language parser is complete."""
    from utils.agent_appearance_color import parse_appearance_spec
    spec = parse_appearance_spec(description)
    required = set()
    for part, value in spec.items():
        if value.get('colors'):
            required.add((part, 'color'))
        if value.get('styles'):
            required.add((part, 'style'))
        if value.get('type'):
            required.add((part, 'type'))
    missing = required - {(r['part'], r['aspect']) for r in requirements}
    if missing:
        raise ValueError(f'Incomplete attribute extraction: missing {sorted(missing)}')


def assessment_schema(requirements):
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {row['id']: {
            'type': 'object', 'additionalProperties': False,
            'properties': {'status': {'type': 'string', 'enum': list(STATUSES)},
                           'observed': {'type': 'string', 'minLength': 1, 'maxLength': 350},
                           'reason': {'type': 'string', 'minLength': 1, 'maxLength': 350}},
            'required': ['status', 'observed', 'reason'],
        } for row in requirements},
        'required': [row['id'] for row in requirements],
    }


def aggregate(requirements, frame_reviews, *, minimum_support=2):
    """Disagreement remains unknown; confidence strings never overwrite evidence."""
    result = []
    for requirement in requirements:
        evidence = []
        for frame in frame_reviews:
            value = frame['assessments'][requirement['id']]
            if value['status'] not in STATUSES:
                raise ValueError('Invalid assessment status')
            evidence.append({**value, 'path': frame['path'], 'sha256': frame['sha256']})
        # Count distinct image bytes rather than duplicated crop requests.
        unique = {item['sha256']: item for item in evidence}
        passes = sum(e['status'] == 'pass' for e in unique.values())
        fails = sum(e['status'] == 'fail' for e in unique.values())
        if passes and fails:
            status, reason = 'unknown', 'judgment_disagreement'
        elif fails >= minimum_support:
            status, reason = 'fail', 'supported_conflict'
        elif passes >= minimum_support and all(e['status'] == 'pass' for e in unique.values()):
            status, reason = 'pass', 'supported_agreement'
        else:
            status, reason = 'unknown', 'insufficient_visibility_or_support'
        result.append({**requirement, 'status': status, 'reason': reason, 'evidence': evidence})
    return result


class AttributeEvidenceComparer(VLMComparer):
    """Experimental script-attribute checker. Unknown cannot register a reference.

    The optional single recheck sees real crops and fixed requirements only,
    never a planner's explanation or previous model answers. It cannot establish
    independent confirmation, and every initial/recheck observation is retained.
    """
    def __init__(self, client, *, max_rechecks=1):
        super().__init__(client)
        if max_rechecks not in (0, 1):
            raise ValueError('Only zero or one evidence recheck is supported')
        self.max_rechecks = max_rechecks
        self._requirements = {}

    def requirements(self, entity, *, shot_index, inputs):
        key = (entity.name, entity.description)
        if key not in self._requirements:
            response = self.client.call(
                prompt=json.dumps({
                    'task': 'Extract every explicit visible appearance requirement from the fixed script description. '
                            'Split separable attributes (hair color, hair style, garment color/type, age/build). '
                            'Each row has one aspect: color, style, type, age, build, or detail. '
                            'For curly red hair return separate hair/style and hair/color rows. '
                            'For a green coat return separate garment/color and garment/type rows. '
                            'Each expected value must be an exact contiguous quote of the description. '
                            'Do not add inferred traits, actions, or references; include broad subject type. '
                            'part is hair, garment, face, body, or other. Strict JSON.',
                    'description': entity.description,
                }), image_paths=[], schema=REQUIREMENTS_SCHEMA, call_kind='attribute_requirements_v2',
                shot_index=shot_index, temporal_inputs=inputs,
            )
            compiled = freeze_requirements(entity.description, response['parsed']['requirements'])
            validate_required_aspects(entity.description, compiled)
            self._requirements[key] = compiled
        return [dict(row) for row in self._requirements[key]]

    def _review(self, requirements, crops, *, shot_index, inputs, recheck=False):
        response = self.client.call(
            prompt=json.dumps({
                'task': 'Assess every fixed appearance requirement against the actual attached crops. '
                        'Each row evaluates only its stated aspect. Color agreement is not evidence of '
                        'a matching hairstyle or garment type. Judge these separately. '
                        'Return pass only for visibly supported agreement, fail only for a clear visible conflict, '
                        'and unknown for blur, occlusion, ambiguity, or contradictory evidence across frames. '
                        'State what is actually observed and why, including expected versus visible for fail. '
                        'Do not infer hidden garment length from a partial view. Natural lighting variation '
                        'is allowed. Do not assume the requested appearance is present. Strict JSON.',
                'requirements': requirements, 'joint_recheck': recheck,
            }), image_paths=[str(p) for p in crops], schema=assessment_schema(requirements),
            call_kind='attribute_joint_recheck_v1' if recheck else 'attribute_frame_review_v1',
            shot_index=shot_index, temporal_inputs=inputs,
        )
        from utils.agent_vlm import _validate_json
        _validate_json(response['parsed'], assessment_schema(requirements))
        return response['parsed']

    def compare(self, entity, reference_crop, crops, *, shot_index, inputs):
        requirements = self.requirements(entity, shot_index=shot_index, inputs=inputs)
        frames = [{'path': str(path), 'sha256': file_sha256(path),
                   'assessments': self._review(requirements, [path], shot_index=shot_index, inputs=inputs)}
                  for path in crops]
        attributes = aggregate(requirements, frames)
        rechecks = []
        uncertain = [r for r in requirements if any(a['id'] == r['id'] and a['status'] == 'unknown' for a in attributes)]
        if uncertain and crops and self.max_rechecks:
            review = self._review(uncertain, crops, shot_index=shot_index, inputs=inputs, recheck=True)
            for item in attributes:
                if item['id'] in review:
                    observation = review[item['id']]
                    item['initial_status'], item['initial_reason'] = item['status'], item['reason']
                    item.update(status=observation['status'], reason='bounded_visual_recheck')
                    item['recheck'] = {**observation, 'frames': [{'path': f['path'], 'sha256': f['sha256']} for f in frames]}
            rechecks.append({'assessments': review, 'independent_model': False})
        # Reference identity remains a separate guard; authored attributes are
        # assessed against the fixed script even after reference registration.
        identity = None
        if reference_crop and crops:
            identity = super().compare(entity, reference_crop, crops, shot_index=shot_index, inputs=inputs)
        all_pass = bool(attributes) and all(a['status'] == 'pass' for a in attributes)
        matches = all_pass and (identity is None or identity['matches_reference'])
        votes = [all(v['status'] == 'pass' for v in f['assessments'].values()) for f in frames]
        return {
            'matches_reference': matches, 'confidence': 1.0 if matches else 0.0,
            'confidence_basis': 'categorical_evidence_not_probability',
            'differences': '; '.join(f"{a['part']} {a['expected']}: {a['status']} ({a['reason']})"
                                     for a in attributes if a['status'] != 'pass'),
            'target_phrase': '', 'crop_votes': votes, 'crop_paths': [str(p) for p in crops],
            'reference_registration_eligible': matches and len({f['sha256'] for f in frames}) >= 3 and all(votes),
            'attribute_assessments': attributes, 'attribute_rechecks': rechecks,
            'identity_review': identity,
            'vetoed_parts': sorted({a['part'] for a in attributes if a['status'] == 'fail' and a['part'] in ('hair', 'garment')}),
        }
