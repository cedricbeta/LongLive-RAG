"""Route rejected attribute evidence without changing any acceptance label.

Frame-label heterogeneity is a property of the recorded judge outputs, not proof
of a real temporal change. A permitted revision is still made from the confirmed
failing parent, and the rejected candidate remains quarantined.
"""
from collections import defaultdict

from utils.agent_constraints import ConstraintAssessment, compare_constraint_assessments


def attribute_evidence(assessment):
    groups = defaultdict(set)
    invalid = False
    rechecks = []
    for item in assessment.evidence:
        meta = item.metadata
        if item.source == 'attribute_frame_review_v1':
            digest, status = meta.get('frame_sha256'), meta.get('status')
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or status not in ('pass', 'fail', 'unknown')):
                invalid = True
                continue
            groups[digest].add(status)
        elif item.source == 'attribute_joint_recheck_v1':
            rechecks.append(dict(meta))
    counts = {status: sum(values == {status} for values in groups.values())
              for status in ('pass', 'fail', 'unknown')}
    conflicts = sorted(key for key, values in groups.items() if len(values) > 1)
    if invalid or not groups:
        category = 'missing_or_invalid_frame_evidence'
    elif conflicts:
        category = 'same_frame_judgment_disagreement'
    elif counts['unknown']:
        category = 'insufficient_visibility_or_support'
    elif counts['pass'] and counts['fail']:
        category = 'mixed_frame_labels'
    elif counts['fail'] >= 2:
        category = 'supported_conflict'
    elif counts['pass'] >= 2:
        category = 'supported_agreement'
    else:
        category = 'insufficient_visibility_or_support'
    bounded_recheck = (len(rechecks) == 1
        and rechecks[0].get('status') in ('pass', 'fail')
        and rechecks[0].get('independent_model') is False
        and len(rechecks[0].get('frames', [])) == len(groups)
        and {r.get('sha256') for r in rechecks[0].get('frames', [])} == set(groups))
    return {'category': category, 'distinct_frame_counts': counts,
            'same_frame_conflicting_sha256': conflicts,
            'bounded_recheck_completed': bounded_recheck,
            'joint_recheck_statuses': [r.get('status') for r in rechecks]}


def mixed_target_revision(parent, candidate, target_ids):
    """Permit at most a policy-bounded changed action; never accept unknown.

Every target needs a supported failing parent. Every candidate unknown must be
an eligible target or its explicitly bound authored-attribute summary. Critical
unknowns, new non-target unknowns, absent rechecks and same-frame disagreement
stop. No free-text reason or confidence threshold controls this route.
"""
    before = {r['constraint']['id']: ConstraintAssessment.from_dict(r)
              for r in parent.metrics.get('constraint_assessments', [])}
    after = {r['constraint']['id']: ConstraintAssessment.from_dict(r)
             for r in candidate.metrics.get('constraint_assessments', [])}
    targets = set(target_ids)
    report = {'allow_revision': False, 'targets': sorted(targets), 'attributes': {},
              'blocked_by': [], 'derived_summaries': [], 'acceptance_changed': False,
              'interpretation': 'Recorded label heterogeneity, not verified temporal change or accuracy.'}
    if (len(before) != len(parent.metrics.get('constraint_assessments', []))
            or len(after) != len(candidate.metrics.get('constraint_assessments', []))
            or not targets or not targets <= before.keys() or set(before) != set(after)
            or any(before[key].constraint != after[key].constraint for key in before)):
        report['blocked_by'].append('missing_or_changed_constraints')
        return report
    if not compare_constraint_assessments(parent, candidate).allow_commit:
        report['blocked_by'].append('constraint_regression')
    eligible = set()
    for key in sorted(targets):
        old, new = before[key], after[key]
        old_evidence, new_evidence = attribute_evidence(old), attribute_evidence(new)
        report['attributes'][key] = {'parent': old_evidence, 'candidate': new_evidence}
        if (old.constraint.kind.value != 'attribute'
                or old.constraint.parameters.get('part') not in ('hair', 'garment')
                or old.status != 'violated' or old_evidence['category'] != 'supported_conflict'):
            report['blocked_by'].append('parent_target_not_supported:' + key)
        if (new.status == 'unknown' and new_evidence['category'] == 'mixed_frame_labels'
                and new_evidence['bounded_recheck_completed']):
            eligible.add(key)
        elif new.status == 'unknown':
            report['blocked_by'].append('unresolved_target_evidence:' + key)
    for key, row in after.items():
        if row.status != 'unknown' or key in eligible:
            continue
        members = [r for r in after.values() if r.constraint.kind.value == 'attribute'
                   and r.constraint.parameters.get('basis') == 'fixed_script_quote'
                   and r.constraint.subjects == row.constraint.subjects
                   and r.constraint.shot_indices == row.constraint.shot_indices]
        expected = {r.id for r in members}
        old_members = [r for r in before.values() if r.id in expected]
        bindings = [e.metadata for e in row.evidence
                    if e.metadata.get('aggregation') == 'authored_attribute_summary_v1']
        old_bindings = [e.metadata for e in before[key].evidence
                       if e.metadata.get('aggregation') == 'authored_attribute_summary_v1']
        derived = (row.constraint.kind.value == 'attribute'
            and row.constraint.parameters.get('comparison_basis') == 'script_description'
            and before[key].status == 'violated' and expected
            and len(bindings) == len(old_bindings) == 1
            and set(bindings[0].get('attribute_constraint_ids', [])) == expected
            and set(old_bindings[0].get('attribute_constraint_ids', [])) == expected
            and any(r.status == 'violated' for r in old_members)
            and any(r.status == 'unknown' for r in members)
            and all(r.status == 'satisfied' or r.id in eligible for r in members))
        if derived:
            report['derived_summaries'].append(key)
        else:
            report['blocked_by'].append('non_target_or_critical_unknown:' + key)
    if not eligible:
        report['blocked_by'].append('no_rechecked_mixed_target')
    report['allow_revision'] = bool(eligible) and not report['blocked_by']
    return report
