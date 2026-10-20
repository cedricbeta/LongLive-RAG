"""Opt-in offline partial-improvement transactions over parent-latent replay.

The default uses a fixed recipe and at most one noise retry per parent/target.
An explicit bounded-feedback option selects finite program-built alternatives.
Visual protection and finite future checks remain development judges, not
independent evaluation; neither mode establishes learned behavior or efficacy.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from utils.agent_constraints import ConstraintAssessment, compare_constraint_assessments
from utils.agent_runtime import Action, CausalShotSession, ShotResult, TemporalInput, Verdict, file_sha256, fingerprint


PROTECTED_DOMAINS = ('identity', 'other_entities', 'non_target_attributes', 'required_actions', 'scene', 'visual_quality')
PROTECTION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {name: {'type': 'object', 'additionalProperties': False,
        'properties': {'status': {'type': 'string', 'enum': ['pass', 'fail', 'unknown']},
                       'observation': {'type': 'string', 'minLength': 1, 'maxLength': 400}},
        'required': ['status', 'observation']} for name in PROTECTED_DOMAINS},
    'required': list(PROTECTED_DOMAINS),
}


def assessments(verdict):
    rows = verdict.metrics.get('constraint_assessments')
    if not rows:
        raise ValueError('Progressive acceptance requires nonempty explicit constraints')
    result = {}
    for row in rows:
        item = ConstraintAssessment.from_dict(row)
        if item.id in result:
            raise ValueError('Duplicate constraint ID')
        result[item.id] = item
    return result


def improvement_decision(parent, candidate, original, target_ids):
    """Require observed fail->pass; unchanged old failures remain in the ledger."""
    before, after, anchor = assessments(parent), assessments(candidate), assessments(original)
    pair = compare_constraint_assessments(parent, candidate).to_dict()
    fixed = compare_constraint_assessments(original, candidate).to_dict()
    reasons = []
    if not pair['allow_commit'] or not fixed['allow_commit']:
        reasons.append('constraint_regression_or_definition_change')
    improved = [key for key in target_ids if key in before and key in after and
                before[key].status == 'violated' and after[key].status == 'satisfied']
    if not improved:
        reasons.append('target_not_confirmed_improved')
    for key in before.keys() & after.keys():
        if before[key].status == 'violated' and after[key].status == 'unknown':
            reasons.append('existing_failure_became_unobservable:' + key)
        if after[key].constraint.kind.value in ('presence', 'identity', 'action', 'relation', 'count', 'state') and after[key].status == 'unknown':
            reasons.append('critical_unknown:' + key)
    unresolved = sorted(key for key, row in after.items() if row.status != 'satisfied')
    return {'allow': not reasons, 'reasons': reasons, 'improved': improved, 'unresolved': unresolved,
            'fully_verified_current_shot': not unresolved and bool(candidate.accept),
            'parent_comparison': pair, 'fixed_original_comparison': fixed}


class PairedVisualProtection:
    """Same-time parent/candidate images, immutable original anchor, no planner prose."""
    def __init__(self, client, protocol='v1'):
        self.client = client
        if protocol not in ('v1', 'v2'):
            raise ValueError('Unknown paired protection protocol')
        self.protocol = protocol

    def review(self, parent, candidate, original, normative, *, target_requirements=(), references=(), entity_specs=None,
               protected_requirements=()):
        # Stable hash ordering prevents always presenting the candidate second.
        swapped = int(fingerprint([parent.input_fingerprint, candidate.input_fingerprint])[:8], 16) % 2
        ordered = [('candidate', candidate), ('parent', parent)] if swapped else [('parent', parent), ('candidate', candidate)]
        images, labels, temporal = [], [], []
        for index, (role, observation) in enumerate(ordered):
            slot = 'A' if index == 0 else 'B'
            for frame_index, path in enumerate(observation.frames):
                images.append(str(path))
                labels.append({'image': len(images), 'slot': slot, 'sequence_index': frame_index})
                temporal.append(TemporalInput(kind='paired_protection_frame', identifier=str(path),
                    sha256=file_sha256(path), available_after_shot=observation.shot_index))
        if not parent.frames or len(parent.frames) != len(candidate.frames) or len(original.frames) != len(parent.frames):
            raise ValueError('Paired protection requires aligned frame sequences')
        original_equals_parent = ([file_sha256(p) for p in original.frames] == [file_sha256(p) for p in parent.frames])
        original_frames = () if self.protocol == 'v2' and original_equals_parent else original.frames
        for path in original_frames:
            images.append(str(path))
            labels.append({'image': len(images), 'slot': 'fixed_original_anchor'})
            temporal.append(TemporalInput(kind='fixed_original_frame', identifier=str(path),
                sha256=file_sha256(path), available_after_shot=original.shot_index))
        for reference in references:
            path = reference['path']
            expected = reference.get('provenance', {}).get('reference_crop_sha256')
            if expected and file_sha256(path) != expected:
                raise ValueError('Trusted identity reference changed')
            images.append(str(path))
            labels.append({'image': len(images), 'slot': 'trusted_reference'})
            temporal.append(TemporalInput(kind='fixed_identity_reference', identifier=str(path),
                sha256=file_sha256(path), available_after_shot=int(reference.get('provenance', {}).get('shot_index', original.shot_index))))
        packet = {
            'parent_slot': 'B' if swapped else 'A', 'candidate_slot': 'A' if swapped else 'B',
            'fixed_requirements': normative.to_dict(), 'intended_target_requirements': list(target_requirements),
            'fixed_entity_descriptions': dict(entity_specs or {}),
            'images': labels,
            'instruction': 'Compare the actual candidate with the parent and fixed original anchor. '
                'No repair success is assumed. For EACH protected domain report pass only when there is '
                'sufficient visible evidence that the candidate introduces no new defect or degradation '
                'relative to the parent AND no cumulative identity/content degradation from the original '
                'anchor. Judge against the fixed script, not resemblance to an incorrect attribute in the '
                'original. Existing unresolved script errors may remain; do not demand that every old error '
                'be fixed. A change toward the authored appearance is allowed, but a changed face, lost '
                'character, impaired required action, or new artifact must fail. Do not offset damage with '
                'a better target color. Unknown visibility, uncertain correspondence, or insufficient action '
                'evidence must be unknown, never pass. Only frames shown are checked; no claim about unseen '
                'times. Return factual observations, strict JSON.',
        }
        if self.protocol == 'v2':
            packet.update(protocol='explicit_domain_scopes_v2',
                fixed_original_equals_parent=original_equals_parent,
                non_target_script_predicates=[row for row in protected_requirements
                    if row.get('kind') == 'attribute' and row.get('parameters', {}).get('part')],
                domain_definitions={
                    'identity': 'Compare the depicted person using visible facial structure (eyes, nose, jaw), apparent age and build. '
                        'Do not use the intended target attribute as an identity test. In particular, when hair color is the target, '
                        'a hair color change alone cannot establish a changed person. If the face or correspondence cannot be judged, return unknown.',
                    'other_entities': 'Check whether another required actor is lost or changed, or an extra actor appears. '
                        'If both versions visibly contain only the same single required actor and no extra actor, this check passes; '
                        'the absence of additional actors is not missing evidence by itself.',
                    'non_target_attributes': 'Compare the concrete non-target script predicates listed in this packet. '
                        'For example clothing color and clothing type remain non-target predicates when only hair color is targeted. '
                        'An unchanged pre-existing clothing error does not count as new damage. A new loss of a previously correct predicate fails.',
                    'required_actions': 'Check the authored action in the sampled frame sequence. New damage to the visible required '
                        'action fails; insufficient temporal evidence is unknown. Do not infer correctness of unsampled times.',
                    'scene': 'Check the scene layout, relevant props and required relationships for new damage against parent and original.',
                    'visual_quality': 'Check for new visible distortion, seams, duplication, blur or other artifacts.',
                },
                instruction='Evaluate each defined protection predicate from the supplied actual images and fixed script. '
                    'The target requirements specify permitted intent, not a claim that the repair succeeded. '
                    'Decide identity from the defined identity cues, and assess every non-target predicate separately from the target attribute. '
                    'Report pass only with visible evidence of no new damage versus BOTH parent and fixed original; report fail for visible damage, '
                    'unknown for insufficient evidence. A domain with no applicable extra objects may pass only when absence is visible. '
                    'Do not require all existing script errors to be repaired. Describe what you actually see. Strict JSON.')
        inputs = tuple(temporal) + parent.inputs + candidate.inputs + original.inputs
        response = self.client.call(prompt=json.dumps(packet, sort_keys=True, ensure_ascii=False),
            image_paths=images, schema=PROTECTION_SCHEMA, call_kind='paired_progressive_protection_' + self.protocol,
            shot_index=max(parent.shot_index, candidate.shot_index, original.shot_index), temporal_inputs=inputs)
        from utils.agent_vlm import _validate_json
        _validate_json(response['parsed'], PROTECTION_SCHEMA)
        checks = response['parsed']
        return {'allow': all(row['status'] == 'pass' for row in checks.values()), 'checks': checks,
                'independent_model': False, 'scope': 'aligned_sampled_frames_only',
                'parent_fingerprint': parent.input_fingerprint, 'candidate_fingerprint': candidate.input_fingerprint,
                'original_fingerprint': original.input_fingerprint, 'packet': packet,
                'protocol': self.protocol,
                'calls': [{key: response.get(key) for key in ('call_id', 'wall_seconds', 'cached', 'transcript')}]}


class ProgressiveShotSession(CausalShotSession):
    def __init__(self, *args, protection, max_repair_seconds=1800, feedback_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.backend, 'progressive', False) or not callable(getattr(self.backend, 'retain_partial', None)):
            raise ValueError('Progressive session needs a parent-latent replay backend')
        if not callable(getattr(self.verifier, 'snapshot_state', None)):
            raise ValueError('Progressive session requires verifier state snapshots')
        self.protection = protection
        self.feedback_policy = feedback_policy
        self.max_repair_seconds = float(max_repair_seconds)
        if self.max_repair_seconds <= 0:
            raise ValueError('Repair budget must be positive')
        self.observations = {}
        self._offline_window_start = None
        self.ledger.append('repair_mode', mode=('progressive_bounded_feedback_v1' if feedback_policy
                                              else 'progressive_fixed_recipe_v1'),
            max_repair_seconds=self.max_repair_seconds, strength=1.0,
            wall_budget_scope=('run_shot_including_initial_diagnosis_and_localization' if feedback_policy
                               else 'repair_loop_after_initial_diagnosis'),
            budget_semantics='No new attempt after deadline; in-flight operations finish')

    def begin_offline_window(self, shot_index):
        """Start on an audited complete parent without certifying its prefix."""
        if (isinstance(shot_index, bool) or not isinstance(shot_index, int)
                or not 0 < shot_index < self.scene.num_shots):
            raise ValueError('Offline window needs a valid nonzero authored start shot')
        if (self._offline_window_start is not None or self.committed_shots
                or self.observations or self.retry_counts or self.edit_counts):
            raise RuntimeError('Offline window can only be declared on a fresh session')
        bind = getattr(self.backend, 'offline_window_context', None)
        if not callable(bind):
            raise RuntimeError('Backend cannot bind an audited complete offline parent')
        context = bind(shot_index)
        if context.get('start_shot') != shot_index:
            raise ValueError('Offline parent binding has a different start shot')
        self.ledger.append('offline_window_start', start_shot=shot_index,
            unassessed_prefix=list(range(shot_index)), parent_binding=context,
            trusted_memory_added=False, identity_references_promoted=False)
        self._offline_window_start = shot_index

    def _check_shot_sequence(self, k):
        if self._offline_window_start is not None:
            if k < self._offline_window_start:
                raise RuntimeError('Shot is before the declared offline window')
            if k == self._offline_window_start:
                return
        super()._check_shot_sequence(k)

    def _observe_and_judge(self, *args, **kwargs):
        observation, verdict = super()._observe_and_judge(*args, **kwargs)
        self.observations[verdict.input_fingerprint] = observation
        return observation, verdict

    @staticmethod
    def _verdict_summary(verdict):
        return {**CausalShotSession._verdict_summary(verdict), 'input_fingerprint': verdict.input_fingerprint}

    def _origin_observation(self, shot_index):
        # A read-only original-video anchor, never promoted to trusted identity.
        generated = self.backend.original(shot_index)
        observation = self.backend.observe(generated)
        return dataclasses.replace(observation, input_fingerprint=fingerprint({
            'original': generated.model_native_candidate, 'shot': shot_index, 'frames': observation.frames}))

    def _lookahead_guard(self, generated, parent_generated=None, *, basis_state=None):
        passed, report = super()._lookahead_guard(generated, parent_generated, basis_state=basis_state)
        if not passed or not report['enabled']:
            return passed, report
        for item in report['checked']:
            parent = self.observations[item['parent']['input_fingerprint']]
            candidate = self.observations[item['candidate']['input_fingerprint']]
            targets = generated.telemetry.get('progressive_target_requirements', ())
            def target_key(row):
                return (tuple(row.get('subjects', ())), row.get('kind'),
                        row.get('parameters', {}).get('part'), row.get('parameters', {}).get('aspect'),
                        fingerprint(row.get('expected')))
            target_keys = {target_key(row) for row in targets}
            try:
                guard = self.protection.review(parent, candidate, self._origin_observation(item['shot_index']),
                    self.scene.view(item['shot_index']),
                    target_requirements=targets,
                    entity_specs=item['parent']['metrics'].get('repair_specs', {}),
                    protected_requirements=[row['constraint'] for row in item['parent']['metrics'].get('constraint_assessments', [])
                        if target_key(row['constraint']) not in target_keys],
                    references=tuple(item['parent']['metrics'].get('repair_references', {}).values()))
            except Exception as error:
                guard = {'allow': False, 'error': f'{type(error).__name__}: {error}'}
            item['paired_visual_protection'] = guard
            if not guard['allow']:
                item['decision'] = 'rollback_candidate'
                report['decision'] = 'rollback_paired_visual_protection'
                return False, report
        return True, report

    @staticmethod
    def _fixed_action(verdict):
        rows = assessments(verdict)
        # Start with attribute families that the development executor can edit.
        targets = [row for row in rows.values() if row.status == 'violated' and
                   row.constraint.parameters.get('part') in ('hair', 'garment')]
        targets.sort(key=lambda row: (row.constraint.parameters.get('aspect') != 'color',
                                     row.constraint.parameters['part'] != 'hair', row.id))
        for target in targets:
            name = target.constraint.subjects[0]
            part = target.constraint.parameters['part']
            options = verdict.metrics.get('repair_options', [])
            option = next((o for o in options if o.get('id') == f'attribute:{name}:{part}'), None)
            if option is None:
                continue
            raw = option['action']
            action = Action(raw['tool'], raw['params'], reason='progressive_fixed_recipe').normalized(
                shot_index=target.constraint.shot_indices[0])
            if action.params['entry_sigma'] != 1.0 or file_sha256(action.params['mask_path']) != option['mask_sha256']:
                raise ValueError('Fixed action strength or offered mask changed')
            # One part can contain several atomic requirements; every remaining
            # old failure stays in the denominator, regardless of target choice.
            ids = [row.id for row in targets if row.constraint.subjects == target.constraint.subjects
                   and row.constraint.parameters['part'] == part]
            return action, ids
        return None, []

    @staticmethod
    def _insufficient_protection(report):
        return bool(report and (report.get('error') or any(
            row.get('status') == 'unknown' for row in report.get('checks', {}).values())))

    def run_shot(self, shot_index):
        overall_started = time.monotonic()
        k = int(shot_index)
        context = self.pre_shot(k)
        program = self.controller.decide(context)
        if program.input_fingerprint != context.input_fingerprint:
            raise RuntimeError('Controller fingerprint mismatch')
        self.apply(k, program, retry=False)
        boundary = self.backend.snapshot(k)
        best = self.backend.generate(k, 0)
        best_state = self.backend.snapshot(k)
        best_review = self._stage_and_judge(context, best, expected_attempt=0)
        initial_review = best_review
        if best_review.accept and all(a.status == 'satisfied' for a in assessments(best_review).values()):
            return self._commit_accepted(best, best_review, boundary_snapshot=boundary,
                                         boundary_snapshot_digest=fingerprint(boundary))
        self._reject_candidate(best, best_review)
        original_observation = self._origin_observation(k)
        if original_observation.telemetry['shot_video'] != self.observations[best_review.input_fingerprint].telemetry['shot_video']:
            state = self.verifier.snapshot_state()
            try:
                _, initial_review = self._observe_and_judge(self.backend.original(k), context.normative,
                    ledger_event='fixed_original_verifier')
            finally:
                self.verifier.restore_state(state)
        attempts, retained = 0, 0
        started = overall_started if self.feedback_policy else time.monotonic()
        repeats = {}
        feedback_history, latest_feedback, latest_observation = [], {}, None
        reason = 'render_budget_exhausted'
        for attempt in range(1, self.config.max_edits_per_shot + 1):
            if time.monotonic() - started >= self.max_repair_seconds:
                reason = 'wall_budget_exhausted'
                break
            try:
                action, targets = self._fixed_action(best_review)
            except Exception as error:
                reason = 'execution_failed'
                self.ledger.append('progressive_execution_failed', shot_index=k, attempt=attempt,
                    phase='action_validation', error=f'{type(error).__name__}: {error}')
                break
            if action is None:
                reason = 'no_supported_confirmed_target'
                break
            if self.feedback_policy:
                try:
                    from utils.agent_bounded_feedback import actual_action
                    parent_observation = self.observations[best_review.input_fingerprint]
                    binding = {'parent_video': parent_observation.telemetry['shot_video'],
                               'parent_latents': best_state['branch']['raw_latents'],
                               'seed': self.backend.noise_seed + attempt}
                    menu, menu_reason = self.feedback_policy.menu(action, best_review.metrics.get('repair_options', []),
                        [assessments(best_review)[key].constraint.to_dict() for key in targets],
                        history=feedback_history, failure=latest_feedback.get('failure'), **binding)
                    if not menu:
                        reason = menu_reason
                        self.ledger.append('bounded_feedback_stop', shot_index=k, attempt=attempt, reason=reason)
                        break
                    images, labels, temporal = [], [], list(context.inputs)
                    horizon = k
                    for label, observation in [('current_best_parent', parent_observation),
                                               ('latest_rejected_candidate', latest_observation)]:
                        if observation is None:
                            continue
                        for path in observation.frames:
                            images.append(path)
                            labels.append({'image': len(images), 'role': label, 'shot': observation.shot_index})
                        temporal.extend(observation.inputs)
                        horizon = max(horizon, observation.shot_index)
                    future = latest_feedback.get('future', {})
                    horizon = max(horizon, int(future.get('decision_as_of_shot', k)))
                    # Preserve one actual failing later pair, with its real
                    # observation horizon; never masquerade as online evidence.
                    for checked in future.get('checked', []):
                        if checked.get('decision') not in ('rollback_candidate',) and checked.get('comparison', {}).get('allow_commit', True):
                            continue
                        for side in ('parent', 'candidate'):
                            digest = checked.get(side, {}).get('input_fingerprint')
                            observation = self.observations.get(digest)
                            if observation is None:
                                raise ValueError('Future feedback has no audited visual observation')
                            for path in observation.frames[:2]:
                                images.append(path)
                                labels.append({'image': len(images), 'role': 'future_' + side, 'shot': observation.shot_index})
                            temporal.extend(observation.inputs)
                            horizon = max(horizon, observation.shot_index)
                        break
                    if menu[0]['kind'] != 'first_fixed_recipe':
                        from utils.agent_repair_regions import region_preview
                        preview_region = getattr(self.backend, 'region_preview', region_preview)
                        for offered in menu:
                            path = offered['action'].params['mask_path']
                            original_option = next((row for row in best_review.metrics.get('repair_options', [])
                                if row['action']['params'].get('mask_path') == path), {})
                            preview = preview_region({**original_option, 'action': offered['action'].to_dict(),
                                'mask_sha256': file_sha256(path)},
                                video=binding['parent_video'], output_dir=Path(path).parent / 'bounded_feedback_previews')
                            images.append(preview['path'])
                            labels.append({'image': len(images), 'role': 'legal_region', 'option_id': offered['id'], 'shot': k})
                            temporal.append(TemporalInput('bounded_region_preview', preview['path'], preview['sha256'], k))
                    packet = {**latest_feedback, 'fixed_requirements': context.normative.to_dict(),
                              'parent_constraints': best_review.metrics['constraint_assessments'],
                              'image_roles': labels, 'observed_through_shot': horizon}
                    temporal.append(TemporalInput.from_value(kind='bounded_feedback_packet',
                        identifier=f'{self.scene.scene}:{k}:{attempt}', value=packet, available_after_shot=horizon))
                    audited_inputs = self._inputs_or_fail(temporal, shot_index=horizon, stage='bounded_feedback_planner')
                    choice, planning = self.feedback_policy.choose(menu, feedback=packet, images=images,
                                                                  inputs=audited_inputs, horizon=horizon)
                    self.ledger.append('bounded_feedback_plan', shot_index=k, attempt=attempt,
                        menu=[{key: value for key, value in row.items() if key != 'action'} for row in menu],
                        feedback=packet, planning=planning)
                    if choice is None:
                        reason = planning.get('stop', 'planner_stop')
                        break
                    action = choice['action']
                    if actual_action(action, **binding) != choice['actual']:
                        raise ValueError('Action inputs changed during planning')
                    if time.monotonic() - started >= self.max_repair_seconds:
                        reason = 'wall_budget_exhausted'
                        break
                    feedback_history.append({'actual': choice['actual'], 'targets': choice['targets']})
                    self.ledger.append('bounded_action_reserved', shot_index=k, attempt=attempt,
                        actual=choice['actual'], kind=choice['kind'], targets=choice['targets'])
                except Exception as error:
                    reason = 'execution_failed'
                    self.ledger.append('progressive_execution_failed', shot_index=k, attempt=attempt,
                        phase='bounded_feedback', error=f'{type(error).__name__}: {error}')
                    break
            from utils.agent_bounded_feedback import mask_geometry
            try:
                mask_digest = mask_geometry(action.params['mask_path'],
                    parent_video=self.observations[best_review.input_fingerprint].telemetry['shot_video'],
                    parent_latents=best_state['branch']['raw_latents'])['sha256']
            except Exception as error:
                reason = 'execution_failed'
                self.ledger.append('progressive_execution_failed', shot_index=k, attempt=attempt,
                    phase='mask_geometry', error=f'{type(error).__name__}: {error}')
                break
            actual = {'parent': best.model_native_candidate['branch_id'],
                      'mask_content_sha256': mask_digest,
                      'prompt': action.params['prompt'], 'sigma': 1.0}
            key = fingerprint(actual)
            if repeats.get(key, 0) >= 2:
                reason = 'fixed_recipe_retry_budget_exhausted'
                break
            repeats[key] = repeats.get(key, 0) + 1
            self.ledger.append('progressive_action', shot_index=k, attempt=attempt, targets=targets,
                actual_action=actual, noise_seed=self.backend.noise_seed + attempt,
                actual_kind='resample' if repeats[key] > 1 else 'first_fixed_recipe')
            # Backend and verifier still point to the current best branch.
            attempts = attempt
            verifier_state = self.verifier.snapshot_state()
            try:
                child, review = self._edit_and_review(context, best_review, action, best_state, fingerprint(best_state), attempt)
                child = dataclasses.replace(child, telemetry={**child.telemetry,
                    'progressive_target_requirements': [assessments(best_review)[key].constraint.to_dict()
                                                        for key in targets]})
                decision = improvement_decision(best_review, review, initial_review, targets)
            except Exception as error:
                self.verifier.restore_state(verifier_state)
                self._verifier_candidate_states.pop((k, attempt), None)
                if self.memory.candidate(k, attempt) is not None:
                    self.memory.quarantine(k, attempt)
                reason = 'render_timeout' if isinstance(error, TimeoutError) else 'execution_failed'
                self.ledger.append('progressive_execution_failed', shot_index=k, attempt=attempt,
                    phase='render_or_review', error=f'{type(error).__name__}: {error}')
                break
            protection = None
            if decision['allow']:
                try:
                    protection = self.protection.review(self.observations[best_review.input_fingerprint],
                        self.observations[review.input_fingerprint], original_observation, context.normative,
                        target_requirements=[assessments(best_review)[key].constraint.to_dict() for key in targets],
                        references=tuple(best_review.metrics.get('repair_references', {}).values()),
                        protected_requirements=[row.constraint.to_dict() for key, row in assessments(best_review).items() if key not in targets],
                        entity_specs=best_review.metrics.get('repair_specs', {}))
                except Exception as error:
                    protection = {'allow': False, 'error': f'{type(error).__name__}: {error}'}
                decision['allow'] = protection['allow']
            self.ledger.append('progressive_current_guard', shot_index=k, attempt=attempt,
                decision=decision, protection=protection, parent_branch_id=actual['parent'],
                candidate_branch_id=child.model_native_candidate['branch_id'],
                elapsed_repair_wall_seconds=time.monotonic()-started)
            if not decision['allow']:
                latest_observation = self.observations[review.input_fingerprint]
                latest_feedback = {'failure': ('protection_regression' if protection or
                    'constraint_regression_or_definition_change' in decision['reasons'] else 'target_not_improved'),
                    'decision': decision, 'protection': protection,
                    'candidate_constraints': review.metrics.get('constraint_assessments', [])}
                self._reject_candidate(child, review)
                if self._insufficient_protection(protection) or any(
                    item.startswith(('critical_unknown:', 'existing_failure_became_unobservable:'))
                    for item in decision['reasons']
                ):
                    reason = 'insufficient_evidence'
                    break
                continue
            if decision['fully_verified_current_shot']:
                committed = self._commit_accepted(child, review, boundary_snapshot=boundary,
                    boundary_snapshot_digest=fingerprint(boundary), fallback=(best, best_state, fingerprint(best_state)),
                    defer_rollback_delivery=True)
                if isinstance(committed, ShotResult):
                    self.ledger.append('repair_finished', shot_index=k, status='fully_verified_current_shot',
                        render_attempts=attempts, retained_improvements=retained, wall_seconds=time.monotonic()-started)
                    return committed
                latest_observation = self.observations[review.input_fingerprint]
                latest_feedback = {'failure': 'future_regression', 'future': committed.metrics.get('lookahead', {}),
                                   'candidate_constraints': review.metrics.get('constraint_assessments', [])}
                future = latest_feedback['future']
                if self.feedback_policy and (future.get('error') or any(
                        self._insufficient_protection(item.get('paired_visual_protection'))
                        for item in future.get('checked', []))):
                    reason = 'insufficient_future_evidence'
                    break
                continue
            basis = self._verifier_candidate_states[(k, child.attempt)]
            future_ok, future = self._lookahead_guard(child, best, basis_state=basis)
            self.ledger.append('progressive_future_guard', shot_index=k, attempt=attempt, passed=future_ok, lookahead=future)
            if not future_ok:
                latest_observation = self.observations[review.input_fingerprint]
                latest_feedback = {'failure': 'future_regression', 'future': future,
                                   'candidate_constraints': review.metrics.get('constraint_assessments', [])}
                self._reject_candidate(child, review)
                if future.get('error') or any(self._insufficient_protection(item.get('paired_visual_protection'))
                                            for item in future.get('checked', [])):
                    reason = 'insufficient_future_evidence'
                    break
                continue
            # Quarantine from identity/retrieval trust, but keep the guarded
            # render as the next parent. These are deliberately separate states.
            self.verifier.restore_state(self._verifier_candidate_states.pop((k, child.attempt)))
            payload = self.memory.quarantine(k, child.attempt)
            best_state = self.backend.retain_partial(child, unresolved_constraints=decision['unresolved'])
            best, best_review = child, review
            latest_feedback, latest_observation = {}, None
            retained += 1
            self.ledger.append('partial_improvement_retained', shot_index=k, attempt=attempt,
                parent=actual['parent'], best_branch=child.model_native_candidate['branch_id'],
                improved=decision['improved'], unresolved=decision['unresolved'],
                snapshot_fingerprint=fingerprint(best_state), payload_fingerprint=payload.payload_fingerprint,
                trusted_identity_reference=False)
        self._restore_checked(k, best_state, fingerprint(best_state), purpose='progressive_best_delivery')
        status = 'partial_improvement' if retained else 'unresolved'
        termination = {'wall_budget_exhausted': 'timeout', 'render_timeout': 'timeout',
                       'execution_failed': 'execution_failed'}.get(reason, reason)
        unresolved = [key for key, row in assessments(best_review).items() if row.status != 'satisfied']
        self.backend.record_unresolved(k, status=status, unresolved=unresolved, termination=termination)
        delivered = self.backend.commit(best, quarantined=True)
        self.committed_shots.add(k)
        self.ledger.append('commit', shot_index=k, attempt=best.attempt, quarantined=True,
            selection='current_best_branch', status=status, reason=reason,
            payload_fingerprint=self.memory.quarantined_payload(k, best.attempt).payload_fingerprint)
        self.ledger.append('repair_finished', shot_index=k, status=status, reason=reason,
            termination=termination,
            render_attempts=attempts, retained_improvements=retained, wall_seconds=time.monotonic()-started,
            unresolved=unresolved)
        self.scene.assert_unchanged()
        if self.memory.audit_invisible():
            raise RuntimeError('Untrusted partial branch leaked into trusted memory')
        return ShotResult(k, attempts + 1, False, True, delivered, best_review)
