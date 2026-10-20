"""Official offline replay backend for the shared progressive transaction.

Branches bind whole native runs. Restoring selects an immutable saved parent;
the next render reconstructs KV/AE/retrieval/text/RNG from its clean latents.
There is no serialized live GPU state, streaming claim, or identity promotion.
"""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import time

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'implementation' if (ROOT / 'implementation').is_dir() else ROOT))
from utils.agent_bounded_feedback import actual_action, mask_geometry
from utils.agent_runtime import GeneratedShot, ShotObservation, TemporalInput, file_sha256


def only(folder, pattern):
    path, = Path(folder).glob(pattern)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


class NativeReplayBackend:
    progressive = True

    def __init__(self, *, parent, output, checkout, assets_manifest, python, gpu,
                 noise_seed=4242, timeout=1800):
        if gpu not in (0, 1, 2):
            raise ValueError('Only authorized GPUs 0, 1 and 2 are supported')
        if timeout <= 0:
            raise ValueError('Positive render timeout required')
        self.output, self.checkout = Path(output).resolve(), Path(checkout).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.assets_manifest = str(Path(assets_manifest).resolve())
        # Resolving a venv executable symlink changes the environment.
        self.python = str(Path(python).absolute())
        self.gpu, self.noise_seed, self.timeout = gpu, int(noise_seed), float(timeout)
        self.cursor, self.sequence = 0, 0
        self.pending, self.branches, self.delivered = {}, {}, []
        self.frame_artifacts = {}
        root = self._register(parent, None, None)
        self.original_branch_id = self.active_branch_id = self.safe_branch_id = root['branch_id']
        self.fully_verified_branch_id = None
        self.timing = read_json(Path(root['run']) / 'diagnostics/timing_0000.json')
        if self.timing['protocol'] != 'continuous_vae_drop_first_frame_v1':
            raise ValueError('Native continuous timing is required')
        self._write_manifest()

    def _register(self, folder, parent_id, action):
        folder = Path(folder).resolve()
        if read_json(folder / 'status.json')['status'] != 'completed':
            raise ValueError('Only completed native runs can become branches')
        video = only(folder / 'videos', '*.mp4')
        audit = read_json(folder / 'audit.json')
        if (file_sha256(video) != audit['video_sha256'] or
                file_sha256(folder / 'diagnostics/latents_0000.pt') != audit['latent_sha256']):
            raise ValueError('Native branch differs from its audited artifacts')
        if parent_id is not None and not audit.get('all_engineering_checks_passed'):
            raise ValueError('Candidate engineering audit did not pass')
        files = [folder / name for name in ('status.json', 'manifest.json', 'config.yaml', 'audit.json',
            'diagnostics/timing_0000.json', 'diagnostics/latents_0000.pt', 'diagnostics/switches_0000.json')]
        files += [video, only(folder / 'videos', '*memory_log.json')]
        if parent_id is not None:
            files.append(folder / 'diagnostics/repair_0000.json')
        bound = {str(path): file_sha256(path) for path in files}
        manifest = read_json(folder / 'manifest.json')
        if manifest['config_sha256'] != bound[str(folder / 'config.yaml')]:
            raise ValueError('Run config changed since preparation')
        if parent_id is not None:
            parent = self.branches[parent_id]
            if manifest['parent_manifest_sha256'] != parent['artifacts'][str(Path(parent['run']) / 'manifest.json')]:
                raise ValueError('Candidate has a different parent manifest')
            if bound[str(folder / 'diagnostics/timing_0000.json')] != parent['artifacts'][str(Path(parent['run']) / 'diagnostics/timing_0000.json')]:
                raise ValueError('Candidate changed the authored timing')
        branch_id = f'native-{self.sequence:04d}'
        self.sequence += 1
        branch = {'branch_id': branch_id, 'parent': parent_id, 'run': str(folder), 'video': str(video),
            'raw_latents': str(folder / 'diagnostics/latents_0000.pt'), 'artifacts': bound,
            'action': deepcopy(action), 'tentative': parent_id is not None, 'quarantined': False,
            'repair_status': 'unreviewed', 'unresolved_constraints': [], 'shot_outcomes': [],
            'trusted_identity_reference': False, 'reference_frames': [],
            'state_restore': 'reconstruct_full_replay_from_parent_clean_latents'}
        if parent_id is not None:
            branch['shot_outcomes'] = deepcopy(self.branches[parent_id]['shot_outcomes'])
            branch['unresolved_constraints'] = list(self.branches[parent_id]['unresolved_constraints'])
        self.branches[branch_id] = branch
        return branch

    @property
    def active_branch(self):
        return self.branches[self.active_branch_id]

    def _validate(self, branch):
        for path, digest in branch['artifacts'].items():
            if file_sha256(path) != digest:
                raise ValueError('Saved native branch artifact changed: ' + path)

    def _write_manifest(self):
        state = {'original_branch_id': self.original_branch_id, 'current_best_branch_id': self.active_branch_id,
            'fully_verified_branch_id': self.fully_verified_branch_id,
            'fully_verified_scope': 'all authored shots, under the configured development verifier and finite guards',
            'branches': list(self.branches.values()), 'delivered': self.delivered,
            'identity_references_promoted': False, 'streaming': False}
        temporary = self.output / 'branches.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.output / 'branches.json')

    def snapshot(self, shot_index):
        self._validate(self.active_branch)
        return deepcopy({'shot_index': int(shot_index), 'cursor': self.cursor,
            'active_branch_id': self.active_branch_id, 'safe_branch_id': self.safe_branch_id,
            'fully_verified_branch_id': self.fully_verified_branch_id, 'branch': self.active_branch})

    def restore(self, snapshot):
        branch_id = snapshot['active_branch_id']
        if self.branches.get(branch_id) != snapshot['branch']:
            raise ValueError('Native snapshot does not match its recorded branch')
        self._validate(self.branches[branch_id])
        for key, pending_id in list(self.pending.items()):
            self.branches[pending_id].update(tentative=False, quarantined=True)
            del self.pending[key]
        self.active_branch_id, self.safe_branch_id = branch_id, snapshot['safe_branch_id']
        self.fully_verified_branch_id = snapshot['fully_verified_branch_id']
        self.cursor = snapshot['cursor']
        self._write_manifest()

    def _candidate(self, branch_id, shot, attempt, *, edited=False):
        branch = self.branches[branch_id]
        self._validate(branch)
        segment = self.timing['segments'][shot]
        first, count = segment['start_export_frame'], segment['export_frames']
        indices = [first + min(count-1, int(count*f)) for f in (.2, .5, .8)]
        destination = self.output / 'frames' / branch_id / f'shot_{shot:02d}'
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        capture = cv2.VideoCapture(branch['video'])
        try:
            for index in indices:
                path = destination / f'{index:06d}.png'
                if str(path) in self.frame_artifacts and (
                        not path.exists() or file_sha256(path) != self.frame_artifacts[str(path)]):
                    raise ValueError('Previously extracted native evidence changed')
                if not path.exists():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, pixels = capture.read()
                    if not ok or not cv2.imwrite(str(path), pixels):
                        raise ValueError('Cannot extract native evidence frame ' + str(index))
                self.frame_artifacts[str(path)] = file_sha256(path)
                paths.append(str(path))
        finally:
            capture.release()
        refs = tuple(TemporalInput('current_shot_frame', p, file_sha256(p), shot) for p in paths)
        return GeneratedShot(shot, attempt, {'video': branch['video'], 'frames': paths},
            {'frames': paths}, {'branch_id': branch_id, 'parent': branch['parent'],
                'source_video': branch['video'], 'source_video_sha256': branch['artifacts'][branch['video']]},
            {'edited': edited, 'frame_indices': indices}, refs)

    def original(self, shot_index):
        return self._candidate(self.original_branch_id, shot_index, 0)

    def generate(self, shot_index, attempt):
        return self._candidate(self.active_branch_id, shot_index, attempt)

    def preview(self, generated, shot_index):
        return self._candidate(generated.model_native_candidate['branch_id'], shot_index, generated.attempt,
                               edited=generated.telemetry.get('edited', False))

    def observe(self, generated):
        branch = self.branches[generated.model_native_candidate['branch_id']]
        self._validate(branch)
        for reference in generated.input_refs:
            if file_sha256(reference.identifier) != reference.sha256:
                raise ValueError('Native evidence changed before observation')
        return ShotObservation(generated.shot_index, generated.attempt, tuple(generated.output['frames']), (),
            {'shot_video': branch['video'], 'native_run': branch['run'],
             'frame_indices': list(generated.telemetry['frame_indices']),
             'reference_frames': [], 'edited': generated.telemetry.get('edited', False)}, generated.input_refs, '')

    def telemetry(self, shot_index):
        return {'active_branch_id': self.active_branch_id, 'committed_shots': self.cursor,
                'state_restore': 'full_parent_latent_replay', 'trusted_identity_reference': False}

    def apply(self, shot_index, program, *, retry):
        if any(action.tool != 'abstain' for action in program.actions):
            raise ValueError('Native replay supports no pre-generation mutations')
        return [{'metrics': {'noop': True}, 'attention_mass_check': 'not_applicable'} for _ in program.actions]

    def prepare_action(self, choice, parent, output):
        """Invoke the real native preparer with the configured generator Python."""
        output = Path(output)
        if any(path.exists() for path in (output, output.with_suffix('.request.json'),
                output.with_suffix('.prepared.json'), output.with_suffix('.preparation.log'))):
            raise FileExistsError('Native attempt already exists; preserve its original records')
        request = {'choice': {**choice, 'action': choice['action'].to_dict()},
            'parent': parent['run'], 'parent_branch': parent['branch_id'], 'checkout': str(self.checkout),
            'output': str(output), 'assets_manifest': self.assets_manifest, 'gpu': self.gpu, 'timeout': self.timeout}
        request_path = output.with_suffix('.request.json')
        with request_path.open('x') as stream:
            stream.write(json.dumps(request, indent=2) + '\n')
        result = output.with_suffix('.prepared.json')
        adapter = Path(__file__).parent
        with output.with_suffix('.preparation.log').open('x') as log:
            subprocess.run([self.python, str(adapter / 'feedback_bridge.py'), '--request', str(request_path),
                            '--result', str(result)], check=True, stdout=log, stderr=subprocess.STDOUT)
        return read_json(result)

    def _execute(self, choice, parent, output):
        prepared = self.prepare_action(choice, parent, output)
        adapter = Path(__file__).parent
        with output.with_suffix('.execution.log').open('x') as log:
            subprocess.run(prepared['command'], check=True, stdout=log, stderr=subprocess.STDOUT)
            status = read_json(output / 'status.json')['status']
            if status == 'timeout':
                raise TimeoutError('Native render exceeded its frozen timeout')
            if status != 'completed':
                raise RuntimeError('Native render ended: ' + status)
            subprocess.run([self.python, str(adapter / 'audit_candidate.py'), '--folder', str(output),
                '--parent', parent['run'], '--output', str(output / 'audit.json')],
                check=True, stdout=log, stderr=subprocess.STDOUT)
        return output

    def edit(self, shot_index, attempt, action, verdict):
        started = time.monotonic()
        self._validate(self.active_branch)
        parent = deepcopy(self.active_branch)
        actual = actual_action(action, parent_video=parent['video'], parent_latents=parent['raw_latents'],
                               seed=self.noise_seed + int(attempt))
        segment = self.timing['segments'][shot_index]
        interval = [segment['start_latent_frame'], segment['start_latent_frame'] + segment['latent_frames']]
        if actual['shot'] != shot_index or actual['coordinates'].get('latent_range') != interval:
            raise ValueError('Native action must cover exactly the requested authored segment')
        key = (int(shot_index), int(attempt))
        output = self.output / f'candidate_{self.sequence:04d}_shot{shot_index:02d}_attempt{attempt}'
        choice = {'action': action, 'actual': actual}
        folder = self._execute(choice, parent, output)
        self._validate(parent)
        branch = self._register(folder, parent['branch_id'], actual)
        self.pending[key] = branch['branch_id']
        self._write_manifest()
        generated = self._candidate(branch['branch_id'], shot_index, attempt, edited=True)
        return generated, {'actual_action': actual, 'native_run': str(folder),
            'audit_sha256': file_sha256(Path(folder) / 'audit.json'),
            'wall_seconds': time.monotonic() - started, 'state_reconstruction': 'full_replay',
            'attention_mass_check': 'not_applicable_latent_rewrite'}

    def _promote(self, generated):
        key = (generated.shot_index, generated.attempt)
        branch_id = generated.model_native_candidate['branch_id']
        if self.pending.get(key) != branch_id or self.branches[branch_id]['parent'] != self.active_branch_id:
            raise ValueError('Only a pending child of current best may be retained')
        self._validate(self.branches[branch_id])
        del self.pending[key]
        self.branches[branch_id].update(tentative=False, quarantined=False)
        self.active_branch_id = self.safe_branch_id = branch_id
        # The new full video has a regenerated suffix, including any previously
        # verified future shots. Invalidate those outcomes rather than inherit trust.
        self.active_branch['shot_outcomes'] = [row for row in self.active_branch['shot_outcomes']
                                               if row['shot'] < generated.shot_index]
        self.fully_verified_branch_id = None

    def _outcome(self, shot, status, unresolved, termination=None):
        rows = {row['shot']: row for row in self.active_branch['shot_outcomes']}
        rows[shot] = {'shot': shot, 'status': status, 'unresolved': sorted(set(unresolved)), 'termination': termination}
        self.active_branch.update(shot_outcomes=[rows[k] for k in sorted(rows)], repair_status=status,
            unresolved_constraints=sorted({item for row in rows.values() for item in row['unresolved']}))
        self._write_manifest()

    def retain_partial(self, generated, *, unresolved_constraints):
        if not unresolved_constraints:
            raise ValueError('A partial improvement must retain unresolved predicates')
        self._promote(generated)
        self._outcome(generated.shot_index, 'partial_improvement', unresolved_constraints)
        return self.snapshot(generated.shot_index)

    def record_unresolved(self, shot_index, *, status, unresolved, termination):
        if status not in ('unresolved', 'partial_improvement'):
            raise ValueError('Invalid unresolved outcome')
        self._outcome(shot_index, status, unresolved, termination)

    def commit(self, generated, *, quarantined):
        branch_id = generated.model_native_candidate['branch_id']
        if branch_id != self.active_branch_id:
            if quarantined:
                raise ValueError('Cannot deliver a rejected tentative child')
            self._promote(generated)
        self._validate(self.active_branch)
        if not quarantined:
            self._outcome(generated.shot_index, 'fully_verified_current_shot', [])
            rows = self.active_branch['shot_outcomes']
            if {row['shot'] for row in rows} == set(range(len(self.timing['segments']))) and all(
                    row['status'] == 'fully_verified_current_shot' for row in rows):
                self.fully_verified_branch_id = self.active_branch_id
        self.cursor = max(self.cursor, generated.shot_index + 1)
        row = {'shot': generated.shot_index, 'attempt': generated.attempt, 'quarantined': quarantined,
               'branch_id': self.active_branch_id, 'video': self.active_branch['video'],
               'repair_status': self.active_branch['repair_status'],
               'unresolved_constraints': list(self.active_branch['unresolved_constraints'])}
        self.delivered.append(row)
        self._write_manifest()
        return row

    def region_preview(self, option, *, video, output_dir):
        from native_region_preview import preview
        if str(Path(video).resolve()) != self.active_branch['video']:
            raise ValueError('Native region preview must use current best parent')
        return preview(option, video=video, parent_latents=self.active_branch['raw_latents'],
                       output_dir=output_dir, timing=self.timing)
