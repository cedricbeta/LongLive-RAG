#!/usr/bin/env python3
"""Frozen 2x2 garment localization diagnostic; no generator or oracle masks."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path[:0] = [str(BASE), str(ROOT / 'adapters/official_longlive_rag')]
from entity_verifier import _NativeJudge, NativePartSegmenter, PinnedSam2Tracker
from utils.agent_entity_judge import EntitySpec
from utils.agent_runtime import file_sha256


class FixedParts:
    """Postprocess different measured part outputs with one unchanged recipe."""
    context = None

    def __init__(self, part, face):
        self.part, self.face = part, face

    def __call__(self, video, chunks, phrase):
        result = self.face if phrase == 'face' else self.part
        return {i: result[i] for i in chunks}


def propagate_masks(predictor, folder, seeds, count, hw, resources):
    started = time.monotonic()
    output = {name: np.zeros((count, *hw), dtype=bool) for name in seeds}
    seed_arrays = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(folder))
        names = list(seeds)
        for i, name in enumerate(names):
            mask = torch.nn.functional.interpolate(seeds[name][None, None].float(), hw, mode='nearest')[0, 0] >= .5
            if not mask.any():
                raise ValueError('Empty automatic first-frame seed; do not substitute a later frame')
            seed_arrays[name] = mask.numpy()
            predictor.add_new_mask(state, frame_idx=0, obj_id=i + 1, mask=mask)
        try:
            for frame, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                if frame % 16 == 0:
                    resources()
                masks = (logits > 0).cpu().numpy()
                for j, oid in enumerate(object_ids):
                    output[names[int(oid) - 1]][frame] = masks[j, 0]
        finally:
            predictor.reset_state(state)
    return output, seed_arrays, time.monotonic() - started


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

    status = {'status': 'running', 'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_unix': time.time()}
    write('status.json', status)
    rows = []
    costs = []
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise ValueError('This frozen diagnostic uses admitted GPU 1 only')
        lock = open('/tmp/longlive_feedback_render_gpu1.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()

        def resources():
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            for line in raw.splitlines():
                device, pid = (s.strip() for s in line.split(','))
                try:
                    group = os.getpgid(int(pid))
                except ProcessLookupError:
                    continue
                if device == uuid and group != os.getpgrp():
                    raise RuntimeError('resource_interrupted: foreign GPU 1 process')

        resources()
        plan = json.loads(args.plan.read_text())
        for name, digest in plan['source_sha256'].items():
            path = ROOT / name
            assert file_sha256(path) == digest, name
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        parent = args.parent.resolve()
        original = args.original.resolve()
        old = json.loads((original / 'manifest.json').read_text())
        video, = (parent / 'videos').glob('*.mp4')
        assert file_sha256(video) == old['parent_video_sha256'] == plan['parent_video_sha256']
        assert file_sha256(args.sam2_checkpoint) == old['sam2_checkpoint_sha256']
        for name, digest in old['clipseg_files_sha256'].items():
            assert file_sha256(args.clipseg_model / name) == digest
        assert file_sha256(original / 'assignments.json') == plan['automatic_assignments_sha256']
        compiled = json.loads(args.compiled.read_text())
        assert file_sha256(args.compiled) == plan['compiled_sha256']
        names = [e['name'] for e in old['entity_specs']]
        heads = {}
        for name in names:
            entity, = [e for e in compiled['entities'] if e['name'] == name]
            values = {r['expected'] for r in entity['appearance'] if r['part'] == 'garment' and r['aspect'] == 'type' and r['source']['shot'] <= old['segment'] <= r['last_active_shot']}
            if len(values) != 1:
                raise ValueError('Exactly one literal garment head is required for this diagnostic')
            heads[name] = values.pop()
        assert heads == plan['literal_heads']
        manifest = {'plan_sha256': file_sha256(args.plan), 'source_sha256': plan['source_sha256'],
                    'original_manifest_sha256': file_sha256(original / 'manifest.json'),
                    'sam2_checkpoint_sha256': old['sam2_checkpoint_sha256'],
                    'clipseg_files_sha256': old['clipseg_files_sha256'],
                    'environment': {'python': sys.version, **{x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'sam2', 'numpy']}},
                    'new_vlm_calls': 0, 'new_video_generations': 0, 'quality_claim': False, 'held_out': False}
        write('manifest.json', manifest)
        grid = tuple(json.loads((parent / 'audit.json').read_text())['latent_shape'][-2:])
        timing = json.loads((parent / 'diagnostics/timing_0000.json').read_text())
        segment = timing['segments'][old['segment']]
        first, count = segment['start_export_frame'], segment['export_frames']
        assert first == 0 and count == 160
        tracker = PinnedSam2Tracker(args.sam2_checkpoint, 'cuda:0')
        part = NativePartSegmenter(args.clipseg_model, 'cuda:0', grid)
        assert type(part.processor.image_processor).__name__ == 'ViTImageProcessor'
        write('processor.json', part.processor.image_processor.to_dict())
        specs = [EntitySpec(e['name'], e['description']) for e in old['entity_specs']]
        judge = _NativeJudge(specs, localizer=None, comparer=None, tracker=tracker, work_dir=out / 'entities',
                            grid=grid, frames_per_chunk=1, part_mask_fn=part)
        judge.progressive_feedback_options = True
        folder, n, hw = judge._shot_frames(video, first, first + count - 1)
        requests = {name: {} for name in names}
        for row in json.loads((original / 'assignments.json').read_text()):
            for name, decision in row['roles'].items():
                if decision['status'] == 'assigned':
                    requests[name][row['frame']] = row['boxes'][decision['subject_index']]
        resources()
        before = time.monotonic()
        tracks = tracker.tracks(folder, requests, n, hw)
        costs.append({'stage': 'sam2_person_tracking', 'wall_seconds': time.monotonic() - before, 'frames': n, 'objects': len(names)})
        soft = {name: judge._soft(track, n) for name, track in tracks.items()}
        chunks = list(range(count))
        faces = part(video, chunks, 'face')
        arrays = {'frame_indices': np.array(chunks)}
        arrays.update({name + '_person': np.stack([v[0].numpy() for v in soft[name].values()]) for name in names})
        arrays['face'] = np.stack([faces[i][0].numpy() for i in chunks])

        def postprocess(name, values):
            judge._part_mask_fn = FixedParts(values, faces)
            return judge._attribute_mask(name, ['garment'], video, 0, n, soft[name],
                                         [s for other, s in soft.items() if other != name])

        def record(name, prompt_mode, time_mode, values, info):
            key = '_'.join([name, prompt_mode, time_mode])
            # Match the latent adapter's final exclusion, rather than equate
            # the old pre-protection tensor with the actually allowed region.
            final = np.stack([((info['masks'][i] >= .35) & (info['protected'][i] < .35))[0].numpy() for i in chunks])
            arrays[key + '_raw'] = np.stack([values[i][0].numpy() for i in chunks])
            arrays[key + '_final'] = final
            areas = final.sum(axis=(1, 2))
            row = {'entity': name, 'prompt_mode': prompt_mode, 'time_mode': time_mode,
                   'phrase': 'clothing' if prompt_mode == 'generic' else heads[name],
                   'empty_frames': int((areas == 0).sum()), 'areas': areas.tolist(),
                   'raw_areas': [int((values[i] >= .35).sum()) for i in chunks],
                   'dilation_steps': info['dilation_steps'], 'mean_coverage': float(final.mean()),
                   'quality_claim': False}
            rows.append(row)
            write('partial_results.json', {'rows': rows, 'costs': costs, 'clipseg_cost': part.cost})
            return final

        for prompt_mode in plan['prompt_modes']:
            if time.monotonic() - start > plan['max_start_seconds']:
                raise TimeoutError('No new prompt condition after frozen wall deadline')
            seeds = {}
            for name in names:
                resources()
                phrase = 'clothing' if prompt_mode == 'generic' else heads[name]
                part.context = soft[name]
                values = part(video, chunks, phrase)
                part.context = None
                info = postprocess(name, values)
                if prompt_mode == 'generic':
                    archived = torch.load(original / (name + '_garment_frames.pt'), map_location='cpu', weights_only=True)
                    assert all(torch.equal(info['masks'][i], archived['masks'][i]) for i in chunks), 'Unchanged baseline failed exact reproduction'
                final = record(name, prompt_mode, 'per_frame', values, info)
                seeds[name] = torch.from_numpy(final[0])
            resources()
            propagated, seed_images, seconds = propagate_masks(tracker.pred, folder, seeds, n, hw, resources)
            costs.append({'stage': 'sam2_part_propagation', 'prompt_mode': prompt_mode,
                          'wall_seconds': seconds, 'frames': n, 'objects': len(names), 'seed_frame': 0})
            for name in names:
                arrays[name + '_' + prompt_mode + '_seed_image'] = seed_images[name]
                values = judge._soft(propagated[name], n)
                record(name, prompt_mode, 'propagated', values, postprocess(name, values))
        np.savez_compressed(out / 'measured_masks.npz', **arrays)
        result = {'rows': rows, 'costs': costs, 'clipseg_cost': part.cost,
                  'unchanged_generic_masks_exact': True, 'first_seed_frame': 0,
                  'manual_pixel_coordinates': False, 'new_vlm_calls': 0, 'new_video_generations': 0,
                  'quality_claim': False, 'ground_truth_masks': False, 'adopted_arm': None,
                  'mask_arrays_sha256': file_sha256(out / 'measured_masks.npz'),
                  'wall_seconds': time.monotonic() - start}
        write('result.json', result)
        status['status'] = 'completed'
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(wall_seconds=time.monotonic() - start, finished_unix=time.time(), completed_rows=len(rows))
        write('status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'parent', 'original', 'compiled', 'output', 'sam2-checkpoint', 'clipseg-model']:
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(4)
    main(parser.parse_args())
