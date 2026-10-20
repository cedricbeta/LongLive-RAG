"""Validated parent-clean-latent input for offline progressive replay.

The pipeline still reconstructs its caches by replaying the prefix. This module
does not serialize KV state or imply streaming. It fixes the clean parent values
used by each subsequent local edit and by prefix recache.
"""
from pathlib import Path

import torch

from utils.agent_runtime import file_sha256


class ParentLatentReplay:
    def __init__(self, config):
        self.config = dict(config)
        for key in ('latent_path', 'latent_sha256', 'video_path', 'video_sha256',
                    'chunk_start', 'chunk_end', 'num_chunks'):
            if key not in self.config:
                raise ValueError(f'Parent replay requires {key}')
        for kind in ('latent', 'video'):
            path = Path(self.config[f'{kind}_path'])
            if file_sha256(path) != self.config[f'{kind}_sha256']:
                raise ValueError(f'Parent replay {kind} content changed')
        self.start = int(self.config['chunk_start'])
        self.end = int(self.config['chunk_end'])
        self.num_chunks = int(self.config['num_chunks'])
        if not 0 <= self.start <= self.end < self.num_chunks:
            raise ValueError('Invalid parent replay chunk range')
        self.latent = torch.load(self.config['latent_path'], map_location='cpu', weights_only=True)
        if not isinstance(self.latent, torch.Tensor) or self.latent.ndim != 5:
            raise ValueError('Parent replay needs a raw [B,T,C,H,W] tensor, not digests')
        if not self.latent.is_floating_point() or self.latent.shape[1] % self.num_chunks:
            raise ValueError('Parent latent dtype or temporal layout is incompatible')
        if not torch.isfinite(self.latent).all():
            raise ValueError('Parent latent contains nonfinite values')
        self.frames_per_chunk = self.latent.shape[1] // self.num_chunks

    def chunk(self, index, current):
        index = int(index)
        if not 0 <= index <= self.end:
            raise ValueError('Cannot pin a chunk outside the declared parent prefix/shot')
        value = self.latent[:, index * self.frames_per_chunk:(index + 1) * self.frames_per_chunk]
        if value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError('Parent replay shape/dtype differs from current generation')
        return value.to(current.device).clone()

    def before_edit(self, index, current):
        return self.chunk(index, current) if self.start <= index <= self.end else current

    def before_recache(self, index, current):
        # Edited chunks retain the edit's result; the strict prefix uses the
        # parent's exact clean latents, then enters ordinary clean recache.
        return self.chunk(index, current) if index < self.start else current


def validate_parent_mask(path, *, parent_video, chunk_start, chunk_end,
                         frames_per_chunk, grid):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not payload.get('video_sha256'):
        raise ValueError('Progressive mask has no source-video hash')
    if payload['video_sha256'] != file_sha256(parent_video):
        raise ValueError('Progressive mask belongs to a different parent video')
    if payload.get('frames_per_chunk') != frames_per_chunk or tuple(payload.get('grid', ())) != tuple(grid):
        raise ValueError('Progressive mask coordinate system differs from the backend')
    masks = payload.get('masks', {})
    if set(masks) != set(range(chunk_start, chunk_end + 1)):
        raise ValueError('Progressive mask must cover exactly the declared current shot')
    for value in masks.values():
        if tuple(value.shape) != (frames_per_chunk, *grid) or not torch.isfinite(value).all():
            raise ValueError('Invalid progressive mask shape or values')
        if value.min() < 0 or value.max() > 1:
            raise ValueError('Progressive mask weights must lie in [0,1]')
    return payload
