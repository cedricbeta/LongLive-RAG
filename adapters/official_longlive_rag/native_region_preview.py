"""Display stored native latent masks on their exact source frame coordinates."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from utils.agent_bounded_feedback import mask_geometry
from utils.agent_runtime import file_sha256


def preview(option, *, video, parent_latents, output_dir, timing):
    path = option['action']['params']['mask_path']
    if file_sha256(path) != option['mask_sha256']:
        raise ValueError('Offered native mask changed')
    geometry = mask_geometry(path, parent_video=video, parent_latents=parent_latents)
    coordinates = geometry['coordinates']
    if coordinates['format'] != 'native':
        raise ValueError('Expected native timing coordinates')
    first, last = coordinates['latent_range']
    selected = sorted({first, (first + last) // 2, last - 1})
    # Each mask is the union of the four frames associated with its latent.
    indices = [0 if i == 0 else 4*i-4 for i in selected]
    if max(indices) >= timing['export_frames']:
        raise ValueError('Preview exceeds the native video')
    source = file_sha256(video)
    key = hashlib.sha256(json.dumps([source, geometry['sha256'], indices]).encode()).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (key + '.png')
    if not target.exists():
        canvas = Image.new('RGB', (960, 35 + 290*len(indices)), '#181818')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'Parent | RED: actual latent mask (four-frame union; not a pixel guarantee)', fill='white')
        cap = cv2.VideoCapture(str(video))
        try:
            for row, (latent, frame) in enumerate(zip(selected, indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError('Cannot read native mask source frame')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = geometry['values'][first][latent-first]
                alpha = cv2.resize(mask.numpy(), (rgb.shape[1], rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)[..., None] * .5
                overlay = np.uint8(rgb*(1-alpha) + np.array([255, 30, 30])*alpha)
                for column, value in enumerate((rgb, overlay)):
                    canvas.paste(ImageOps.pad(Image.fromarray(value), (480, 270)), (480*column, 35+row*290))
                draw.text((8, 307+row*290), f'Latent {latent}, source export frame {frame}', fill='white')
        finally:
            cap.release()
        canvas.save(target)
    return {'path': str(target), 'sha256': file_sha256(target), 'source_video_sha256': source,
            'frame_indices': indices, 'mask_content_sha256': geometry['sha256'],
            'coverage': float(geometry['values'][first].mean())}
