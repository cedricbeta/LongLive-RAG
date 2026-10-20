#!/usr/bin/env python3
"""One frozen CLIPSeg group-region attempt for a native posture probe."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegForImageSegmentation,CLIPSegProcessor

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(root):
    root=root.resolve();plan=json.loads((root/'plan.json').read_text());out=root/'group_mask';out.mkdir()
    assert sha(Path(__file__))==plan['source_sha256']['mask']
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    started=time.monotonic();status={'status':'running','pid':os.getpid(),'started_unix':time.time()};write('status.json',status)
    try:
        parent=Path(plan['parent']);video,=(parent/'videos').glob('*.mp4')
        assert sha(video)==plan['parent_video_sha256']
        assert sha(parent/'diagnostics/latents_0000.pt')==plan['parent_latents_sha256']
        assert sha(parent/'diagnostics/timing_0000.json')==plan['parent_timing_sha256']
        audit=json.loads((parent/'audit.json').read_text());grid=audit['latent_shape'][-2:]
        recipe=plan['mask_recipe'];modelpath=Path(recipe['model'])
        assets={p.name:sha(p) for p in modelpath.iterdir() if p.is_file()}
        torch.set_num_threads(8)
        processor=CLIPSegProcessor.from_pretrained(modelpath,local_files_only=True)
        model=CLIPSegForImageSegmentation.from_pretrained(modelpath,local_files_only=True).eval()
        cap=cv2.VideoCapture(str(video));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==16
        masks=[];records=[];previews=[]
        try:
            for index in range(*plan['repair_latent_range']):
                frames=[];indices=list(range(4*index-4,4*index))
                assert set(indices)<=set(range(*plan['repair_export_range']))
                for frame in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
                    frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
                inputs=processor(text=[recipe['phrase']]*4,images=frames,return_tensors='pt',padding=True)
                with torch.inference_mode():
                    logits=model(**inputs).logits
                    soft=torch.nn.functional.interpolate(logits.sigmoid()[:,None],size=grid,mode='bilinear',align_corners=False)[:,0]
                body=(soft>=recipe['threshold']).any(dim=0);ys,xs=torch.where(body)
                mask=torch.zeros(grid,dtype=torch.bool);box=None
                if len(xs):
                    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
                    px=math.ceil((x1-x0)*recipe['padding_fraction_xy'][0]);py=math.ceil((y1-y0)*recipe['padding_fraction_xy'][1])
                    box=[max(0,x0-px),max(0,y0-py),min(grid[1],x1+px),min(grid[0],y1+py)]
                    mask[box[1]:box[3],box[0]:box[2]]=True
                masks.append(mask)
                records.append({'latent':index,'frames':indices,'source_rgb_sha256':[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in frames],
                    'program_box_xyxy':box,'segmentation_union_fraction':float(body.float().mean()),'coverage':float(mask.float().mean())})
                if index in plan['mask_preview_latents']:
                    rgb=np.asarray(frames[0]).copy();overlay=rgb.copy()
                    region=cv2.resize(mask.numpy().astype('uint8'),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay[region]=(.65*overlay[region]+.35*np.array([255,0,255])).astype('uint8')
                    canvas=np.concatenate([rgb,overlay],axis=1);Image.fromarray(canvas).save(out/f'preview_{index:04d}.png')
                    previews.append(f'preview_{index:04d}.png')
                write('progress.json',records);print(index,records[-1]['coverage'],flush=True)
        finally:cap.release()
        tensor=torch.stack(masks)
        payload={'mask':tensor,'grid':grid,'latent_range':plan['repair_latent_range'],
            'parent_video_sha256':plan['parent_video_sha256'],'parent_latents_sha256':plan['parent_latents_sha256'],
            'parent_timing_sha256':plan['parent_timing_sha256'],'manual_pixel_coordinates':False,'recipe':recipe,
            'source_sha256':sha(Path(__file__)),'model_assets_sha256':assets}
        torch.save(payload,out/'mask.pt')
        gate=all(0<r['coverage']<=recipe['maximum_coverage_per_latent'] for r in records)
        write('result.json',{'mask_sha256':sha(out/'mask.pt'),'frames_segmented':len(records)*4,'latent_frames':len(records),
            'every_latent_nonempty':bool(tensor.flatten(1).any(1).all()),'maximum_coverage':max(r['coverage'] for r in records),
            'mean_coverage':float(tensor.float().mean()),'program_geometry_gate_passed':gate,'previews':previews,
            'visible_target_coverage_review_pending':True,'independent_quality_evaluation':False})
        status.update(status='completed',program_geometry_gate_passed=gate)
    except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',status)
    print(json.dumps(status),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
