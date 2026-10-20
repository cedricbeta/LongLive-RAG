#!/usr/bin/env python3
"""Audit recorded per-frame/native masks and plot actual measured regions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(run,output):
    run=run.resolve();output.mkdir(parents=True,exist_ok=False)
    result=json.loads((run/'result.json').read_text());rows=[];frames=[0,80,159]
    folder,=(run/'entities/frames').iterdir()
    fig,axes=plt.subplots(3,4,figsize=(16,9.6),squeeze=False)
    colors=['#ffb020','#ffb020','#2be7c1','#2be7c1']
    for col,row in enumerate(result['masks']):
        name=row['entity']+'_'+row['part'];native_path=run/(name+'_native.pt')
        assert hashlib.sha256(native_path.read_bytes()).hexdigest()==row['sha256']
        d=torch.load(native_path,map_location='cpu',weights_only=True)
        frame_path=run/(name+'_frames.pt');protected_path=run/(name+'_protected.pt')
        assert hashlib.sha256(frame_path.read_bytes()).hexdigest()==d['source_mask_sha256']
        assert hashlib.sha256(protected_path.read_bytes()).hexdigest()==d['source_protected_sha256']
        f=torch.load(frame_path,map_location='cpu',weights_only=True);p=torch.load(protected_path,map_location='cpu',weights_only=True)
        for i,mapping in enumerate(d['source_frame_mapping']):
            union=torch.stack([f['masks'][k][0]>=.35 for k in mapping['export_frames']]).any(0)
            exclusion=torch.stack([p['masks'][k][0]>=.35 for k in mapping['export_frames']]).any(0)
            assert torch.equal(d['mask'][i],union&~exclusion)
            assert not (d['mask'][i]&exclusion).any()
        coverage=d['mask'].float().flatten(1).mean(1)
        np.savez_compressed(output/(name+'.npz'),native_mask=d['mask'].numpy(),
            export_frame_indices=np.array(sorted(f['masks'])),
            frame_masks=np.stack([f['masks'][k].numpy() for k in sorted(f['masks'])]),
            protected=np.stack([p['masks'][k].numpy() for k in sorted(p['masks'])]))
        meta={k:v for k,v in d.items() if k!='mask'};(output/(name+'_metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')
        rows.append({'entity':row['entity'],'part':row['part'],'latent_frames':len(coverage),
            'empty_latent_frames':int((coverage==0).sum()),'mean_coverage':float(coverage.mean()),
            'minimum_coverage':float(coverage.min()),'maximum_coverage':float(coverage.max()),
            'exact_union_minus_protection':True,'protected_overlap_cells':0})
        for r,index in enumerate(frames):
            image=np.asarray(Image.open(folder/f'chunk{index:02d}.png').convert('RGB'))
            # Actual export-frame mask, rather than a larger temporal union.
            mask=(f['masks'][index][0]>=.35)&~(p['masks'][index][0]>=.35)
            mask=np.asarray(Image.fromarray(mask.numpy()).resize((image.shape[1],image.shape[0]),Image.Resampling.NEAREST))
            ax=axes[r,col];ax.imshow(image);layer=np.zeros((*mask.shape,4));layer[mask]=[*matplotlib.colors.to_rgb(colors[col]),.5];ax.imshow(layer)
            ax.set_title(f"{row['entity']} {row['part']} · export frame {index}",fontsize=10);ax.axis('off')
    fig.suptitle('Automatic actor/part localization on original native video\n160 frames tracked; shown frames 0 / 80 / 159. Colored regions are measured masks, not ground truth.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'region_preview.png',dpi=120);plt.close(fig)
    audit={'source_result_sha256':hashlib.sha256((run/'result.json').read_bytes()).hexdigest(),'masks':rows,
        'new_video_generations':0,'quality_claim':False,'ground_truth_localization':False,
        'sam2_postprocessing':'Optional native extension unavailable; hole filling skipped, as recorded in worker log.',
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.run,args.output)
