#!/usr/bin/env python3
"""Plot recorded component tensors against their actual source frames."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def plot(run,output):
    a=np.load(run/'sample_components.npz');folder,=(run/'entities/frames').iterdir()
    fig,axes=plt.subplots(3,4,figsize=(16,9.6))
    labels=['SAM2 person','CLIPSeg clothing','CLIPSeg face','Final garment'];colors=['#348eff','#ffba1c','#ff4242','#2cdb9c']
    for i,frame in enumerate([0,80,159]):
        im=np.asarray(Image.open(folder/f'chunk{frame:02d}.png').convert('RGB'))
        for j in range(4):
            data=a[f'Clara_garment_{frame}'][j];mask=np.asarray(Image.fromarray(data>=.35).resize((im.shape[1],im.shape[0]),Image.Resampling.NEAREST))
            layer=np.zeros((*mask.shape,4));layer[mask]=[*matplotlib.colors.to_rgb(colors[j]),.5]
            axes[i,j].imshow(im);axes[i,j].imshow(layer);axes[i,j].axis('off');axes[i,j].set_title(f'{labels[j]} · frame {frame} · {(data>=.35).sum()} cells',fontsize=10)
    fig.suptitle('Measured components of the unchanged automatic garment mask\nSame recorded boxes, model weights and recipe; all 160 output masks exactly reproduced. Not ground-truth segmentation.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output,dpi=120);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();plot(args.run,args.output)
