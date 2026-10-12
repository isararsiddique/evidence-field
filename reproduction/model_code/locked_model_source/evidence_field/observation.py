"""Explicit preview/ROI observation contract for evidence-field research.

The simulator reads full source images for training. The callback interface can
request only selected native ROIs at inference. Missing pixels use a recorded
preview reconstruction; its certificate concerns that observed/reconstructed
input, not an unseen full image. No sensor-energy saving is inferred.
"""
from itertools import product
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceSensor(nn.Module):
    def __init__(self,channels=1,preview_size=8,blocks_per_axis=2):
        super().__init__();self.channels=channels;self.preview_size=preview_size;self.blocks=blocks_per_axis
        self.priority=nn.Parameter(torch.zeros(2*channels));self.priority.data[:channels]=1

    def rank(self,preview):
        dim=preview.ndim-2;features=[]
        for bits in product(range(self.blocks),repeat=dim):
            slices=tuple(slice(i*preview.shape[j+2]//self.blocks,(i+1)*preview.shape[j+2]//self.blocks) for j,i in enumerate(bits))
            part=preview[(slice(None),slice(None),*slices)].flatten(2)
            features.append(torch.cat([part.mean(-1),part.std(-1,unbiased=False)],-1))
        return torch.stack(features,1)@self.priority

    def simulate(self,x,keep=1,mode='learned',generator=None,straight_through=False):
        dim=x.ndim-2;pool=F.adaptive_avg_pool2d if dim==2 else F.adaptive_avg_pool3d
        preview=pool(x,(self.preview_size,)*dim);scores=self.rank(preview);B=scores.shape[1]
        if not 0<=keep<=B:raise ValueError('Invalid ROI count')
        if mode=='random':scores=torch.rand(scores.shape,device=x.device,dtype=x.dtype,generator=generator)
        elif mode=='fixed':scores=torch.arange(B,device=x.device,dtype=x.dtype).neg()[None].expand(len(x),-1)
        elif mode!='learned':raise ValueError('Unknown observation policy')
        chosen=scores.topk(keep,-1).indices;hard=torch.zeros_like(scores).scatter(1,chosen,1.)
        gate=hard
        if straight_through and mode=='learned':
            soft=keep*scores.softmax(-1);gate=hard+soft-soft.detach()
        masks=[];bounds=[]
        for bits in product(range(self.blocks),repeat=dim):
            box=[(i*x.shape[j+2]//self.blocks,(i+1)*x.shape[j+2]//self.blocks) for j,i in enumerate(bits)]
            mask=torch.zeros(x.shape[2:],device=x.device,dtype=x.dtype);mask[tuple(slice(lo,hi) for lo,hi in box)]=1
            masks.append(mask);bounds.append(box)
        masks=torch.stack(masks);observed=torch.einsum('nb,b...->n...',gate,masks)[:,None]
        interpolation='bilinear' if dim==2 else 'trilinear';background=F.interpolate(preview,size=x.shape[2:],mode=interpolation,align_corners=False)
        result=background+observed*(x-background)
        volumes=torch.tensor([m.sum() for m in masks],device=x.device,dtype=x.dtype)
        cost=self.preview_size**dim+(hard*volumes[None]).sum(-1)
        return result,dict(preview=preview,selected_rois=chosen,roi_bounds=bounds,
            observation_samples=cost,full_native_pixels=x[0,0].numel(),
            reconstruction='bilinear/trilinear preview with selected native ROI replacement',
            gradient_estimator='biased straight-through soft gate' if straight_through else 'hard inference',
            scope='Simulated observation budget; preprocessing reads the full source array')

    @torch.no_grad()
    def acquire(self,preview,native_shape,fetch_roi,keep=1):
        """Single-input callback acquisition; fetch_roi(box) returns C,*ROI tensor."""
        if len(preview)!=1:raise ValueError('Single input callback contract')
        dim=len(native_shape);scores=self.rank(preview);chosen=scores.topk(keep,-1).indices[0].tolist()
        result=F.interpolate(preview,size=native_shape,mode='bilinear' if dim==2 else 'trilinear',align_corners=False)
        boxes=[];cost=preview[0,0].numel()
        for index,bits in enumerate(product(range(self.blocks),repeat=dim)):
            if index not in chosen:continue
            box=[(i*native_shape[j]//self.blocks,(i+1)*native_shape[j]//self.blocks) for j,i in enumerate(bits)]
            roi=fetch_roi(box)
            if tuple(roi.shape)!=(self.channels,*[hi-lo for lo,hi in box]):raise ValueError('ROI shape')
            result[(0,slice(None),*(slice(lo,hi) for lo,hi in box))]=roi;boxes.append(box);cost+=roi[0].numel()
        return result,dict(selected_rois=chosen,roi_bounds=boxes,observation_samples=cost,
            observed_input_sha256=hashlib.sha256(result.cpu().contiguous().numpy().tobytes()).hexdigest(),
            scope='Certificate applies to reconstructed observed input; preview acquisition cost must be measured')
