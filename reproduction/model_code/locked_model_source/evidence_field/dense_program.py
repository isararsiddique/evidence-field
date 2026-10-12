"""Direct FP64 support contractions for the unchanged EF 0.4 mathematics.

Dense masked matrix contractions trade higher arithmetic count for fewer scans,
scatter-backward operations and copies. This is a measured execution hypothesis,
not a new statistical architecture. The prefix implementation remains preserved.
"""
import numpy as np
import torch
from .stable_program import StableRegistryFieldEncoder
from .vector_program import VectorStableProgram,VectorOpticalProgram

class DenseRegistryFieldEncoder(StableRegistryFieldEncoder):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self._support_cache={}
    def mask(self,q,g):
        shape=tuple(q.shape[2:]);key=(shape,str(q.device),q.dtype,g['bounds'].tobytes())
        if key not in self._support_cache:
            grid=np.indices(shape).reshape(len(shape),-1);bounds=g['bounds']
            mask=np.ones((len(bounds),grid.shape[1]),bool)
            for axis in range(len(shape)):
                mask&=(grid[axis][None]>=bounds[:,axis,0,None])&(grid[axis][None]<bounds[:,axis,1,None])
            self._support_cache[key]=torch.from_numpy(mask.T.copy()).to(device=q.device,dtype=q.dtype)
        return self._support_cache[key]
    def forward(self,q,g):
        q=q.double();shape=q.shape[2:];dim=len(shape);device=q.device;dtype=q.dtype
        axes=[(torch.arange(s,device=device,dtype=dtype)+.5)/s*2-1 for s in shape]
        coord=list(torch.meshgrid(*axes,indexing='ij'))
        if dim==2:coord=[torch.zeros_like(coord[0])]+coord
        u=torch.stack(coord);phase=torch.einsum('cp,lc->lp',u.flatten(1),self.wavevectors())
        harmonic=torch.stack([phase.cos(),phase.sin()],1).reshape(2*self.frequencies,*shape)
        basis=torch.cat([torch.stack([torch.ones_like(coord[0]),*coord,
            *[coord[i]*coord[j] for i in range(3) for j in range(i,3)]]),harmonic])
        response=self.response(q);bounds=torch.as_tensor(g['bounds'],device=device);mask=self.mask(q,g);pieces=[]
        for start in range(0,len(basis),4):
            weighted=(response[:,:,None]*basis[None,None,start:start+4]).flatten(1,2).flatten(2)
            total=weighted@mask
            pieces.append(total.transpose(1,2).reshape(len(q),len(bounds),self.responses_count,-1))
        sums=torch.cat(pieces,-1);self.last_sums=sums if not self.training else None
        volume=(bounds[:,:,1]-bounds[:,:,0]).prod(1).to(dtype)
        sizes=torch.tensor(shape,device=device,dtype=dtype)
        midpoint=(bounds[:,:,1]+bounds[:,:,0]).to(dtype)/sizes-1
        halfwidth=(bounds[:,:,1]-bounds[:,:,0]).to(dtype)/sizes
        if dim==2:
            midpoint=torch.cat([torch.zeros_like(midpoint[:,:1]),midpoint],1)
            halfwidth=torch.cat([torch.ones_like(halfwidth[:,:1]),halfwidth],1)
        weight=sums[...,0].clamp(min=1e-6);centre=sums[...,1:4]/weight[...,None]
        cov=torch.stack([(sums[...,4+k]/weight-centre[...,i]*centre[...,j])/
            (halfwidth[None,:,None,i]*halfwidth[None,:,None,j])
            for k,(i,j) in enumerate(((i,j) for i in range(3) for j in range(i,3)))],-1)
        valid=sums[...,0]>1e-6
        local=torch.where(valid[...,None],(centre-midpoint[None,:,None])/halfwidth[None,:,None],0.)
        cov=torch.where(valid[...,None],cov,0.)
        angle=midpoint@self.wavevectors().T;c=angle.cos()[None,:,None];s=angle.sin()[None,:,None]
        raw=sums[...,10:].reshape(len(q),len(bounds),self.responses_count,self.frequencies,2)
        real=(raw[...,0]*c+raw[...,1]*s)/weight[...,None]
        imag=(raw[...,1]*c-raw[...,0]*s)/weight[...,None]
        signatures=torch.stack([real,imag],-1).flatten(3)
        return torch.cat([(sums[...,0]/volume[None,:,None])[...,None],local,cov,signatures],-1).flatten(2)

def install(model):
    c=model.config
    model.encoder=DenseRegistryFieldEncoder(c.responses,c.seed,c.learned_encoder,c.frequencies,c.learned_frequencies,c.frequency_limit,'response_mass')
    return model

class DenseStableProgram(VectorStableProgram):
    def __init__(self,config):super().__init__(config);install(self)

class DenseOpticalProgram(VectorOpticalProgram):
    def __init__(self,config):super().__init__(config);install(self)
