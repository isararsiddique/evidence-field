"""Versioned optical observation contract; preserves RGB absorption channels.

Beer--Lambert optical density and finite differences are established primitives.
This adapter supplies six measurements to the unchanged covering-program core.
It does not identify stains/cells or authenticate scanner calibration.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .adaptive_program import AdaptiveProgram

SPECTRAL_CONTRACT='spectral_optical_density/1.0'

def spectral_measurements(x,spacing=None):
    if x.ndim!=4 or x.shape[1]!=3:raise ValueError('This contract requires N,3,H,W optical RGB')
    if not torch.isfinite(x).all() or bool((x<0).any()) or bool((x>1).any()):raise ValueError('Finite [0,1] RGB required')
    scales=[1.,1.] if spacing is None else list(map(float,spacing))
    if len(scales)!=2 or any(not np.isfinite(v) or v<=0 for v in scales):raise ValueError('Positive finite spacing')
    od=-x.clamp(min=1/256,max=1).log()
    smooth=F.avg_pool2d(od,3,1,1,count_include_pad=False)
    derivatives=[]
    for axis,s in zip((2,3),scales):
        grad=(smooth.roll(-1,axis)-smooth.roll(1,axis))/(2*s)
        lo=[slice(None)]*4;hi=lo.copy();nxt=lo.copy();prv=lo.copy()
        lo[axis]=0;hi[axis]=-1;nxt[axis]=1;prv[axis]=-2
        grad[tuple(lo)]=(smooth[tuple(nxt)]-smooth[tuple(lo)])/s
        grad[tuple(hi)]=(smooth[tuple(hi)]-smooth[tuple(prv)])/s
        derivatives.append(grad)
    magnitude=torch.sqrt(sum(a.square() for a in derivatives)+1e-8)
    return torch.cat([od,magnitude],1)

class OpticalAdaptiveProgram(AdaptiveProgram):
    def __init__(self,config):
        if config.calibration!=SPECTRAL_CONTRACT:raise ValueError('Explicit optical contract required')
        super().__init__(config)
    @torch.no_grad()
    def initialize(self,x,spacing=None):self.encoder.initialize(spectral_measurements(x,spacing))
    def descriptors(self,x,spacing=None):
        g=self.layout(x.shape[2:]);f=self.encoder(spectral_measurements(x,spacing),g)
        mass=torch.as_tensor(g['mass'],device=x.device,dtype=x.dtype)
        parent=torch.as_tensor(g['parent'],device=x.device);depth=np.zeros(len(mass))
        for h,ids in enumerate(g['levels']):depth[ids]=h/g['depth']
        topology=torch.stack([mass,torch.as_tensor(depth,device=x.device,dtype=x.dtype),torch.ones_like(mass)],-1)
        return torch.cat([f,f-f[:,parent],f-f[:,:1],topology[None].expand(len(x),-1,-1)],-1).clamp(-6,6),g
