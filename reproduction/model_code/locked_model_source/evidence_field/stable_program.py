"""EF 0.4: guarded FP64 field construction, original signed program arithmetic.

FP32 integral-sum subtraction on small/low-mass regions can corrupt normalized
moments. Build responses, references and associative contractions in FP64, then
cast the completed descriptors to the classifier's dtype. The original 0.3
implementation/checkpoints are retained for the numerical failure record.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .adaptive_program import AdaptiveProgram,AdaptiveProgramConfig,RegistryFieldEncoder
from .medical_input import OpticalAdaptiveProgram,SPECTRAL_CONTRACT
from .medical_input import spectral_measurements
from .encoder import measurements

@dataclass
class StableProgramConfig(AdaptiveProgramConfig):
    core_version:str='ef-adaptive-program/0.4'

class StableRegistryFieldEncoder(RegistryFieldEncoder):
    def wavevectors(self):return self.frequency_limit*self.raw_wavevectors.double().tanh()
    def response(self,q,reference=None):
        q=q.double();centres=self.centres.double();precision=F.softplus(self.log_precision.double())+1e-3
        absolute=((q.movedim(1,-1)-self.q_mean.double())/self.q_scale.double()).clamp(-6,6)
        absolute=torch.exp(-.5*((absolute.unsqueeze(-2)-centres).square()*precision).sum(-1)).movedim(-1,1)
        if reference is None:relative,_,_=self.reference(q)
        else:relative=(q-reference[0])/reference[1]
        z=((relative.movedim(1,-1)-self.relative_mean.double())/self.relative_scale.double()).clamp(-6,6)
        relative=torch.exp(-.5*((z.unsqueeze(-2)-centres).square()*precision).sum(-1)).movedim(-1,1)
        alpha=self.mixture_logits.double().sigmoid().reshape(1,-1,*([1]*(q.ndim-2)))
        return alpha*absolute+(1-alpha)*relative
    def forward(self,q,g):
        dtype=q.dtype
        return super().forward(q.double(),g).to(dtype)

def install_stable_encoder(model):
    c=model.config
    model.encoder=StableRegistryFieldEncoder(c.responses,c.seed,c.learned_encoder,c.frequencies,c.learned_frequencies,c.frequency_limit,'response_mass')
    return model

def stable_descriptors(model,x,spacing,measurement):
    g=model.layout(x.shape[2:]);q=measurement(x.double(),spacing)
    f=model.encoder(q,g).to(x.dtype)
    mass=torch.as_tensor(g['mass'],device=x.device,dtype=x.dtype)
    parent=torch.as_tensor(g['parent'],device=x.device);depth=np.zeros(len(mass))
    for h,ids in enumerate(g['levels']):depth[ids]=h/g['depth']
    topology=torch.stack([mass,torch.as_tensor(depth,device=x.device,dtype=x.dtype),torch.ones_like(mass)],-1)
    return torch.cat([f,f-f[:,parent],f-f[:,:1],topology[None].expand(len(x),-1,-1)],-1).clamp(-6,6),g

class StableProgram(AdaptiveProgram):
    def __init__(self,config):
        if config.core_version!='ef-adaptive-program/0.4':raise ValueError('Explicit stable version required')
        super().__init__(config);install_stable_encoder(self)
    @torch.no_grad()
    def initialize(self,x,spacing=None):self.encoder.initialize(measurements(x.double(),spacing,self.config.calibration))
    def descriptors(self,x,spacing=None):
        return stable_descriptors(self,x,spacing,lambda a,s:measurements(a,s,self.config.calibration))

class StableOpticalProgram(OpticalAdaptiveProgram):
    def __init__(self,config):
        if config.core_version!='ef-adaptive-program/0.4':raise ValueError('Explicit stable version required')
        super().__init__(config);install_stable_encoder(self)
    @torch.no_grad()
    def initialize(self,x,spacing=None):self.encoder.initialize(spectral_measurements(x.double(),spacing))
    def descriptors(self,x,spacing=None):return stable_descriptors(self,x,spacing,spectral_measurements)
