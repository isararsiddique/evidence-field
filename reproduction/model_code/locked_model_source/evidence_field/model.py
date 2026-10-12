"""End-to-end learned fields, signed cover recursion, and explicit records."""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
import numpy as np
import torch
from torch import nn
from .encoder import FieldEncoder,measurements
from .geometry import geometry

@dataclass
class ModelConfig:
    classes:int=2
    responses:int=16
    node_types:int=4
    pair_types:int=2
    depth:int=2
    seed:int=11
    learned_encoder:bool=True
    readout:str='cover'  # cover, omit, dense, linear
    bilinear:bool=True
    calibration:str='display'
    unit_penalty:float=0.
    reference_conditioning:bool=False

class EvidenceField(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__();self.config=config
        if config.classes<2 or config.node_types<1:raise ValueError('At least two classes and one node type')
        if config.readout not in ('cover','omit','dense','linear'):raise ValueError('Unknown readout')
        if config.reference_conditioning:
            from .adaptive_encoder import AdaptiveFieldEncoder
            self.encoder=AdaptiveFieldEncoder(config.responses,config.seed,config.learned_encoder)
        else:self.encoder=FieldEncoder(config.responses,config.seed,config.learned_encoder)
        self.width=3*10*config.responses+3
        rng=np.random.default_rng(config.seed+29)
        self.theta=nn.Parameter(torch.tensor(rng.normal(0,.01,(config.classes,config.node_types,self.width)),dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(config.classes))
        if config.pair_types:
            self.pair=nn.Parameter(torch.tensor(rng.normal(0,.002,(config.classes,config.pair_types,2*self.width-1)),dtype=torch.float32))
            self.pair.data[:,:,self.width-1]=-.01
            self.projection=nn.Parameter(torch.tensor(rng.normal(0,.02,(config.classes,config.pair_types,self.width-1)),dtype=torch.float32),requires_grad=config.bilinear)
            self.gamma=nn.Parameter(torch.full((config.classes,config.pair_types),.01),requires_grad=config.bilinear)
        else:
            self.register_parameter('pair',None);self.register_parameter('projection',None);self.register_parameter('gamma',None)
        self._geometry={}
    def layout(self,shape):
        key=tuple(shape)
        if key not in self._geometry:self._geometry[key]=geometry(shape,self.config.depth)
        return self._geometry[key]
    def descriptors(self,x,spacing=None):
        g=self.layout(x.shape[2:]);q=measurements(x,spacing,self.config.calibration);f=self.encoder(q,g)
        parent=torch.tensor(np.maximum(g['parent'],0),device=x.device)
        differences=f-f[:,parent];root=f-f[:,:1]
        mass=torch.as_tensor(g['mass'],device=x.device,dtype=x.dtype)
        depth=np.zeros(len(mass))
        for i,ids in enumerate(g['levels']):depth[ids]=i/max(1,self.config.depth)
        geometry_features=torch.stack([mass,torch.as_tensor(depth,device=x.device,dtype=x.dtype),torch.ones_like(mass)],-1)
        phi=torch.cat([f,differences,root,geometry_features[None].expand(len(x),-1,-1)],-1).clamp(-6,6)
        return phi,g
    def scored(self,phi,g,tau=0.):
        device=phi.device;mass=torch.as_tensor(g['mass'],device=device,dtype=phi.dtype)
        h=torch.einsum('nvd,kmd->nvkm',phi,self.theta)
        reduce=lambda z:torch.logsumexp(z/tau,-1)*tau if tau>0 else z.max(-1).values
        own=reduce(h)*mass[None,:,None]
        first=torch.as_tensor(g['first'],device=device);second=torch.as_tensor(g['second'],device=device)
        if self.config.pair_types:
            a=phi[:,first];b=phi[:,second];ma=mass[first];mb=mass[second]
            chi=torch.cat([(a*ma[None,:,None]+b*mb[None,:,None])/(ma+mb)[None,:,None],.5*(a[:,:,:-1]-b[:,:,:-1]).abs()],-1)
            ps=torch.einsum('npd,kmd->npkm',chi,self.pair)
            if self.config.bilinear:
                ap=torch.einsum('npd,kmd->npkm',a[:,:,:-1],self.projection)
                bp=torch.einsum('npd,kmd->npkm',b[:,:,:-1],self.projection)
                ps=ps+ap*bp*self.gamma[None,None]
            pair=reduce(ps)*(ma+mb)[None,:,None]+own[:,first]+own[:,second]
        else:chi=phi.new_zeros(len(phi),len(first),2*self.width-1);ps=None;pair=None
        return own,pair,dict(phi=phi,own=own,pair=pair,node_types=h.argmax(-1),pair_types=None if ps is None else ps.argmax(-1),g=g,chi=chi)
    def compose(self,own,pair,g,tau=0.,return_routes=False):
        if self.config.readout=='linear':
            # Conventional learned encoder + root linear head, represented by node type 0.
            return own[:,0]/float(g['mass'][0])+self.bias
        if self.config.readout=='dense':return own[:,g['levels'][-1]].sum(1)+self.bias
        device=own.device;levels=g['levels'];branch=2**g['dim']
        routes=torch.zeros_like(own,dtype=torch.long) if return_routes else None
        mass=torch.as_tensor(g['mass'],device=device,dtype=own.dtype)
        value=own[:,levels[-1]]
        if self.config.readout=='omit':value=value.clamp(min=0)
        incidence=torch.as_tensor(g['incidence'],device=device,dtype=own.dtype)
        unmatched=torch.as_tensor(g['unmatched'],device=device,dtype=own.dtype)
        for level in reversed(range(len(levels)-1)):
            ids=levels[level];child=value.reshape(len(own),len(ids),branch,self.config.classes).permute(0,1,3,2)
            if pair is None:split=child.sum(-1,keepdim=True)
            else:
                pv=pair[:,g['pair_levels'][level]].permute(0,1,3,2)
                split=torch.einsum('nvkb,qb->nvkq',child,unmatched)+torch.einsum('nvke,qe->nvkq',pv,incidence)
            coarse=own[:,ids,:,None]-self.config.unit_penalty
            options=torch.cat([coarse,split],-1)
            if self.config.readout=='omit':options=torch.cat([options,torch.zeros_like(coarse)],-1)
            if tau>0:
                temperature=tau*mass[ids][None,:,None,None]
                value=torch.logsumexp(options/temperature,-1)*temperature[:,:,:,0]
            else:
                value,choice=options.max(-1)
                if return_routes:routes[:,ids]=choice
        logits=value[:,0]+self.bias
        return (logits,routes) if return_routes else logits
    def forward(self,x,tau=0.,spacing=None):
        phi,g=self.descriptors(x,spacing)
        if self.config.readout=='linear':
            return phi[:,0]@self.theta[:,0].T+self.bias
        own,pair,_=self.scored(phi,g,tau)
        return self.compose(own,pair,g,tau)
    @torch.no_grad()
    def initialize(self,source_images,spacing=None):
        self.encoder.initialize(measurements(source_images,spacing,self.config.calibration))
    def fingerprint(self):
        h=sha256(json.dumps(asdict(self.config),sort_keys=True).encode())
        for key,value in self.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    @torch.no_grad()
    def records(self,x,spacing=None,temperature=1.,input_ids=None):
        if self.config.readout!='cover' or self.config.unit_penalty:
            raise ValueError('Signed-cover record exporter currently requires cover with zero penalty')
        self.eval();phi,g=self.descriptors(x,spacing);own,pair,cache=self.scored(phi,g,0.)
        logits,routes=self.compose(own,pair,g,0.,return_routes=True);model_id=self.fingerprint();cache['routes']=routes
        cache['encoder_state']={k:v.detach().cpu().numpy().tolist() for k,v in self.encoder.state_dict().items()}
        if self.config.reference_conditioning:
            _,mean,scale=self.encoder.reference(measurements(x,spacing,self.config.calibration))
            cache['image_reference_mean']=mean.flatten(1).cpu().numpy()
            cache['image_reference_scale']=scale.flatten(1).cpu().numpy()
        cache={key:value.detach().cpu().numpy() if torch.is_tensor(value) else value for key,value in cache.items()}
        from .replay import build_records
        return logits,build_records(self,cache,logits.cpu().numpy(),x.detach().cpu().numpy(),spacing,temperature,input_ids,model_id)
