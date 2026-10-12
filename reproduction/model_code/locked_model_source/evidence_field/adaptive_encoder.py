"""Dual absolute/reference-conditioned image fields, with learned mixture weights.

Per-image normalization and radial units have prior art. The hypothesis is that
their recorded mixture inside signed-cover programs helps acquisition changes.
The current-image reference is not assumed healthy.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .encoder import FieldEncoder

class AdaptiveFieldEncoder(FieldEncoder):
    def __init__(self,responses=16,seed=11,learned=True):
        super().__init__(responses,seed,learned)
        self.mixture_logits=nn.Parameter(torch.zeros(responses),requires_grad=learned)
        self.register_buffer('relative_mean',torch.zeros(6));self.register_buffer('relative_scale',torch.ones(6))
    def reference(self,q):
        axes=tuple(range(2,q.ndim));mean=q.mean(axes,keepdim=True)
        scale=q.std(axes,keepdim=True,unbiased=False).clamp(min=.03)
        return (q-mean)/scale,mean,scale
    @torch.no_grad()
    def initialize(self,q_samples):
        super().initialize(q_samples)
        relative,_,_=self.reference(q_samples);flat=relative.movedim(1,-1).reshape(-1,6)
        self.relative_mean.copy_(flat.mean(0));self.relative_scale.copy_(flat.std(0).clamp(min=.03))
        self.mixture_logits.zero_()
    def response(self,q,reference=None):
        absolute=super().response(q)
        if reference is None:relative,_,_=self.reference(q)
        else:
            mean,scale=reference;relative=(q-mean)/scale
        z=((relative.movedim(1,-1)-self.relative_mean)/self.relative_scale).clamp(-6,6)
        delta=z.unsqueeze(-2)-self.centres;precision=F.softplus(self.log_precision)+1e-3
        response=torch.exp(-.5*(delta.square()*precision).sum(-1)).movedim(-1,1)
        alpha=torch.sigmoid(self.mixture_logits).reshape(1,-1,*([1]*(q.ndim-2)))
        return alpha*absolute+(1-alpha)*response
