"""Order-batched implementation of the same EF 0.4 score equations.

This execution backend changes no registered support, score bank or program.
All groups of one order are contracted together instead of launching per-group
operators. Training/evaluation source locks identify this implementation.
"""
import torch
from .stable_program import StableProgram,StableOpticalProgram

def vector_scored(model,phi,g,tau=0.):
    mass=torch.as_tensor(g['mass'],device=phi.device,dtype=phi.dtype)
    reduce=lambda a:tau*torch.logsumexp(a/tau,-1) if tau>0 else a.max(-1).values
    typed=torch.einsum('nvd,kmd->nvkm',phi,model.theta);own=reduce(typed)*mass[None,:,None]
    ids=torch.as_tensor(g['child_ids'],device=phi.device);par=torch.as_tensor(g['production_parent'],device=phi.device)
    children=phi[:,ids];cm=mass[ids];context=phi[:,par] if model.config.cross_scale else torch.zeros_like(phi[:,par])
    all_scores=[];all_types=[]
    for order in range(2,min(model.config.max_order,2**g['dim'])+1):
        gi=torch.as_tensor([group for group in g['groups'] if len(group)==order],device=phi.device)
        oi=order-2;members=children[:,:,gi];weights=cm[:,gi];gm=weights.sum(-1)
        mean=(members*weights[None,:,:,:,None]).sum(3)/gm[None,:,:,None]
        spread=((members[...,:-1]-mean[:,:,:,None,:-1]).abs()*weights[None,:,:,:,None]).sum(3)/gm[None,:,:,None]
        parent=context[:,:,None].expand(-1,-1,len(gi),-1);features=torch.cat([mean,spread,parent],-1)
        linear=torch.einsum('npqd,ktd->npqkt',features,model.group_linear[oi])
        product=torch.einsum('npqgd,ktd->npqgkt',members[...,:-1],model.member_projection[oi]).tanh().prod(3)
        gate=1+torch.einsum('npd,ktd->npkt',context[...,:-1],model.parent_projection[oi]).tanh()
        correction=linear+model.gamma[oi][None,None,None]*product*gate[:,:,None]
        anchors=own[:,ids[:,gi]].sum(3)
        all_scores.append(anchors+gm[None,:,:,None]*reduce(correction));all_types.append(correction.argmax(-1))
    return own,torch.cat(all_scores,2),dict(node_types=typed.argmax(-1),group_types=torch.cat(all_types,2))

class VectorStableProgram(StableProgram):
    def scored(self,phi,g,tau=0.):return vector_scored(self,phi,g,tau)

class VectorOpticalProgram(StableOpticalProgram):
    def scored(self,phi,g,tau=0.):return vector_scored(self,phi,g,tau)

def zero_interactions(model):
    """Independent-score ablation in the same registered covering grammar.

    Zero corrections make every group the sum of its independent own anchors;
    recursive unmatched children dominate these groups in exact arithmetic.
    """
    with torch.no_grad():
        for name in ['group_linear','member_projection','parent_projection','gamma']:
            parameter=getattr(model,name);parameter.zero_();parameter.requires_grad_(False)
    return model
