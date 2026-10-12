"""Exact execution, rule challenges and ensemble interfaces for adaptive EF.

Bounded composition skips packing reductions only, after descriptor/candidate
computation. Review statistics are numerical, not validated clinical thresholds.
"""
import numpy as np
import torch


@torch.no_grad()
def bounded_compose(model,own,groups,g):
    device=own.device;dtype=own.dtype;N,_,K=own.shape;A=g['alternatives']
    values=own.clone();inc=torch.as_tensor(g['incidence'],device=device,dtype=dtype)
    unm=torch.as_tensor(g['unmatched'],device=device,dtype=dtype)
    ids=torch.as_tensor(g['child_ids'],device=device)
    membership=torch.as_tensor(g['members'].T,device=device,dtype=torch.bool)
    orders=torch.as_tensor(g['orders'],device=device,dtype=dtype)
    skipped=0;total=0
    for h in reversed(range(g['depth'])):
        nodes=g['levels'][h];ps=g['production_levels'][h];P=len(ps);V=len(nodes)
        child=values[:,ids[ps]].permute(0,1,3,2).reshape(-1,2**g['dim'])
        scores=groups[:,ps].permute(0,1,3,2).reshape(-1,len(g['groups']))
        shares=scores/orders[None]
        endpoint=shares[:,None,:].expand(-1,2**g['dim'],-1).masked_fill(~membership[None],float('-inf')).max(-1).values
        U=torch.maximum(child,endpoint).sum(-1).reshape(N,V,A,K).max(2).values
        coarse=own[:,nodes]
        guard=32*torch.finfo(dtype).eps*(1+coarse.abs()+U.abs())
        dominates=coarse>U+guard;eligible=(~dominates)[:,:,None,:].expand(-1,-1,A,-1).reshape(-1)
        active=eligible.nonzero().ravel();out=coarse[:,:,None,:].expand(-1,-1,A,-1).reshape(-1).clone()
        if len(active):
            candidates=child[active]@unm.T+scores[active]@inc.T
            out[active]=torch.maximum(out[active],candidates.max(-1).values)
        chosen=out.reshape(N,V,A,K).max(2).values
        values=values.index_copy(1,torch.as_tensor(nodes,device=device),chosen)
        skipped+=int(dominates.sum());total+=dominates.numel()
    return values[:,0]+model.bias,dict(skipped_parent_class_reductions=skipped,parent_class_reductions=total,
        scope='Packing reduction only; field, descriptor and group scoring still evaluated')


@torch.no_grad()
def relation_challenge(model,x,forbidden_orders,spacing=None):
    """Remove group productions at fixed descriptors; preserve own and singleton cover."""
    phi,g=model.descriptors(x,spacing);own,groups,_=model.scored(phi,g)
    original=model.compose(own,groups,g);inc=np.asarray(g['incidence'])
    forbidden=np.isin(g['orders'],list(forbidden_orders))
    keep=~(inc[:,forbidden].sum(-1)>0)
    restricted={**g,'incidence':g['incidence'][keep],'unmatched':g['unmatched'][keep]}
    result=model.compose(own,groups,restricted)
    if (result>original+1e-5).any():raise ArithmeticError('Rule restriction increased score')
    return dict(original_logits=original,restricted_logits=result,forbidden_orders=list(forbidden_orders),
        permitted_packings=int(keep.sum()),scope='Fixed-descriptor grammar intervention, not a clinical counterfactual')


@torch.no_grad()
def root_route_statistics(model,x,temperature=.1):
    phi,g=model.descriptors(x);own,groups,_=model.scored(phi,g)
    z,_,values=model.compose(own,groups,g,trace=True)
    ps=g['production_levels'][0];ids=torch.as_tensor(g['child_ids'][ps],device=x.device)
    inc=torch.as_tensor(g['incidence'],device=x.device,dtype=x.dtype)
    unm=torch.as_tensor(g['unmatched'],device=x.device,dtype=x.dtype)
    options=torch.einsum('npbk,qb->npkq',values[:,ids],unm)+torch.einsum('npgk,qg->npkq',groups[:,ps],inc)
    options=options.permute(0,2,1,3).flatten(2);options=torch.cat([own[:,0,:,None],options],-1)
    best=options.topk(2,-1).values;prob=(options/temperature).softmax(-1)
    return dict(logits=z,root_production_margin=best[...,0]-best[...,1],
        root_production_entropy=-(prob*prob.clamp(min=1e-12).log()).sum(-1),
        scope='Conditional root-production ambiguity; not all-program or biological uncertainty')


@torch.no_grad()
def ensemble_predictions(models,x,temperature=1.):
    """Arithmetic average of member logits; retains each member program/certificate."""
    logits=torch.stack([m(x) for m in models]);mean=logits.mean(0)
    p=(logits/temperature).softmax(-1);avg=p.mean(0)
    entropy=lambda v:-(v*v.clamp(min=1e-12).log()).sum(-1)
    return dict(logits=mean,probabilities=(mean/temperature).softmax(-1),
        member_logits=logits,predictive_entropy=entropy(avg),
        disagreement=entropy(avg)-entropy(p).mean(0),
        scope='Ensemble numerical disagreement; requires application calibration')


@torch.no_grad()
def pixel_interventions(model,x,fraction=.1,seed=11):
    """Matched-count high/low/random evidence replacement on one actual input.

    The current-image references are recomputed; this is an engineering stress
    probe, not lesion removal or a validated biological counterfactual.
    """
    if len(x)!=1 or not 0<fraction<1:raise ValueError('Single input and proper fraction required')
    cert=model.certificate(x);k=int(np.argmax(cert['logits']));shape=x.shape[2:]
    density=np.zeros(shape,float);bounds=cert['geometry']['bounds']
    for unit in cert['classes'][k]['units']:
        count=sum(np.prod([hi-lo for lo,hi in bounds[v]]) for v in unit['nodes'])
        for v in unit['nodes']:density[tuple(slice(lo,hi) for lo,hi in bounds[v])]=unit['score']/count
    count=max(1,int(np.prod(shape)*fraction));order=np.argsort(abs(density).ravel(),kind='stable')
    rng=np.random.default_rng(seed);selections=dict(high=order[-count:],low=order[:count],random=rng.choice(len(order),count,replace=False))
    original=model(x);rows={}
    for mode,ids in selections.items():
        changed=x.clone();flat=changed.flatten(2);replacement=x.flatten(2).mean(-1,keepdim=True)
        flat[:,:,torch.as_tensor(ids.copy(),device=x.device)]=replacement
        result=model(changed);rows[mode]=dict(logits=result,changed_pixels=count,
            predicted_class_logit_change=float(original[0,k]-result[0,k]))
    return dict(original_logits=original,interventions=rows,predicted_class=k,
        scope='Matched-count pixel replacement with recomputed references; no medical causality claim')
