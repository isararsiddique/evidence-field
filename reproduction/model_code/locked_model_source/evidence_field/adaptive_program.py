"""EF adaptive partition and cross-scale hypergroup research implementation.

Finite support alternatives are selected by the score-producing covering program.
This is not a claim of inventing region grammars, tensor products or Fourier units.
The original EF files/checkpoint contracts remain unchanged.
"""
from dataclasses import dataclass, asdict
from itertools import product, combinations
import hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .structural_field import StructuralFieldEncoder, StructuralConfig
from .encoder import measurements


def packing_registry(dim, max_order=3):
    positions=list(product((0,1),repeat=dim)); b=len(positions)
    adjacent=lambda i,j:sum(a!=c for a,c in zip(positions[i],positions[j]))==1
    groups=[]
    for order in range(2,min(max_order,b)+1):
        for group in combinations(range(b),order):
            seen={group[0]}
            while True:
                extra={i for i in group if any(adjacent(i,j) for j in seen)}
                if extra<=seen:break
                seen|=extra
            if len(seen)==order:groups.append(group)
    # Unique group packings; remaining children are recursive singletons.
    packings=[]
    def visit(remaining, selected):
        if not remaining:packings.append(selected);return
        i=min(remaining);visit(remaining-{i},selected)
        for gi,group in enumerate(groups):
            if i in group and set(group)<=remaining:visit(remaining-set(group),selected+[gi])
    visit(set(range(b)),[])
    incidence=np.zeros((len(packings),len(groups)),np.float32)
    unmatched=np.ones((len(packings),b),np.float32)
    for pi,packing in enumerate(packings):
        for gi in packing:incidence[pi,gi]=1;unmatched[pi,list(groups[gi])]=0
    members=np.zeros((len(groups),b),np.float32)
    for gi,group in enumerate(groups):members[gi,list(group)]=1
    return dict(groups=groups,packings=packings,incidence=incidence,
                unmatched=unmatched,members=members,orders=np.array([len(g) for g in groups]))


def adaptive_geometry(shape,depth=2,alternatives=2,max_order=3):
    if len(shape) not in (2,3) or depth<1 or alternatives not in (1,2):
        raise ValueError('2D/3D, depth>=1 and one/two registered partitions required')
    dim=len(shape);b=2**dim
    bounds=[[(0,int(s)) for s in shape]];parent=[0];levels=[[0]];productions=[];production_levels=[]
    for h in range(depth):
        following=[];ps=[]
        for v in levels[-1]:
            cuts=[]
            for alt in range(alternatives):
                mids=[]
                for j,(lo,hi) in enumerate(bounds[v]):
                    if hi-lo<2:raise ValueError('Insufficient spatial size for declared depth')
                    ratio=.5 if alt==0 else (.4 if j%2==0 else .6)
                    mids.append(min(hi-1,max(lo+1,lo+int((hi-lo)*ratio))))
                key=tuple(mids)
                # Keep the same production count even if integer cuts coincide.
                cuts.append(key)
                children=[]
                for bits in product((0,1),repeat=dim):
                    box=[(lo,m) if bit==0 else (m,hi)
                         for (lo,hi),m,bit in zip(bounds[v],mids,bits)]
                    children.append(len(bounds));following.append(len(bounds));bounds.append(box);parent.append(v)
                ps.append(len(productions));productions.append(dict(parent=v,alternative=alt,children=children))
        levels.append(following);production_levels.append(ps)
    bounds=np.array(bounds,int);volume=(bounds[:,:,1]-bounds[:,:,0]).prod(1)
    g=dict(shape=tuple(shape),dim=dim,depth=depth,alternatives=alternatives,bounds=bounds,
        parent=np.array(parent),mass=volume/np.prod(shape),levels=levels,
        productions=productions,production_levels=production_levels,
        child_ids=np.array([p['children'] for p in productions]),
        production_parent=np.array([p['parent'] for p in productions]))
    g.update(packing_registry(dim,max_order));return g


class RegistryFieldEncoder(StructuralFieldEncoder):
    """Prefix contractions on arbitrary registered rectangles; channel-chunked.

    Whole-image responses and references are computed once. Prefix work grows
    with channel count, not with the number of overlapping support alternatives.
    """
    def forward(self,q,g):
        shape=q.shape[2:];dim=len(shape);device=q.device;dtype=q.dtype
        axes=[(torch.arange(s,device=device,dtype=dtype)+.5)/s*2-1 for s in shape]
        coord=list(torch.meshgrid(*axes,indexing='ij'))
        if dim==2:coord=[torch.zeros_like(coord[0])]+coord
        u=torch.stack(coord);phase=torch.einsum('cp,lc->lp',u.flatten(1),self.wavevectors())
        harmonic=torch.stack([phase.cos(),phase.sin()],1).reshape(2*self.frequencies,*shape)
        basis=torch.cat([torch.stack([torch.ones_like(coord[0]),*coord,
            *[coord[i]*coord[j] for i in range(3) for j in range(i,3)]]),harmonic])
        response=self.response(q);bounds=torch.as_tensor(g['bounds'],device=device)
        pieces=[]
        for start in range(0,len(basis),4):
            weighted=(response[:,:,None]*basis[None,None,start:start+4]).flatten(1,2)
            prefix=weighted
            for axis in range(2,2+dim):prefix=prefix.cumsum(axis)
            prefix=F.pad(prefix,tuple(v for _ in range(dim) for v in (1,0)))
            total=0
            for bits in product((0,1),repeat=dim):
                ids=tuple(bounds[:,j,bit] for j,bit in enumerate(bits))
                total=total+(-1)**(dim-sum(bits))*prefix[(slice(None),slice(None),*ids)]
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
            for k,(i,j) in enumerate(( (i,j) for i in range(3) for j in range(i,3) ))],-1)
        valid=sums[...,0]>1e-6
        local=torch.where(valid[...,None],(centre-midpoint[None,:,None])/halfwidth[None,:,None],0.)
        cov=torch.where(valid[...,None],cov,0.)
        angle=midpoint@self.wavevectors().T;c=angle.cos()[None,:,None];s=angle.sin()[None,:,None]
        raw=sums[...,10:].reshape(len(q),len(bounds),self.responses_count,self.frequencies,2)
        real=(raw[...,0]*c+raw[...,1]*s)/weight[...,None]
        imag=(raw[...,1]*c-raw[...,0]*s)/weight[...,None]
        signatures=torch.stack([real,imag],-1).flatten(3)
        return torch.cat([(sums[...,0]/volume[None,:,None])[...,None],local,cov,signatures],-1).flatten(2)


@dataclass
class AdaptiveProgramConfig(StructuralConfig):
    responses:int=4
    node_types:int=2
    pair_types:int=2
    frequencies:int=2
    reference_conditioning:bool=True
    alternatives:int=2
    max_order:int=3
    cross_scale:bool=True
    core_version:str='ef-adaptive-program/0.3'


class AdaptiveProgram(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if config.classes<2 or config.pair_types<1 or config.max_order not in (2,3,4):
            raise ValueError('At least two classes and a valid relation bank required')
        self.encoder=RegistryFieldEncoder(config.responses,config.seed,config.learned_encoder,
            config.frequencies,config.learned_frequencies,config.frequency_limit,'response_mass')
        self.width=3*config.responses*(10+2*config.frequencies)+3;D=self.width;K=config.classes;T=config.pair_types
        generator=torch.Generator().manual_seed(config.seed+29)
        self.theta=nn.Parameter(torch.randn(K,config.node_types,D,generator=generator)*.01)
        self.group_linear=nn.Parameter(torch.randn(config.max_order-1,K,T,3*D-1,generator=generator)*.002)
        self.member_projection=nn.Parameter(torch.randn(config.max_order-1,K,T,D-1,generator=generator)*.02)
        self.parent_projection=nn.Parameter(torch.randn(config.max_order-1,K,T,D-1,generator=generator)*.02)
        self.gamma=nn.Parameter(torch.full((config.max_order-1,K,T),.01));self.bias=nn.Parameter(torch.zeros(K))
        self._geometry={}

    def layout(self,shape):
        shape=tuple(shape)
        if shape not in self._geometry:self._geometry[shape]=adaptive_geometry(shape,self.config.depth,self.config.alternatives,self.config.max_order)
        return self._geometry[shape]

    @torch.no_grad()
    def initialize(self,x,spacing=None):self.encoder.initialize(measurements(x,spacing,self.config.calibration))

    def descriptors(self,x,spacing=None):
        g=self.layout(x.shape[2:]);q=measurements(x,spacing,self.config.calibration);f=self.encoder(q,g)
        device=x.device;mass=torch.as_tensor(g['mass'],device=device,dtype=x.dtype)
        parent=torch.as_tensor(g['parent'],device=device);depth=np.zeros(len(mass))
        for h,ids in enumerate(g['levels']):depth[ids]=h/g['depth']
        topology=torch.stack([mass,torch.as_tensor(depth,device=device,dtype=x.dtype),torch.ones_like(mass)],-1)
        phi=torch.cat([f,f-f[:,parent],f-f[:,:1],topology[None].expand(len(x),-1,-1)],-1).clamp(-6,6)
        return phi,g

    def scored(self,phi,g,tau=0.):
        mass=torch.as_tensor(g['mass'],device=phi.device,dtype=phi.dtype)
        reduce=lambda a:tau*torch.logsumexp(a/tau,-1) if tau>0 else a.max(-1).values
        node_types=torch.einsum('nvd,kmd->nvkm',phi,self.theta);own=reduce(node_types)*mass[None,:,None]
        ids=torch.as_tensor(g['child_ids'],device=phi.device);par=torch.as_tensor(g['production_parent'],device=phi.device)
        ch=phi[:,ids];cm=mass[ids];parent=phi[:,par];scores=[];types=[]
        for group in g['groups']:
            order=len(group);oi=order-2;members=ch[:,:,list(group)];weights=cm[:,list(group)]
            gm=weights.sum(-1);mean=(members*weights[None,:,:,None]).sum(2)/gm[None,:,None]
            spread=((members[...,:-1]-mean[:,:,None,:-1]).abs()*weights[None,:,:,None]).sum(2)/gm[None,:,None]
            context=parent if self.config.cross_scale else torch.zeros_like(parent)
            features=torch.cat([mean,spread,context],-1)
            linear=torch.einsum('npd,ktd->npkt',features,self.group_linear[oi])
            projection=torch.einsum('npgd,ktd->npgkt',members[...,:-1],self.member_projection[oi]).tanh().prod(2)
            modulation=1+torch.einsum('npd,ktd->npkt',context[...,:-1],self.parent_projection[oi]).tanh()
            correction=linear+self.gamma[oi][None,None]*projection*modulation
            scores.append(own[:,ids[:,list(group)]].sum(2)+gm[None,:,None]*reduce(correction));types.append(correction.argmax(-1))
        return own,torch.stack(scores,2),dict(node_types=node_types.argmax(-1),group_types=torch.stack(types,2))

    def compose(self,own,group_scores,g,tau=0.,trace=False):
        values=own.clone();routes=torch.zeros_like(own,dtype=torch.long)
        device=own.device;inc=torch.as_tensor(g['incidence'],device=device,dtype=own.dtype)
        unm=torch.as_tensor(g['unmatched'],device=device,dtype=own.dtype)
        ids=torch.as_tensor(g['child_ids'],device=device);mass=torch.as_tensor(g['mass'],device=device,dtype=own.dtype)
        for h in reversed(range(g['depth'])):
            nodes=g['levels'][h];ps=g['production_levels'][h]
            ch=values[:,ids[ps]]
            options=torch.einsum('npbk,qb->npkq',ch,unm)+torch.einsum('npgk,qg->npkq',group_scores[:,ps],inc)
            options=options.reshape(len(own),len(nodes),g['alternatives'],self.config.classes,-1).permute(0,1,3,2,4).flatten(3)
            options=torch.cat([own[:,nodes,:,None],options],-1)
            if tau>0:
                temp=tau*mass[nodes][None,:,None,None];v=torch.logsumexp(options/temp,-1)*temp[...,0]
            else:v,choice=options.max(-1);routes[:,nodes]=choice
            # Avoid in-place writes into tensors needed by an earlier autograd op.
            indices=torch.as_tensor(nodes,device=device);values=values.index_copy(1,indices,v)
        result=values[:,0]+self.bias
        return (result,routes,values) if trace else result

    def forward(self,x,tau=0.,spacing=None):
        phi,g=self.descriptors(x,spacing);own,groups,_=self.scored(phi,g,tau)
        return self.compose(own,groups,g,tau)

    def fingerprint(self):
        h=hashlib.sha256(json.dumps(asdict(self.config),sort_keys=True).encode())
        for name,a in self.state_dict().items():h.update(name.encode());h.update(a.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def evidence_distribution(self,x,tau=.03,labels=None):
        """Normalized gradient responsibility over own anchors for consistency.

        This training statistic is distinct from the exported hard signed cover.
        """
        phi,g=self.descriptors(x);own,groups,_=self.scored(phi,g,tau)
        z=self.compose(own,groups,g,tau)
        if labels is None:labels=z.detach().argmax(-1)
        weights=torch.autograd.grad(z.gather(1,labels[:,None]).sum(),own,
            create_graph=True,retain_graph=True)[0].gather(2,labels[:,None,None].expand(-1,len(g['mass']),1))[...,0]
        weights=weights.clamp(min=0)+1e-8;weights=weights/weights.sum(1,keepdim=True)
        return z,weights

    @torch.no_grad()
    def certificate(self,x,spacing=None):
        if len(x)!=1:raise ValueError('One input per certificate')
        self.eval();phi,g=self.descriptors(x,spacing);own,groups,types=self.scored(phi,g)
        logits,routes,values=self.compose(own,groups,g,trace=True)
        rows=[]
        for k in range(self.config.classes):
            units=[]
            def visit(v):
                choice=int(routes[0,v,k]);nt=int(types['node_types'][0,v,k])
                if choice==0 or v in g['levels'][-1]:
                    units.append(dict(kind='node',nodes=[v],type=nt,score=float(own[0,v,k])));return
                alt,packing=divmod(choice-1,len(g['packings']));h=next(h for h,level in enumerate(g['levels']) if v in level)
                pi=g['production_levels'][h][g['levels'][h].index(v)*g['alternatives']+alt]
                children=g['child_ids'][pi]
                for gi in g['packings'][packing]:
                    members=children[list(g['groups'][gi])].tolist()
                    anchor=float(own[0,members,k].sum());score=float(groups[0,pi,gi,k])
                    units.append(dict(kind='group',nodes=members,parent=v,production=pi,group=gi,
                        type=int(types['group_types'][0,pi,gi,k]),anchors=anchor,correction=score-anchor,score=score))
                for ci in np.flatnonzero(g['unmatched'][packing]):visit(int(children[ci]))
            visit(0);rows.append(dict(class_index=k,units=units,logit=float(logits[0,k])))
        geometry={key:g[key].tolist() if isinstance(g[key],np.ndarray) else g[key]
            for key in ['shape','dim','depth','alternatives','bounds','parent','mass','levels','productions',
                        'production_levels','child_ids','production_parent','groups','packings','incidence','unmatched','orders']}
        return dict(schema='ef-adaptive-program-certificate/0.3',config=asdict(self.config),
            input_sha256=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            spacing=spacing,model_id=self.fingerprint(),geometry=geometry,phi=phi[0].cpu().numpy().tolist(),
            parameters={name:a.detach().cpu().numpy().tolist() for name,a in self.state_dict().items()},
            classes=rows,logits=logits[0].cpu().numpy().tolist(),
            verification_contract='Requires external trusted parameters and independently computed descriptors')
