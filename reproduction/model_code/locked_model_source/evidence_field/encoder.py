"""Learned radial fields and vectorized integral moments, with no learned spatial filter.

The six input measurements are display intensity, smooth residual, three
axis derivatives, and gradient magnitude. 2D occupies the y/x plane with z=0.
Physical units must be supplied by a validated adapter; display data have none.
"""
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def measurements(x,spacing=None,mode='display'):
    dim=x.ndim-2
    if dim not in (2,3):raise ValueError('Expected N,C,H,W or N,C,D,H,W')
    if mode=='optical_density':
        # Preserve a scalar OD plus channel spread rather than assigning stain identity.
        od=-torch.log(x.clamp(min=1/256,max=1));a=od.mean(1,keepdim=True)/3
    elif mode=='display':a=x.mean(1,keepdim=True)
    elif mode=='ct_hu':a=(x.mean(1,keepdim=True).clamp(-1000,2000)+1000)/3000
    else:raise ValueError('Calibration mode is not implemented; never guess units')
    pool=F.avg_pool2d if dim==2 else F.avg_pool3d
    smooth=pool(a,3,1,1,count_include_pad=False);residual=a-smooth
    scales=[1.]*dim if spacing is None else [float(v) for v in spacing]
    if len(scales)!=dim or any(v<=0 for v in scales):raise ValueError('Positive spacing per spatial axis required')
    derivatives=[]
    for j,s in enumerate(scales):
        axis=j+2;f=torch.roll(smooth,-1,axis);b=torch.roll(smooth,1,axis)
        grad=(f-b)/(2*s)
        # No circular edge dependence: one-sided finite differences at borders.
        lo=[slice(None)]*x.ndim;hi=lo.copy();nxt=lo.copy();prv=lo.copy()
        lo[axis]=0;hi[axis]=-1;nxt[axis]=1;prv[axis]=-2
        grad[tuple(lo)]=(smooth[tuple(nxt)]-smooth[tuple(lo)])/s
        grad[tuple(hi)]=(smooth[tuple(hi)]-smooth[tuple(prv)])/s
        derivatives.append(grad)
    if dim==2:derivatives=[torch.zeros_like(a)]+derivatives
    mag=torch.sqrt(sum(z.square() for z in derivatives)+1e-8)
    return torch.cat([a,residual,*derivatives,mag],1)

class FieldEncoder(nn.Module):
    def __init__(self,responses=16,seed=11,learned=True):
        super().__init__();self.responses_count=responses
        rng=np.random.default_rng(seed)
        self.centres=nn.Parameter(torch.tensor(rng.normal(0,1,(responses,6)),dtype=torch.float32),requires_grad=learned)
        self.log_precision=nn.Parameter(torch.zeros(responses,6),requires_grad=learned)
        self.register_buffer('q_mean',torch.zeros(6));self.register_buffer('q_scale',torch.ones(6))
        self._cache={}
    @torch.no_grad()
    def initialize(self,q_samples):
        flat=q_samples.movedim(1,-1).reshape(-1,6)
        self.q_mean.copy_(flat.mean(0));self.q_scale.copy_(flat.std(0).clamp(min=.03))
        z=((flat-self.q_mean)/self.q_scale).clamp(-6,6)
        # Deterministic source-only farthest-point prototypes; no validation/test data.
        inds=[int(z.square().sum(1).argmin())];distance=(z-z[inds[0]]).square().sum(1)
        for _ in range(1,self.responses_count):
            inds.append(int(distance.argmax()));distance=torch.minimum(distance,(z-z[inds[-1]]).square().sum(1))
        self.centres.copy_(z[inds]);self.log_precision.fill_(-.7)
    def response(self,q,reference=None):
        z=((q.movedim(1,-1)-self.q_mean)/self.q_scale).clamp(-6,6)
        delta=z.unsqueeze(-2)-self.centres
        precision=F.softplus(self.log_precision)+1e-3
        return torch.exp(-.5*(delta.square()*precision).sum(-1)).movedim(-1,1)
    def constants(self,shape,device,dtype,bounds):
        key=(tuple(shape),str(device),dtype)
        if key not in self._cache:
            axes=[(torch.arange(s,device=device,dtype=dtype)+.5)/s*2-1 for s in shape]
            coord=list(torch.meshgrid(*axes,indexing='ij'))
            if len(shape)==2:coord=[torch.zeros_like(coord[0])]+coord
            basis=torch.stack([torch.ones_like(coord[0]),*coord,*[coord[i]*coord[j] for i in range(3) for j in range(i,3)]])
            b=torch.as_tensor(bounds,device=device);volume=(b[:,:,1]-b[:,:,0]).prod(1)
            midpoint=(b[:,:,1]+b[:,:,0]).to(dtype)/torch.tensor(shape,device=device,dtype=dtype)-1
            halfwidth=(b[:,:,1]-b[:,:,0]).to(dtype)/torch.tensor(shape,device=device,dtype=dtype)
            if len(shape)==2:
                midpoint=torch.cat([torch.zeros_like(midpoint[:,:1]),midpoint],1)
                halfwidth=torch.cat([torch.ones_like(halfwidth[:,:1]),halfwidth],1)
            self._cache[key]=(basis,b,volume,midpoint,halfwidth)
        return self._cache[key]
    def forward(self,q,g):
        if int(np.prod(q.shape[2:]))>131072:return self.tiled(q,g)
        r=self.response(q);shape=q.shape[2:];dim=len(shape)
        basis,bounds,volume,midpoint,halfwidth=self.constants(shape,q.device,q.dtype,g['bounds'])
        weighted=(r[:,:,None]*basis[None,None]).flatten(1,2)
        grid=2**g['depth']
        if all(s%grid==0 for s in shape):
            # Equal-size leaf blocks admit exact pooled sums. Recursively sum
            # children rather than retaining multiple full-volume prefixes.
            kernel=tuple(s//grid for s in shape)
            pool=F.avg_pool2d if dim==2 else F.avg_pool3d
            pooled=pool(weighted,kernel,kernel)
            leaves=g['levels'][-1];lb=bounds[leaves]
            ids=tuple(lb[:,j,0]//kernel[j] for j in range(dim))
            cur=pooled[(slice(None),slice(None),*ids)].transpose(1,2)*volume[leaves][None,:,None]
            totals=[cur]
            for level in reversed(range(g['depth'])):
                cur=cur.reshape(len(q),len(g['levels'][level]),2**dim,-1).sum(2);totals.append(cur)
            total=torch.cat(list(reversed(totals)),1)
        else:
            # General unequal/odd input dimensions use integral sums with the
            # same region registry, rather than changing support definitions.
            prefix=weighted
            for axis in range(2,2+dim):prefix=prefix.cumsum(axis)
            prefix=F.pad(prefix,tuple(v for _ in range(dim) for v in (1,0)))
            summed=0
            for bits in product((0,1),repeat=dim):
                ids=tuple(bounds[:,j,bit] for j,bit in enumerate(bits))
                summed=summed+((-1)**(dim-sum(bits)))*prefix[(slice(None),slice(None),*ids)]
            total=summed.transpose(1,2)
        moments=total.reshape(len(q),len(bounds),self.responses_count,10)
        weight=moments[...,0].clamp(min=1e-6);occupancy=moments[...,0]/volume[None,:,None]
        centre=moments[...,1:4]/weight[...,None]
        covariance=[]
        for index,(i,j) in enumerate(( (i,j) for i in range(3) for j in range(i,3) )):
            covariance.append((moments[...,4+index]/weight-centre[...,i]*centre[...,j])/(halfwidth[None,:,None,i]*halfwidth[None,:,None,j]))
        local=(centre-midpoint[None,:,None,:])/halfwidth[None,:,None,:]
        valid=moments[...,0]>1e-6
        local=torch.where(valid[...,None],local,torch.zeros_like(local))
        cov=torch.stack(covariance,-1);cov=torch.where(valid[...,None],cov,torch.zeros_like(cov))
        return torch.cat([occupancy[...,None],local,cov],-1).flatten(2)

    def tiled(self,q,g,tile_voxels=131072):
        """Exact leaf moment sums in bounded tiles; checkpoint tiles in training.

        Full calibrated measurement maps still reside in memory. This reduces
        radial-response/basis intermediates, not the native image storage itself.
        """
        from torch.utils.checkpoint import checkpoint
        shape=q.shape[2:];dim=len(shape)
        reference=self.reference(q)[1:] if hasattr(self,'reference') else None
        totals=[]
        def tile_sum(tensor,starts,stops):
            axes=[(torch.arange(a,b,device=q.device,dtype=q.dtype)+.5)/shape[j]*2-1 for j,(a,b) in enumerate(zip(starts,stops))]
            coords=list(torch.meshgrid(*axes,indexing='ij'))
            if dim==2:coords=[torch.zeros_like(coords[0])]+coords
            basis=torch.stack([torch.ones_like(coords[0]),*coords,*[coords[i]*coords[j] for i in range(3) for j in range(i,3)]])
            response=self.response(tensor,reference=reference)
            return (response[:,:,None]*basis[None,None]).flatten(3).sum(-1)
        for v in g['levels'][-1]:
            bounds=g['bounds'][v];width=bounds[:,1]-bounds[:,0]
            slab=max(1,int(tile_voxels//max(1,np.prod(width[1:]))));total=None
            for a in range(int(bounds[0,0]),int(bounds[0,1]),slab):
                starts=[a,*bounds[1:,0].tolist()];stops=[min(a+slab,int(bounds[0,1])),*bounds[1:,1].tolist()]
                part=q[(slice(None),slice(None),*(slice(lo,hi) for lo,hi in zip(starts,stops)))]
                # Bind bounds per invocation: recomputation must use the same tile.
                function=lambda tensor,starts=starts,stops=stops:tile_sum(tensor,starts,stops)
                value=checkpoint(function,part,use_reentrant=False) if self.training and torch.is_grad_enabled() else function(part)
                total=value if total is None else total+value
            totals.append(total.flatten(1))
        cur=torch.stack(totals,1);levels=[cur]
        for level in reversed(range(g['depth'])):
            cur=cur.reshape(len(q),len(g['levels'][level]),2**dim,-1).sum(2);levels.append(cur)
        moments=torch.cat(list(reversed(levels)),1).reshape(len(q),len(g['bounds']),self.responses_count,10)
        bounds=torch.as_tensor(g['bounds'],device=q.device);sizes=torch.tensor(shape,device=q.device,dtype=q.dtype)
        volume=(bounds[:,:,1]-bounds[:,:,0]).prod(1)
        midpoint=(bounds[:,:,1]+bounds[:,:,0]).to(q.dtype)/sizes-1;halfwidth=(bounds[:,:,1]-bounds[:,:,0]).to(q.dtype)/sizes
        if dim==2:
            midpoint=torch.cat([torch.zeros_like(midpoint[:,:1]),midpoint],1);halfwidth=torch.cat([torch.ones_like(halfwidth[:,:1]),halfwidth],1)
        weight=moments[...,0].clamp(min=1e-6);centre=moments[...,1:4]/weight[...,None]
        cov=torch.stack([(moments[...,4+k]/weight-centre[...,i]*centre[...,j])/(halfwidth[None,:,None,i]*halfwidth[None,:,None,j]) for k,(i,j) in enumerate(( (i,j) for i in range(3) for j in range(i,3) ))],-1)
        valid=moments[...,0]>1e-6;local=(centre-midpoint[None,:,None,:])/halfwidth[None,:,None,:]
        local=torch.where(valid[...,None],local,torch.zeros_like(local));cov=torch.where(valid[...,None],cov,torch.zeros_like(cov))
        return torch.cat([(moments[...,0]/volume[None,:,None])[...,None],local,cov],-1).flatten(2)
