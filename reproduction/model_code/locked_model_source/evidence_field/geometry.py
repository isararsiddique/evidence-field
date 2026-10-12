"""Exact dyadic support registry and face-adjacent sibling matchings."""
from itertools import product
import numpy as np

def matchings(dim):
    bits=list(product((0,1),repeat=dim))
    edges=[(i,j) for i in range(len(bits)) for j in range(i+1,len(bits))
           if sum(a!=b for a,b in zip(bits[i],bits[j]))==1]
    out=[]
    def visit(left,chosen):
        if not left:out.append(tuple(chosen));return
        i=min(left);rest=left-{i};visit(rest,chosen)
        for e,(a,b) in enumerate(edges):
            j=b if a==i else a if b==i else None
            if j is not None and j in rest:visit(rest-{j},chosen+[e])
    visit(set(range(len(bits))),[])
    inc=np.zeros((len(out),len(edges)),np.float32)
    un=np.ones((len(out),len(bits)),np.float32)
    for q,choice in enumerate(out):
        for e in choice:inc[q,e]=1;un[q,list(edges[e])]=0
    return edges,out,inc,un

def geometry(shape,depth=2):
    shape=tuple(int(x) for x in shape);dim=len(shape)
    if dim not in (2,3) or min(shape)<2**depth:raise ValueError('2D/3D input must fit the declared dyadic depth')
    bounds=[tuple((0,s) for s in shape)];parent=[-1];levels=[[0]]
    for level in range(depth):
        ids=[]
        for v in levels[-1]:
            spans=bounds[v]
            for bits in product((0,1),repeat=dim):
                child=tuple((a,(a+b)//2) if bit==0 else ((a+b)//2,b) for (a,b),bit in zip(spans,bits))
                ids.append(len(bounds));bounds.append(child);parent.append(v)
        levels.append(ids)
    bounds=np.array(bounds,np.int64);mass=np.prod(bounds[:,:,1]-bounds[:,:,0],axis=1)/np.prod(shape)
    edges,choices,inc,un=matchings(dim);first=[];second=[];pair_levels=[]
    for ids,children in zip(levels[:-1],levels[1:]):
        start=len(first)
        for row in np.array(children).reshape(len(ids),2**dim):
            for i,j in edges:first.append(row[i]);second.append(row[j])
        pair_levels.append(np.arange(start,len(first)).reshape(len(ids),len(edges)))
    return dict(shape=shape,dim=dim,depth=depth,bounds=bounds,parent=np.array(parent),mass=mass,
                levels=[np.array(x) for x in levels],first=np.array(first),second=np.array(second),
                pair_levels=pair_levels,edges=edges,matchings=choices,incidence=inc,unmatched=un)
