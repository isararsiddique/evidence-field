"""Record construction and independent arithmetic replay. Replay imports no torch.

Record scoring uses stored descriptors and coefficient registries. This is not
encoder replay, a causal explanation, or validated lesion identification.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np

REPLAY_ATOL=1e-4

def numpy_program(own,pair,g,forbidden=None):
    own=np.array(own,copy=True);pair=None if pair is None else np.array(pair,copy=True)
    if forbidden is not None:
        kind,index=forbidden
        if kind=='node':own[index]=-np.inf
        elif kind=='pair':pair[index]=-np.inf
        else:raise ValueError('Rule exclusion must be node or pair')
    v=own.copy();routes=np.zeros_like(v,dtype=np.int64);b=2**g['dim']
    for level in reversed(range(len(g['levels'])-1)):
        for pos,node in enumerate(g['levels'][level]):
            children=g['levels'][level+1].reshape(-1,b)[pos]
            options=[own[node]]
            for matching in (g['matchings'] if pair is not None else [()]):
                used=set();total=np.zeros_like(own[node])
                for edge in matching:
                    i,j=g['edges'][edge];used.update([i,j]);total+=pair[g['pair_levels'][level][pos,edge]]
                for i,child in enumerate(children):
                    if i not in used:total+=v[child]
                options.append(total)
            candidates=np.stack(options);routes[node]=candidates.argmax(0);v[node]=candidates.max(0)
    return v[0],routes,v

def build_records(model,cache,logits,inputs,spacing,temperature,input_ids,model_id):
    from dataclasses import asdict
    g=cache['g'];phi=cache['phi'];theta=model.theta.detach().cpu().numpy()
    paircoef=None if model.pair is None else model.pair.detach().cpu().numpy()
    projections=None if model.projection is None else model.projection.detach().cpu().numpy()
    gamma=None if model.gamma is None else model.gamma.detach().cpu().numpy()
    mass=g['mass'];records=[];b=2**g['dim']
    for n in range(len(inputs)):
        own=cache['own'][n];pairs=None if cache['pair'] is None else cache['pair'][n]
        routes=cache['routes'][n]
        feature_registry={};coefficient_registry={};classes=[]
        def node_term(k,v):
            m=int(cache['node_types'][n,v,k]);key=f'node:{k}:{m}'
            coefficient_registry[key]=dict(weights=theta[k,m].tolist())
            feature_registry[str(v)]=phi[n,v].tolist()
            return dict(kind='node',node=int(v),type=m,mass=float(mass[v]),coefficient=key,score=float(own[v,k]))
        for k in range(model.config.classes):
            units=[];decisions=[]
            def visit(level,v):
                route=int(routes[v,k]);decisions.append(dict(node=int(v),choice=route))
                if level==g['depth'] or route==0:units.append(node_term(k,v));return
                pos=int(np.flatnonzero(g['levels'][level]==v)[0]);children=g['levels'][level+1].reshape(-1,b)[pos]
                matching=g['matchings'][route-1] if pairs is not None else ();blocked=set()
                for edge in matching:
                    i,j=g['edges'][edge];blocked.update([i,j]);pi=int(g['pair_levels'][level][pos,edge]);m=int(cache['pair_types'][n,pi,k])
                    key=f'pair:{k}:{m}'
                    coefficient_registry[key]=dict(weights=paircoef[k,m].tolist(),projection=projections[k,m].tolist(),gamma=float(gamma[k,m]) if model.config.bilinear else 0.)
                    anchors=[node_term(k,int(children[i])),node_term(k,int(children[j]))]
                    units.append(dict(kind='pair',pair_id=pi,nodes=[int(children[i]),int(children[j])],type=m,mass=float(mass[children[i]]+mass[children[j]]),coefficient=key,anchors=anchors,score=float(pairs[pi,k])))
                for i,child in enumerate(children):
                    if i not in blocked:visit(level+1,int(child))
            visit(0,0)
            classes.append(dict(index=k,bias=float(model.bias[k].detach().cpu()),units=units,routes=decisions,original_logit=float(logits[n,k])))
        z=logits[n].astype(float);p=np.exp(z-z.max());p/=p.sum();pc=np.exp(z/temperature-(z/temperature).max());pc/=pc.sum()
        used=set(int(v) for v in feature_registry);refs={int(g['parent'][v]) for v in used if g['parent'][v]>=0}|{0}
        code_root=Path(__file__).parent
        record=dict(schema='evidence-field-signed-cover/1.0',model_sha256=model_id,input_sha256=hashlib.sha256(np.ascontiguousarray(inputs[n]).tobytes()).hexdigest(),input_hash_scope='Prepared input tensor bytes; native source identifier is separate',input_id=None if input_ids is None else str(input_ids[n]),implementation_sha256={name:hashlib.sha256((code_root/name).read_bytes()).hexdigest() for name in ['encoder.py','model.py','geometry.py','replay.py','encoder_replay.py','trust.py']},config=asdict(model.config),shape=list(g['shape']),spacing=None if spacing is None else list(spacing),units='Adapter-defined intensity; derivatives use declared spacing if present; region moments use normalized coordinates',support_bounds=g['bounds'].tolist(),parents=g['parent'].tolist(),feature_registry=feature_registry,coefficient_registry=coefficient_registry,reference_dependencies=sorted(refs),classes=classes,probabilities=p.tolist(),calibration=dict(temperature=float(temperature),probabilities=pc.tolist()),interpretation='Selected computation and shared references; not causal or clinically labelled anatomy')
        # Verify before export instead of silently substituting replayed logits.
        result=replay(record);record['replay_max_absolute_error']=float(np.max(abs(result['logits']-z)))
        record['encoder_state']=cache['encoder_state']
        if model.config.reference_conditioning:
            record['image_reference']=dict(mean=cache['image_reference_mean'][n].tolist(),scale=cache['image_reference_scale'][n].tolist(),assumed_healthy=False)
            record['implementation_sha256']['adaptive_encoder.py']=hashlib.sha256((code_root/'adaptive_encoder.py').read_bytes()).hexdigest()
        if record['replay_max_absolute_error']>REPLAY_ATOL:raise ArithmeticError(record['replay_max_absolute_error'])
        records.append(record)
    return records

def replay(record):
    if record['schema']!='evidence-field-signed-cover/1.0':raise ValueError('Unsupported record version')
    features=record['feature_registry'];coefficients=record['coefficient_registry']
    def dot(weights,values):
        if len(weights)!=len(values):raise ValueError('Coefficient/feature length mismatch')
        if not all(math.isfinite(float(v)) for v in [*weights,*values]):raise ValueError('Nonfinite replay coordinates')
        return math.fsum(float(a)*float(b) for a,b in zip(weights,values))
    def node_score(unit):
        return unit['mass']*dot(coefficients[unit['coefficient']]['weights'],features[str(unit['node'])])
    def unit_score(unit):
        if unit['kind']=='node':return node_score(unit)
        if unit['kind']!='pair':raise ValueError('Unknown evidence unit')
        a,b=[np.array(features[str(v)],dtype=float) for v in unit['nodes']]
        ma,mb=[u['mass'] for u in unit['anchors']]
        chi=np.r_[(a*ma+b*mb)/(ma+mb),.5*abs(a[:-1]-b[:-1])]
        c=coefficients[unit['coefficient']]
        linear=dot(c['weights'],chi)
        pa=dot(c['projection'],a[:-1]);pb=dot(c['projection'],b[:-1])
        correction=unit['mass']*(linear+c['gamma']*pa*pb)
        return math.fsum([*(node_score(u) for u in unit['anchors']),correction])
    z=np.array([math.fsum([c['bias'],*(unit_score(u) for u in c['units'])]) for c in record['classes']])
    p=np.exp(z-z.max());p/=p.sum()
    t=float(record['calibration']['temperature'])
    if not math.isfinite(t) or t<=0:raise ValueError('Invalid calibration temperature')
    cal=np.exp(z/t-(z/t).max());cal/=cal.sum()
    return dict(logits=z,probabilities=p,calibrated_probabilities=cal)

def check_cover(record):
    shape=tuple(record['shape']);bounds=np.array(record['support_bounds']);counts=[]
    if bounds.shape[1:]!=(len(shape),2) or np.any(bounds[:,:,0]<0) or np.any(bounds[:,:,1]>np.array(shape)[None]) or np.any(bounds[:,:,0]>=bounds[:,:,1]):raise ValueError('Invalid support registry')
    for c in record['classes']:
        grid=np.zeros(shape,np.int16)
        for u in c['units']:
            for v in ([u['node']] if u['kind']=='node' else u['nodes']):
                grid[tuple(slice(int(a),int(b)) for a,b in bounds[v])]+=1
            expected=sum(np.prod(bounds[v,:,1]-bounds[v,:,0])/np.prod(shape) for v in ([u['node']] if u['kind']=='node' else u['nodes']))
            if abs(u['mass']-expected)>1e-7:raise ValueError('Evidence mass differs from spatial support')
        if not np.all(grid==1):raise ValueError('Evidence supports do not form an exact cover')
        counts.append(len(c['units']))
    return counts

def rule_challenges(cache,n=0):
    own=cache['own'][n];pair=None if cache['pair'] is None else cache['pair'][n];g=cache['g']
    base,routes,_=numpy_program(own,pair,g);results=[]
    for kind,count in [('node',len(own)),('pair',0 if pair is None else len(pair))]:
        for index in range(count):
            challenged,_,_=numpy_program(own,pair,g,(kind,index))
            if np.any(challenged>base+1e-5):raise ArithmeticError('Restricted program improved over its superset')
            results.append(dict(kind=kind,index=index,drop=(base-challenged).tolist(),feasible=np.isfinite(challenged).tolist()))
    return results
