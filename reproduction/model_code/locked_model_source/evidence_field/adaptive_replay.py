"""NumPy/SciPy independent reader for EF adaptive-program certificates.

No Torch dependency. External parameter identity and original image bytes are
required. Checks numeric provenance, cover, chosen expression and optimum;
does not assert clinical sufficiency or acquisition authenticity.
"""
import hashlib
from itertools import product, combinations
import numpy as np
from .structural_replay import structural_features


def independent_phi(certificate,prepared):
    c=certificate;g=c['geometry'];x=np.ascontiguousarray(prepared)
    if hashlib.sha256(x.tobytes()).hexdigest()!=c['input_sha256']:raise ValueError('Input hash mismatch')
    if list(x.shape[1:])!=list(g['shape']):raise ValueError('Shape mismatch')
    state={k.removeprefix('encoder.'):v for k,v in c['parameters'].items() if k.startswith('encoder.')}
    # Reuse the independent direct-region sums, then supply occurrence depths.
    dummy=dict(input_sha256=c['input_sha256'],encoder_schema='evidence-field-characteristic-sums/0.2',
        config=c['config'],shape=list(g['shape']),spacing=c['spacing'],encoder_state=state,
        support_bounds=g['bounds'],parents=g['parent'],image_reference=dict(mean=np.zeros(6),scale=np.zeros(6)))
    if c['config']['calibration']=='spectral_optical_density/1.0':
        from .medical_input_replay import spectral_measurements_numpy
        x=spectral_measurements_numpy(x,c['spacing'])
        dummy['prepared_input_contract']='six-coordinate-structural-diagnostic/1.0'
        dummy['input_sha256']=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    phi,_=structural_features(dummy,x)
    width=c['config']['responses']*(10+2*c['config']['frequencies']);f=phi[:,:width]
    parent=np.array(g['parent']);depth=np.zeros(len(f))
    for h,level in enumerate(g['levels']):depth[level]=h/g['depth']
    return np.c_[f,f-f[parent],f-f[0],g['mass'],depth,np.ones(len(f))].clip(-6,6)


def verify_geometry(g,config):
    dim=g['dim'];b=2**dim;bounds=np.array(g['bounds']);shape=np.array(g['shape'])
    if dim not in (2,3) or len(shape)!=dim or config['alternatives']!=g['alternatives']:raise ValueError('Geometry header')
    if not np.array_equal(bounds[0],np.c_[np.zeros(dim,int),shape]):raise ValueError('Root bounds')
    mass=np.prod(bounds[:,:,1]-bounds[:,:,0],1)/np.prod(shape)
    if np.any(bounds[:,:,0]<0) or np.any(bounds[:,:,1]>shape) or np.any(bounds[:,:,1]<=bounds[:,:,0]):raise ValueError('Bounds')
    if not np.allclose(mass,g['mass'],atol=1e-12):raise ValueError('Mass')
    expected_parent=np.zeros(len(bounds),int);seen_children=[]
    for h,ps in enumerate(g['production_levels']):
        if len(ps)!=len(g['levels'][h])*g['alternatives']:raise ValueError('Production count')
        for offset,pi in enumerate(ps):
            p=g['productions'][pi];v=g['levels'][h][offset//g['alternatives']];alt=offset%g['alternatives']
            if p['parent']!=v or p['alternative']!=alt or p['children']!=g['child_ids'][pi]:raise ValueError('Production identity')
            if g['production_parent'][pi]!=v:raise ValueError('Production parent vector')
            if len(p['children'])!=b:raise ValueError('Children count')
            cuts=[]
            for j,(lo,hi) in enumerate(bounds[v]):
                ratio=.5 if alt==0 else (.4 if j%2==0 else .6)
                cuts.append(min(hi-1,max(lo+1,lo+int((hi-lo)*ratio))))
            for child,bits in zip(p['children'],product((0,1),repeat=dim)):
                expected=np.array([(lo,m) if bit==0 else (m,hi) for (lo,hi),m,bit in zip(bounds[v],cuts,bits)])
                if not np.array_equal(bounds[child],expected):raise ValueError('Partition bounds')
                expected_parent[child]=v;seen_children.append(child)
            if abs(mass[p['children']].sum()-mass[v])>1e-12:raise ValueError('Partition mass')
    if sorted(seen_children)!=list(range(1,len(bounds))) or not np.array_equal(expected_parent,g['parent']):raise ValueError('Occurrence tree')
    if sorted(sum(g['levels'],[]))!=list(range(len(bounds))):raise ValueError('Levels')
    positions=list(product((0,1),repeat=dim));groups=[]
    adjacent=lambda i,j:sum(a!=c for a,c in zip(positions[i],positions[j]))==1
    for order in range(2,min(config['max_order'],b)+1):
        for group in combinations(range(b),order):
            seen={group[0]}
            while True:
                extra={i for i in group if any(adjacent(i,j) for j in seen)}
                if extra<=seen:break
                seen|=extra
            if len(seen)==order:groups.append(list(group))
    if groups!=[list(a) for a in g['groups']]:raise ValueError('Missing or extra group')
    # Independent subset enumeration checks completeness of every packing.
    def pack(remaining):
        if not remaining:return [()]
        i=min(remaining);rows=pack(remaining-{i})
        for gi,group in enumerate(groups):
            if i in group and set(group)<=remaining:rows+= [(gi,*p) for p in pack(remaining-set(group))]
        return rows
    all_packings=pack(set(range(b)))
    if [tuple(p) for p in g['packings']]!=all_packings:raise ValueError('Incomplete packings')
    for qi,packing in enumerate(all_packings):
        expected_inc=np.zeros(len(groups));unm=np.ones(b)
        for gi in packing:expected_inc[gi]=1;unm[groups[gi]]=0
        if not np.array_equal(expected_inc,g['incidence'][qi]) or not np.array_equal(unm,g['unmatched'][qi]):raise ValueError('Packing matrices')
    return mass


def verify_adaptive(certificate,prepared,trusted_parameters,atol=1e-3,trusted_config=None,trusted_model_id=None):
    c=certificate
    if c['schema']!='ef-adaptive-program-certificate/0.3':raise ValueError('Schema')
    config=c['config']
    if trusted_config is not None and config!=trusted_config:raise ValueError('External configuration identity')
    if trusted_model_id is not None and c['model_id']!=trusted_model_id:raise ValueError('External model identity')
    if config['signature_normalization']!='response_mass' or config['readout']!='cover' or not config['reference_conditioning'] or config['unit_penalty']!=0:
        raise ValueError('Unsupported declared core configuration')
    if len(c['classes'])!=config['classes'] or len(c['logits'])!=config['classes']:raise ValueError('Incomplete classes')
    if set(trusted_parameters)!=set(c['parameters']):raise ValueError('Parameter registry')
    for key,value in trusted_parameters.items():
        if not np.array_equal(np.asarray(value),np.asarray(c['parameters'][key])):raise ValueError('Parameter identity')
        if not np.isfinite(np.asarray(value)).all():raise ValueError('Nonfinite trusted parameter')
    g=c['geometry'];config=c['config']
    if g['depth']!=config['depth'] or g['alternatives']!=config['alternatives']:raise ValueError('Configuration/geometry mismatch')
    mass=verify_geometry(g,config)
    phi=independent_phi(c,prepared);declared=np.array(c['phi']);error=float(abs(phi-declared).max())
    if not np.isfinite(phi).all() or not np.isfinite(declared).all() or not np.isfinite(c['logits']).all():raise ValueError('Nonfinite descriptor or logit')
    if error>atol:raise ArithmeticError('Descriptor replay mismatch')
    # Independently verified declared descriptors are used for strict expression
    # replay; native float32 prefix rounding is covered by the separate tolerance.
    a={k:np.array(v,float) for k,v in trusted_parameters.items()};p=declared
    own=np.einsum('vd,kmd->vkm',p,a['theta']).max(-1)*mass[:,None]
    scores=np.zeros((len(g['productions']),len(g['groups']),config['classes']))
    group_type_scores={}
    for pi,production in enumerate(g['productions']):
        children=np.array(production['children']);parent=p[production['parent']]
        for gi,group in enumerate(g['groups']):
            ids=children[list(group)];weights=mass[ids];gm=weights.sum();members=p[ids];mean=(members*weights[:,None]).sum(0)/gm
            spread=(abs(members[:,:-1]-mean[None,:-1])*weights[:,None]).sum(0)/gm
            context=parent if config['cross_scale'] else np.zeros_like(parent)
            features=np.r_[mean,spread,context];oi=len(group)-2
            linear=np.einsum('d,ktd->kt',features,a['group_linear'][oi])
            projection=np.tanh(np.einsum('gd,ktd->gkt',members[:,:-1],a['member_projection'][oi])).prod(0)
            modulation=1+np.tanh(np.einsum('d,ktd->kt',context[:-1],a['parent_projection'][oi]))
            typed=own[ids].sum(0)[:,None]+gm*(linear+a['gamma'][oi]*projection*modulation)
            group_type_scores[pi,gi]=typed
            scores[pi,gi]=typed.max(-1)
    values=own.copy();bounds_skips=0;bounds_checked=0
    for h in reversed(range(g['depth'])):
        for vi,v in enumerate(g['levels'][h]):
            options=[own[v]];upper=[]
            for alt in range(g['alternatives']):
                pi=g['production_levels'][h][vi*g['alternatives']+alt];children=g['child_ids'][pi]
                B=values[children].copy()
                for gi,group in enumerate(g['groups']):
                    for member in group:B[member]=np.maximum(B[member],scores[pi,gi]/len(group))
                upper.append(B.sum(0))
                for qi in range(len(g['packings'])):
                    candidate=(values[children]*np.array(g['unmatched'][qi])[:,None]).sum(0)
                    if g['packings'][qi]:candidate+=scores[pi,g['packings'][qi]].sum(0)
                    if np.any(candidate>B.sum(0)+1e-9):raise ArithmeticError('Admissible bound failed')
                    options.append(candidate);bounds_checked+=config['classes']
            bounds_skips+=int((own[v]>np.array(upper).max(0)+1e-9).sum())
            values[v]=np.array(options).max(0)
    optimum=values[0]+a['bias'];replay=[];coverage=[]
    for k,row in enumerate(c['classes']):
        if row['class_index']!=k or abs(row['logit']-c['logits'][k])>1e-12:raise ValueError('Class order/logit')
        image=np.zeros(tuple(g['shape']),int);total=float(a['bias'][k])
        for unit in row['units']:
            nodes=unit['nodes']
            if not nodes or any(not isinstance(v,int) or not 0<=v<len(g['mass']) for v in nodes):raise ValueError('Unit nodes')
            if not np.isfinite(unit['score']):raise ValueError('Unit score')
            for v in nodes:image[tuple(slice(lo,hi) for lo,hi in g['bounds'][v])]+=1
            if unit['kind']=='node':
                v=nodes[0];t=unit['type'];selected=mass[v]*float(a['theta'][k,t]@p[v])
                if len(nodes)!=1 or not isinstance(t,int) or not 0<=t<config['node_types']:raise ValueError('Node type/support')
                if abs(selected-own[v,k])>1e-7:raise ArithmeticError('Non-optimal node type')
            elif unit['kind']=='group':
                pi=unit['production'];gi=unit['group'];selected=scores[pi,gi,k]
                if not isinstance(unit['type'],int) or not 0<=unit['type']<config['pair_types']:raise ValueError('Group type')
                if nodes!=np.array(g['child_ids'][pi])[list(g['groups'][gi])].tolist():raise ValueError('Group support')
                if unit['parent']!=g['production_parent'][pi]:raise ValueError('Group parent')
                if abs(group_type_scores[pi,gi][k,unit['type']]-selected)>1e-7:raise ArithmeticError('Non-optimal group type')
                if abs(float(own[nodes,k].sum())-unit['anchors'])>1e-5:raise ArithmeticError('Anchor mismatch')
                if abs(selected-unit['anchors']-unit['correction'])>1e-5:raise ArithmeticError('Correction mismatch')
            else:raise ValueError('Unit kind')
            if abs(selected-unit['score'])>1e-5:raise ArithmeticError('Selected expression mismatch')
            total+=selected
        if not np.all(image==1):raise ArithmeticError('Cover failed')
        # Spatial coverage alone is insufficient: verify that the selected atoms
        # belong to one admissible recursive production, rather than mixing
        # coincident supports from incompatible partition occurrences.
        node_atoms={};group_atoms={}
        for atom_index,unit in enumerate(row['units']):
            if unit['kind']=='node':node_atoms[unit['nodes'][0]]=atom_index
            else:group_atoms[(unit['production'],unit['group'])]=atom_index
        def parse(v):
            if v in node_atoms:return {node_atoms[v]}
            if v in g['levels'][-1]:return None
            h=next(h for h,level in enumerate(g['levels']) if v in level)
            vi=g['levels'][h].index(v)
            for alt in range(g['alternatives']):
                pi=g['production_levels'][h][vi*g['alternatives']+alt]
                selected={gi for p,gi in group_atoms if p==pi}
                for qi,packing in enumerate(g['packings']):
                    if set(packing)!=selected:continue
                    consumed={group_atoms[pi,gi] for gi in packing};valid=True
                    for ci in np.flatnonzero(g['unmatched'][qi]):
                        subtree=parse(g['child_ids'][pi][ci])
                        if subtree is None:valid=False;break
                        consumed|=subtree
                    if valid:return consumed
            return None
        if parse(0)!=set(range(len(row['units']))):raise ArithmeticError('Inadmissible selected grammar')
        replay.append(total);coverage.append(int(image.size))
    score_error=float(abs(np.array(replay)-np.array(c['logits'])).max())
    optimum_error=float(abs(optimum-np.array(c['logits'])).max())
    if max(score_error,optimum_error)>1e-4:raise ArithmeticError('Score/optimum mismatch')
    return dict(passed=True,descriptor_error=error,score_error=score_error,optimum_error=optimum_error,
        cover_voxels=coverage,admissible_bound_candidates=bounds_checked,dominated_expansions=bounds_skips,
        scope='Real prepared-image numeric provenance, exact cover and registered optimum')
