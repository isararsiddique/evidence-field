"""Replayable probability ensembles, separate from additive mean-logit programs."""
import hashlib,json
import numpy as np
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def build_probability_mixture(certificates,temperatures=None,weights=None):
    n=len(certificates)
    if n<1:raise ValueError('Member certificates required')
    temperatures=np.ones(n) if temperatures is None else np.asarray(temperatures,float)
    weights=np.full(n,1/n) if weights is None else np.asarray(weights,float)
    if temperatures.shape!=(n,) or weights.shape!=(n,) or not np.isfinite(temperatures).all() or not np.isfinite(weights).all():raise ValueError('Finite member settings required')
    if (temperatures<=0).any() or (weights<0).any() or abs(weights.sum()-1)>1e-12:raise ValueError('Positive temperatures and normalized nonnegative weights required')
    if len({c['input_sha256'] for c in certificates})!=1:raise ValueError('Mixture members must share the same prepared input')
    logits=np.asarray([c['logits'] for c in certificates],float)
    if logits.ndim!=2 or not np.isfinite(logits).all():raise ValueError('Finite compatible class logits required')
    scaled=logits/temperatures[:,None];scaled-=scaled.max(1,keepdims=True)
    member=np.exp(scaled);member/=member.sum(1,keepdims=True);prob=weights@member
    entropy=lambda p:float(-(p*np.log(np.clip(p,1e-15,1))).sum())
    disagreement=entropy(prob)-sum(w*entropy(p) for w,p in zip(weights,member))
    if disagreement<-1e-12:raise ValueError('Invalid mixture entropy')
    return dict(schema='ef-probability-mixture/1.0',input_sha256=certificates[0]['input_sha256'],
        members=[dict(model_id=c['model_id'],program_digest=digest(c),temperature=float(t),weight=float(w)) for c,t,w in zip(certificates,temperatures,weights)],
        probabilities=prob.tolist(),predicted_class=int(prob.argmax()),predictive_entropy=entropy(prob),disagreement=max(0.,disagreement),
        scope='Member signed logits followed by declared temperature transforms and probability averaging; distinct from a mean-logit additive program')
def verify_probability_mixture(mixture,certificates,trusted_model_ids):
    if [c['model_id'] for c in certificates]!=list(trusted_model_ids):raise ValueError('Externally trusted member identity mismatch')
    rebuilt=build_probability_mixture(certificates,[r['temperature'] for r in mixture['members']],[r['weight'] for r in mixture['members']])
    if rebuilt!=mixture:raise ValueError('Probability mixture was altered')
    return dict(passed=True,member_count=len(certificates),scope='Mixture arithmetic and identity; each member needs its own complete-program audit')
