"""Compare a record to an independently supplied approved model object.

A self-reported model hash is insufficient. The caller must load an approved,
trusted checkpoint and verify its file hash against their own model registry.
"""
from dataclasses import asdict
import numpy as np
from .replay import check_cover

def validate_trusted_model(record,model):
 if record['model_sha256']!=model.fingerprint():raise ValueError('Record does not belong to supplied trusted model')
 if record['config']!=asdict(model.config):raise ValueError('Model configuration differs')
 state={k:v.detach().cpu().numpy() for k,v in model.encoder.state_dict().items()}
 if set(state)!=set(record['encoder_state']):raise ValueError('Encoder registry differs')
 for key,value in state.items():
  if not np.array_equal(value,np.array(record['encoder_state'][key])):raise ValueError('Recorded encoder state differs from trusted model')
 g=model.layout(record['shape'])
 if not np.array_equal(g['bounds'],record['support_bounds']) or not np.array_equal(g['parent'],record['parents']):raise ValueError('Geometry differs from trusted model')
 if len(record['classes'])!=model.config.classes:raise ValueError('Class registry differs')
 for key,entry in record['coefficient_registry'].items():
  kind,k,m=key.split(':');k=int(k);m=int(m)
  if not 0<=k<model.config.classes or m<0:raise ValueError('Coefficient index outside approved model')
  if kind=='node':expected=model.theta[k,m].detach().cpu().numpy()
  elif kind=='pair':
   expected=model.pair[k,m].detach().cpu().numpy()
   if not np.array_equal(model.projection[k,m].detach().cpu().numpy(),entry['projection']):raise ValueError('Pair projection differs')
   gamma=float(model.gamma[k,m].detach().cpu()) if model.config.bilinear else 0.
   if gamma!=entry['gamma']:raise ValueError('Pair product weight differs')
  else:raise ValueError('Invalid coefficient key')
  if not np.array_equal(expected,entry['weights']):raise ValueError('Coefficient differs from trusted model')
 for k,entry in enumerate(record['classes']):
  if entry['index']!=k or float(model.bias[k].detach().cpu())!=entry['bias']:raise ValueError('Class/bias differs')
  def check_unit(unit,kind):
   if unit['coefficient']!=f"{kind}:{k}:{unit['type']}":raise ValueError('Evidence unit uses a coefficient from another class/type')
  for unit in entry['units']:
   check_unit(unit,unit['kind'])
   if unit['kind']=='pair':
    for anchor in unit['anchors']:check_unit(anchor,'node')
 check_cover(record)
 return dict(passed=True,model_fingerprint=record['model_sha256'],encoder_and_selected_coefficients_checked=True,scope='Compared to supplied trusted model; trust in its approved checkpoint registry remains external')
