"""Optional versioned deterministic verification. Never overwrites the model.

Facts and a genuinely validated evidence source must be supplied by the user or
a validated measurement pipeline. Latent image-field names are not medical facts.
"""
import hashlib,json

def verify_candidate(candidate,facts,evidence_source=None):
 base=dict(original_candidate=candidate,prediction_overridden=False)
 if evidence_source is None:return dict(base,status='unresolved',reason='No applicable external evidence source supplied',checks=[])
 if not evidence_source.get('validated') or not evidence_source.get('version'):
  return dict(base,status='unresolved',reason='Source validation/version missing',checks=[])
 if evidence_source.get('stale') or evidence_source.get('contradictory'):
  return dict(base,status='unresolved',reason='Evidence source stale or internally contradictory',checks=[])
 checks=[]
 for rule in evidence_source.get('rules',[]):
  if candidate not in rule.get('applies_to',[candidate]):continue
  field=rule['field']
  if field not in facts:checks.append(dict(rule=rule['id'],status='insufficient-evidence',reason='Required validated measurement missing',field=field));continue
  value=facts[field];allowed=rule.get('allowed');bounds=rule.get('range')
  if allowed is not None:passed=value in allowed
  elif bounds is not None:passed=bounds[0]<=value<=bounds[1]
  else:checks.append(dict(rule=rule['id'],status='unresolved',reason='Unsupported rule operation'));continue
  checks.append(dict(rule=rule['id'],status='supported' if passed else 'contradicted',field=field,observed=value,operation='set membership' if allowed is not None else 'closed interval'))
 statuses={c['status'] for c in checks}
 status='contradicted' if 'contradicted' in statuses else 'insufficient-evidence' if 'insufficient-evidence' in statuses else 'supported' if statuses=={'supported'} else 'unresolved'
 return dict(base,status=status,checks=checks,source=evidence_source.get('source'),version=evidence_source['version'],source_sha256=hashlib.sha256(json.dumps(evidence_source,sort_keys=True).encode()).hexdigest(),reason='Model output preserved; result concerns supplied rules, not universal clinical correctness')
