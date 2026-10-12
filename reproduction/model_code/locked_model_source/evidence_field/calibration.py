"""Source-fitted temperature and coverage controls; no shift guarantee."""
import hashlib,json
import numpy as np
from scipy.optimize import minimize_scalar

def probabilities(logits,temperature=1.):
 z=np.asarray(logits,dtype=float)/temperature;z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)

def fit_calibration(logits,labels,model_sha256,coverage=.9):
 y=np.asarray(labels,dtype=int)
 def loss(log_t):
  p=probabilities(logits,np.exp(log_t));return -np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()
 result=minimize_scalar(loss,bounds=(np.log(.05),np.log(10)),method='bounded')
 t=float(np.exp(result.x));p=probabilities(logits,t);confidence=p.max(1)
 # The lower empirical quantile, including on the archived NumPy 1.21 runtime.
 threshold=float(np.sort(confidence)[int(np.floor((len(confidence)-1)*(1-coverage)))])
 receipt=dict(schema='ef-calibration/1.0',model_sha256=model_sha256,temperature=t,confidence_threshold=threshold,target_source_coverage=coverage,source_samples=len(y),source_logit_sha256=hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),source_only=True,guarantee='Empirical source calibration; hospital-shift coverage/correctness not guaranteed')
 receipt['version_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest();return receipt

def risk_coverage(logits,labels,calibration):
 p=probabilities(logits,calibration['temperature']);confidence=p.max(1);correct=p.argmax(1)==labels;order=np.argsort(-confidence,kind='stable')
 coverages=np.linspace(.1,1,10);curve=[]
 for fraction in coverages:
  n=max(1,int(np.ceil(len(p)*fraction)));curve.append(dict(coverage=n/len(p),risk=float(1-correct[order[:n]].mean())))
 accepted=confidence>=calibration['confidence_threshold']
 return dict(actual_coverage=float(accepted.mean()),accepted_error=None if not accepted.any() else float(1-correct[accepted].mean()),risk_coverage=curve)
