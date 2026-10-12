"""Experimental native-field review features, source-fitted against simple confidence.

Dictionary coverage and program competition are computational measures, not
proof of healthy reference tissue or validated clinical uncertainty.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .encoder import measurements
from .replay import numpy_program
from .calibration import probabilities

@torch.no_grad()
def native_features(model,x,batch=64,spacing=None):
 model.eval();output=[];b=None
 for start in range(0,len(x),batch):
  inp=x[start:start+batch];phi,g=model.descriptors(inp,spacing);own,pair,cache=model.scored(phi,g)
  actual_logits,actual_routes=model.compose(own,pair,g,return_routes=True)
  logits=actual_logits.cpu().numpy().astype(float);actual_routes=actual_routes.cpu().numpy()
  own=own.cpu().numpy();pair=None if pair is None else pair.cpu().numpy()
  coverage=model.encoder.response(measurements(inp,spacing,mode=model.config.calibration)).max(1).values.flatten(1)
  coverage=(coverage>.3).float().mean(1).cpu().numpy();b=2**g['dim']
  for n in range(len(inp)):
   base,_,_=numpy_program(own[n],None if pair is None else pair[n],g);routes=actual_routes[n];winner=int(logits[n].argmax());selected=[]
   def visit(level,v):
    choice=int(routes[v,winner])
    if level==g['depth'] or choice==0:selected.append(('node',v,float(own[n,v,winner])));return
    pos=int(np.flatnonzero(g['levels'][level]==v)[0]);children=g['levels'][level+1].reshape(-1,b)[pos];used=set()
    matching=g['matchings'][choice-1] if pair is not None else ()
    for edge in matching:
     i,j=g['edges'][edge];used.update([i,j]);pi=int(g['pair_levels'][level][pos,edge]);selected.append(('pair',pi,float(pair[n,pi,winner])))
    for i,child in enumerate(children):
     if i not in used:visit(level+1,int(child))
   visit(0,0);top=max(selected,key=lambda s:abs(s[2]));challenged,_,_=numpy_program(own[n],None if pair is None else pair[n],g,(top[0],top[1]))
   drop=max(0,float(base[winner]-challenged[winner]))/(1+abs(float(base[winner])))
   output.append([float(coverage[n]),len(selected),sum(v[2]<0 for v in selected)/len(selected),drop])
 return np.array(output,float)

def confidence_features(logits,temperature=1.):
 p=probabilities(logits,temperature);sortedp=np.sort(p,axis=1)
 return np.c_[p.max(1),-(p*np.log(np.maximum(p,1e-15))).sum(1),sortedp[:,-1]-sortedp[:,-2]]

def fit_gate(source_logits,source_native,source_labels,temperature=1.,coverage=.9):
 c=confidence_features(source_logits,temperature);x=np.c_[c,source_native];correct=(source_logits.argmax(1)==source_labels).astype(int)
 if len(np.unique(correct))<2:return None,dict(status='confidence_fallback',reason='Source calibration has no correctness variation')
 model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=11));model.fit(x,correct)
 quality=model.predict_proba(x)[:,1];threshold=float(np.quantile(quality,1-coverage))
 return model,dict(status='experimental_evidence_gate',source_only=True,threshold=threshold,target_source_coverage=coverage,features=['calibrated_confidence','entropy','probability_margin','dictionary_coverage','selected_unit_count','negative_unit_fraction','top_rule_drop'],scope='Fit on source validation after model selection; not an independent clinical calibration study')

def evaluate_gate(model,receipt,logits,native,labels,temperature):
 if model is None:return None
 quality=model.predict_proba(np.c_[confidence_features(logits,temperature),native])[:,1]
 accepted=quality>=receipt['threshold'];correct=logits.argmax(1)==labels;order=np.argsort(-quality,kind='stable');curve=[]
 for fraction in np.linspace(.1,1,10):
  n=max(1,int(np.ceil(len(quality)*fraction)));curve.append(dict(coverage=n/len(quality),risk=float(1-correct[order[:n]].mean())))
 return dict(coverage=float(accepted.mean()),accepted_error=None if not accepted.any() else float(1-correct[accepted].mean()),risk_coverage=curve)
