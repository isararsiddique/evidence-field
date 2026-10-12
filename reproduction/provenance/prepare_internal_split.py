"""Fixed patient-only subdivision of the six source validation patients."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 old=ROOT/'prepared_v01';out=ROOT/'prepared';out.mkdir(exist_ok=True)
 audit=json.loads((old/'preparation_audit.json').read_text());a=np.load(old/'source_validation.npz');patients=sorted(set(a['patient'].tolist()))
 val=patients[::2];test=patients[1::2];assert len(val)==len(test)==3
 for name in ['train','hospital_1','hospital_2']:shutil.copy2(old/(name+'.npz'),out/(name+'.npz'))
 for name,ids in [('source_validation',val),('source_test',test)]:
  mask=np.isin(a['patient'],ids);np.savez_compressed(out/(name+'.npz'),**{k:a[k][mask] for k in a.files})
  assert len(np.unique(a['labels'][mask]))==2
 rows=list(csv.DictReader((old/'sample_manifest.csv').open()))
 for row in rows:
  if row['pool']=='source_validation' and int(row['patient']) in test:row['pool']='source_test'
 with (out/'sample_manifest.csv').open('w') as s:
  writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 counts={}
 for pool in ['train','source_validation','source_test','hospital_1','hospital_2']:
  data=np.load(out/(pool+'.npz'));counts[pool]=dict(images=len(data['labels']),patients=sorted(set(data['patient'].tolist())),label_counts=np.bincount(data['labels'],minlength=2).tolist(),sha256=sha(out/(pool+'.npz')))
 sets=[set(v['patients']) for v in counts.values()];assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i+1,len(sets)))
 hashes=[{r['decoded_sha256'] for r in rows if r['pool']==pool} for pool in counts];assert all(not hashes[i]&hashes[j] for i in range(len(hashes)) for j in range(i+1,len(hashes)))
 audit.update(version='camelyon-native-patient-hospital/1.1',counts=counts,source_validation_patients=val,source_test_patients=test,
  patient_split_rule='Sort six existing source-validation patient IDs; alternating IDs to validation and internal test, labels unused for assignment',
  previous_preparation_audit_sha256=sha(old/'preparation_audit.json'),sample_manifest_sha256=sha(out/'sample_manifest.csv'))
 (out/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('SOURCE INTERNAL PATIENT SPLIT LOCKED',counts,flush=True)
if __name__=='__main__':run()
