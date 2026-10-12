"""Frozen patient/hospital protocol from the complete public CAMELYON17 mirror."""
from pathlib import Path
import csv,hashlib,io,json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parent
POOLS=['train','source_validation','hospital_1','hospital_2']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  while b:=s.read(1<<20):h.update(b)
 return h.hexdigest()

def run():
 out=ROOT/'prepared_v01';out.mkdir(exist_ok=True)
 old=json.loads((ROOT/'source_documents/prior_patient_partition.json').read_text())
 patients=dict(train=old['train']['patients'],source_validation=old['source_val']['patients'],
               hospital_1=old['hospital1']['patients'],hospital_2=old['hospital2']['patients'])
 source=ROOT/'complete_metadata.csv';rows=list(csv.DictReader(source.open()))
 for row in rows:
  for key in ['patient','label','center','row','image_id','node','slide','x_coord','y_coord']:row[key]=int(row[key])
 for i,a in enumerate(POOLS):
  for b in POOLS[i+1:]:assert not set(patients[a])&set(patients[b])
 assert set(patients['hospital_1'])=={r['patient'] for r in rows if r['center']==1}
 assert set(patients['hospital_2'])=={r['patient'] for r in rows if r['center']==2}
 protocol=dict(version='camelyon-native-patient-hospital/1.0',origin='Public wltjr1007/Camelyon17-WILDS mirror of WILDS prepared PNG patches',
  native_shape=[96,96,3],source_centers=[0,3,4],external_centers=[1,2],patients=patients,
  per_patient_per_label_cap=300,sampling_seed=20261011,generated_images=0,
  partition_origin='Reuses previously disclosed source patient partition; larger newly fixed patch sample; both targets were examined in historical projects, not pristine new cohorts',
  evaluation='All eight/nine external patients represented; enriched sampled patch task, not clinical patient diagnosis or the full WILDS leaderboard',
  deduplication='Exclude selected conflicting-label pixel-hash groups and deduplicate exact decoded arrays across pools with fixed train/validation/hospital1/hospital2 precedence',
  tuning='Training accesses source train/validation only; lock every checkpoint before external evaluation',
  metadata_sha256=sha(source))
 target=out/'protocol.json'
 if target.exists():assert json.loads(target.read_text())==protocol
 target.write_text(json.dumps(protocol,indent=2)+'\n')
 rng=np.random.RandomState(20261011);chosen=[]
 for pool in POOLS:
  for patient in patients[pool]:
   for label in [0,1]:
    candidates=[r for r in rows if r['patient']==patient and r['label']==label]
    inds=rng.permutation(len(candidates))[:300]
    for j in inds:chosen.append(dict(**candidates[j],pool=pool))
 groups={}
 for i,row in enumerate(chosen):groups.setdefault(row['file'],[]).append((i,row['row']))
 images=[None]*len(chosen)
 for filename,ids in sorted(groups.items()):
  p=ROOT/'original_mirror'/filename;pf=pq.ParquetFile(p);offset=0;lookup={r:i for i,r in ids}
  for gi in range(pf.num_row_groups):
   n=pf.metadata.row_group(gi).num_rows;selected=[(r-offset,i) for r,i in lookup.items() if offset<=r<offset+n]
   if selected:
    column=pf.read_row_group(gi,columns=['image']).column('image')
    for r,i in selected:
     packed=column[r].as_py();raw=packed['bytes'];image=np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
     assert image.shape==(96,96,3)
     images[i]=image;chosen[i]['encoded_png_sha256']=hashlib.sha256(raw).hexdigest()
     chosen[i]['decoded_sha256']=hashlib.sha256(image.tobytes()).hexdigest()
   offset+=n
  print('NATIVE PATCHES DECODED',filename,len(ids),flush=True)
 hash_labels={}
 for row in chosen:hash_labels.setdefault(row['decoded_sha256'],set()).add(row['label'])
 conflicts={h for h,v in hash_labels.items() if len(v)>1};seen=set();keep=[]
 for i,row in enumerate(chosen):
  h=row['decoded_sha256']
  if h in conflicts or h in seen:continue
  seen.add(h);keep.append(i)
 manifest=[chosen[i] for i in keep];images=[images[i] for i in keep]
 with (out/'sample_manifest.csv').open('w') as s:
  writer=csv.DictWriter(s,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
 counts={}
 for pool in POOLS:
  ids=[i for i,r in enumerate(manifest) if r['pool']==pool]
  x=np.stack([images[i] for i in ids]);y=np.array([manifest[i]['label'] for i in ids]);p=out/(pool+'.npz')
  np.savez_compressed(p,images=x,labels=y,patient=np.array([manifest[i]['patient'] for i in ids]),image_id=np.array([manifest[i]['image_id'] for i in ids]))
  counts[pool]=dict(images=len(ids),patients=sorted({manifest[i]['patient'] for i in ids}),label_counts=np.bincount(y,minlength=2).tolist(),sha256=sha(p))
 (out/'preparation_audit.json').write_text(json.dumps(dict(passed=True,selected_before_deduplication=len(chosen),retained=len(keep),excluded_conflicting_hash_groups=len(conflicts),
  dropped_selected_arrays=len(chosen)-len(keep),patient_overlap=0,exact_pixel_hash_overlap=0,counts=counts,
  original_mirror_checksums=json.loads((ROOT/'original_mirror/checksums.json').read_text()),sample_manifest_sha256=sha(out/'sample_manifest.csv')),indent=2)+'\n')
 print('PATIENT/HOSPITAL PREPARATION COMPLETE',counts,flush=True)
if __name__=='__main__':run()
