"""Decode every CAMELYON mirror image once; no per-patient sampling cap."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,io,json,collections
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for block in iter(lambda:s.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def decode(filename):
 source=ROOT/'original_mirror'/filename;out=ROOT/'full_camelyon_prepared/shards';out.mkdir(parents=True,exist_ok=True)
 receipt=out/(filename+'.json');array=out/(filename+'.npy');manifest=out/(filename+'.csv')
 if receipt.exists():
  r=json.loads(receipt.read_text());assert sha(array)==r['array_sha256'] and sha(manifest)==r['manifest_sha256'];return r
 checks=json.loads((ROOT/'original_mirror/checksums.json').read_text());expected=next(r for r in checks if r['file']==filename);assert sha(source)==expected['sha256']
 pf=pq.ParquetFile(source);images=np.lib.format.open_memmap(array,mode='w+',dtype='uint8',shape=(pf.metadata.num_rows,96,96,3));offset=0
 with manifest.open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['file','row','patient','center','label','image_id','node','slide','x_coord','y_coord','decoded_sha256'],lineterminator='\n');writer.writeheader()
  for batch in pf.iter_batches(batch_size=1024):
   for record in batch.to_pylist():
    raw=record.pop('image')['bytes'];image=np.array(Image.open(io.BytesIO(raw)).convert('RGB'));assert image.shape==(96,96,3)
    images[offset]=image
    row={k:record[k] for k in ['patient','center','label','image_id','node','slide','x_coord','y_coord']}
    writer.writerow(dict(file=filename+'.npy',row=offset,**row,decoded_sha256=hashlib.sha256(image.tobytes()).hexdigest()));offset+=1
 images.flush();del images;assert offset==pf.metadata.num_rows
 r=dict(file=filename,rows=offset,array_sha256=sha(array),manifest_sha256=sha(manifest),source_sha256=expected['sha256'])
 receipt.write_text(json.dumps(r,indent=2)+'\n');print('FULL SHARD DECODED',filename,offset,flush=True);return r

def run():
 out=ROOT/'full_camelyon_prepared';out.mkdir(exist_ok=True)
 source_audit=json.loads((ROOT/'prepared/preparation_audit.json').read_text());groups={k:set(v['patients']) for k,v in source_audit['counts'].items()}
 assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
 files=sorted(p.name for p in (ROOT/'original_mirror').glob('*.parquet'))
 with ProcessPoolExecutor(max_workers=6) as pool:receipts=list(pool.map(decode,files))
 rows=[]
 for f in files:
  for r in csv.DictReader((out/'shards'/(f+'.csv')).open()):
   patient=int(r['patient']);pools=[p for p,ids in groups.items() if patient in ids];assert len(pools)==1;r['pool']=pools[0];rows.append(r)
 hashes=collections.defaultdict(set)
 for r in rows:hashes[r['decoded_sha256']].add(r['label'])
 conflicts={h for h,labels in hashes.items() if len(labels)>1};seen=set();retained=[]
 precedence={p:i for i,p in enumerate(['train','source_validation','source_test','hospital_1','hospital_2'])}
 for r in sorted(rows,key=lambda r:(precedence[r['pool']],r['file'],int(r['row']))):
  h=r['decoded_sha256']
  if h in conflicts or h in seen:continue
  seen.add(h);retained.append(r)
 counts={}
 for pool in groups:
  members=[r for r in retained if r['pool']==pool];path=out/(pool+'.csv')
  with path.open('w',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(members[0]),lineterminator='\n');writer.writeheader();writer.writerows(members)
  counts[pool]=dict(images=len(members),patients=sorted({int(r['patient']) for r in members}),labels=collections.Counter(r['label'] for r in members),manifest_sha256=sha(path))
 audit=dict(passed=True,version='camelyon-complete-eligible-streamed/1.0',mirror_rows=len(rows),retained_images=len(retained),per_patient_cap=None,
  conflicts_excluded=len(conflicts),duplicates_or_conflicts_excluded=len(rows)-len(retained),counts=counts,shards=receipts,
  patient_overlap=0,exact_image_hash_overlap=0,source_patient_split_sha256=sha(ROOT/'prepared/preparation_audit.json'),
  prior_use='Both external centres previously used; full-data expansion remains exploratory source-locked evaluation',
  source='Full third-party mirror, decoded-byte and mirror-LFS verification; original-publisher byte identity not asserted',
  generated_images=0)
 (out/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('FULL ELIGIBLE CAMELYON PREPARATION COMPLETE',len(retained),flush=True)
if __name__=='__main__':run()
