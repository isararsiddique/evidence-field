"""Hospital-local manifest validation and image/volume loading."""
from pathlib import Path
import csv,hashlib
import numpy as np
from PIL import Image

REQUIRED={'file','patient_id','site_id','split','label'}

def validate_manifest(path):
 path=Path(path)
 with path.open(newline='') as f:
  reader=csv.DictReader(f)
  if not REQUIRED.issubset(reader.fieldnames or []):raise ValueError('Manifest requires file, patient_id, site_id, split, label')
  rows=list(reader)
 if not rows:raise ValueError('Empty hospital manifest')
 patients={};inputs={};files=set()
 for row in rows:
  if not row['patient_id'] or not row['site_id']:raise ValueError('Patient/site IDs must be supplied, not invented')
  if row['split'] not in ('train','val','calibration','test'):raise ValueError('Unknown split')
  p=Path(row['file']);p=p if p.is_absolute() else path.parent/p
  if not p.is_file():raise ValueError('Missing declared input')
  if str(p) in files:raise ValueError('Duplicate manifest input');
  files.add(str(p));row['resolved_file']=str(p)
  patient=row['patient_id']
  if patient in patients and patients[patient]!=row['split']:raise ValueError('Patient crosses train/validation/calibration/test split')
  patients[patient]=row['split'];h=hashlib.sha256(p.read_bytes()).hexdigest();row['input_sha256']=h
  if h in inputs and inputs[h]!=row['split']:raise ValueError('Identical input bytes cross splits')
  inputs[h]=row['split']
 pools={r['split'] for r in rows}
 if not {'train','val','calibration','test'}.issubset(pools):raise ValueError('Separate train, model-validation, calibration and test groups are required')
 labels={int(r['label']) for r in rows}
 if labels!=set(range(len(labels))) or len(labels)<2:raise ValueError('Contiguous task-defined labels 0..K-1 are required')
 return rows,dict(rows=len(rows),patients=len(patients),sites=sorted({r['site_id'] for r in rows}),split_patients={s:sum(v==s for v in patients.values()) for s in pools},input_hashes_audited=True,patient_separation_verified=True,site_separation='Explicit IDs available; local training does not itself establish an unseen-hospital test')

def load_image(path,calibration='display'):
 p=Path(path)
 if p.suffix.lower()=='.npy':a=np.load(p,allow_pickle=False)
 elif p.name.endswith('.nii') or p.name.endswith('.nii.gz'):
  import nibabel as nib
  image=nib.load(p);a=image.get_fdata(dtype=np.float32)
  # NIfTI physical spacing/affine must be preserved by the task adapter.
 elif p.suffix.lower()=='.dcm':
  import pydicom
  d=pydicom.dcmread(p);a=d.pixel_array.astype(np.float32)
  if calibration=='ct_hu':
   if getattr(d,'Modality',None)!='CT' or not hasattr(d,'RescaleSlope') or not hasattr(d,'RescaleIntercept'):raise ValueError('CT modality and explicit rescale slope/intercept required for HU')
   a=a*float(d.RescaleSlope)+float(d.RescaleIntercept)
  else:raise ValueError('A validated DICOM display/sequence adapter is required')
 else:
  image=Image.open(p);image=image.convert('RGB') if image.mode=='P' else image;a=np.array(image)
 original_uint8=a.dtype==np.uint8
 if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff') and a.ndim==3 and a.shape[-1] in (3,4):a=a[...,:3].mean(-1)
 if a.ndim not in (2,3):raise ValueError('Expected scalar image or volume; declare multichannel conversion explicitly')
 if not np.isfinite(a).all():raise ValueError('Nonfinite image measurements')
 if calibration=='display':
  if original_uint8:a=a.astype(np.float32)/255
  elif np.min(a)<0 or np.max(a)>1:raise ValueError('Display float images require declared scaling into [0,1]')
 return a.astype(np.float32)
