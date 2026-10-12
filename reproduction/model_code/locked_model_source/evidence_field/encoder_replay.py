"""Independent NumPy/SciPy input-to-descriptor audit; imports no Torch.

Prepared image bytes must be supplied. This checks numeric provenance, not
lesion meaning, original DICOM authenticity, trained weights or clinical safety.
"""
import hashlib,numpy as np
from scipy.ndimage import convolve
ENCODER_ATOL=1e-3

def recompute_features(record,prepared):
 original=np.ascontiguousarray(prepared)
 if hashlib.sha256(original.tobytes()).hexdigest()!=record['input_sha256']:raise ValueError('Prepared input hash mismatch')
 x=original.astype(np.float64);config=record['config'];dim=x.ndim-1
 if dim not in [2,3] or list(x.shape[1:])!=record['shape']:raise ValueError('Input grid differs from record')
 mode=config['calibration']
 if mode=='display':a=x.mean(0)
 elif mode=='ct_hu':a=(x.mean(0).clip(-1000,2000)+1000)/3000
 elif mode=='optical_density':a=(-np.log(x.clip(1/256,1))).mean(0)/3
 else:raise ValueError('Unknown recorded calibration mode')
 kernel=np.ones((3,)*dim);smooth=convolve(a,kernel,mode='constant',cval=0)/convolve(np.ones_like(a),kernel,mode='constant',cval=0)
 spacing=record['spacing'] or [1.]*dim;derivatives=list(np.gradient(smooth,*spacing,edge_order=1))
 if dim==2:derivatives=[np.zeros_like(a),*derivatives]
 q=np.stack([a,a-smooth,*derivatives,np.sqrt(sum(v*v for v in derivatives)+1e-8)],-1)
 state={k:np.array(v,dtype=float) for k,v in record['encoder_state'].items()};precision=np.logaddexp(0,state['log_precision'])+1e-3
 def radial(z):return np.exp(-.5*(((z[...,None,:]-state['centres'])**2)*precision).sum(-1))
 response=radial(((q-state['q_mean'])/state['q_scale']).clip(-6,6));reference_error=None
 if config.get('reference_conditioning',False):
  axes=tuple(range(dim));mean=q.mean(axes);scale=q.std(axes).clip(.03,None)
  declared=record['image_reference'];reference_error=max(float(abs(mean-np.array(declared['mean'])).max()),float(abs(scale-np.array(declared['scale'])).max()))
  relative=(q-mean)/scale;relative=(relative-state['relative_mean'])/state['relative_scale'];alpha=1/(1+np.exp(-state['mixture_logits']))
  response=alpha*response+(1-alpha)*radial(relative.clip(-6,6))
 bounds=np.array(record['support_bounds']);f=[]
 for bound in bounds:
  axes=[(np.arange(lo,hi)+.5-lo)/(hi-lo)*2-1 for lo,hi in bound];coords=list(np.meshgrid(*axes,indexing='ij'))
  if dim==2:coords=[np.zeros_like(coords[0]),*coords]
  coords=np.stack(coords,-1).reshape(-1,3);values=response[tuple(slice(lo,hi) for lo,hi in bound)].reshape(-1,response.shape[-1]);weight=values.sum(0);valid=weight>1e-6
  centre=(values.T@coords)/weight.clip(1e-6,None)[:,None];cov=[]
  for i in range(3):
   for j in range(i,3):cov.append((values.T@(coords[:,i]*coords[:,j]))/weight.clip(1e-6,None)-centre[:,i]*centre[:,j])
  centre[~valid]=0;cov=np.stack(cov,-1);cov[~valid]=0;f.append(np.c_[values.mean(0),centre,cov].ravel())
 f=np.stack(f);parents=np.maximum(np.array(record['parents']),0);mass=np.prod(bounds[:,:,1]-bounds[:,:,0],1)/np.prod(record['shape']);b=2**dim;depths=np.zeros(len(f),dtype=float);index=0
 for level in range(config['depth']+1):depths[index:index+b**level]=level/max(1,config['depth']);index+=b**level
 phi=np.c_[f,f-f[parents],f-f[0],mass,depths,np.ones(len(f))].clip(-6,6)
 errors={k:float(np.max(abs(phi[int(k)]-np.array(v)))) for k,v in record['feature_registry'].items()};maximum=max(errors.values())
 return dict(passed=maximum<ENCODER_ATOL and (reference_error is None or reference_error<ENCODER_ATOL),max_descriptor_error=maximum,max_reference_error=reference_error,encoder_atol=ENCODER_ATOL,feature_errors=errors,scope='Independent prepared-image/descriptor provenance; not a clinical explanation or native acquisition validation')
