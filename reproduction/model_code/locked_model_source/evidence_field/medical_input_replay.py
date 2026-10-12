"""Independent SciPy optical input reconstruction; no Torch imports."""
import numpy as np
from scipy.ndimage import convolve

def spectral_measurements_numpy(prepared,spacing=None):
    x=np.asarray(prepared,dtype=np.float64)
    if x.ndim!=3 or x.shape[0]!=3 or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError('Optical input contract mismatch')
    scales=[1.,1.] if spacing is None else list(map(float,spacing))
    if len(scales)!=2 or any(not np.isfinite(v) or v<=0 for v in scales):raise ValueError('Spacing')
    od=-np.log(x.clip(1/256,1));den=convolve(np.ones_like(od[0]),np.ones((3,3)),mode='constant')
    smooth=np.stack([convolve(c,np.ones((3,3)),mode='constant')/den for c in od])
    magnitude=[]
    for c in smooth:
        derivatives=np.gradient(c,*scales)
        magnitude.append(np.sqrt(sum(a*a for a in derivatives)+1e-8))
    return np.concatenate([od,np.stack(magnitude)],0)
