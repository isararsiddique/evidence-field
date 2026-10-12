"""Independent NumPy/SciPy prepared-input replay for structural EF 0.2."""
import hashlib
import numpy as np
from scipy.ndimage import convolve


def structural_features(record, prepared):
    original = np.ascontiguousarray(prepared)
    if hashlib.sha256(original.tobytes()).hexdigest() != record['input_sha256']:
        raise ValueError('Prepared input hash mismatch')
    if record.get('encoder_schema') != 'evidence-field-characteristic-sums/0.2':
        raise ValueError('Wrong structural encoder contract')
    x = original.astype(np.float64)
    config = record['config']; dim = x.ndim-1
    if list(x.shape[1:]) != record['shape'] or dim not in (2, 3):
        raise ValueError('Input shape differs')
    if record.get('prepared_input_contract')=='six-coordinate-structural-diagnostic/1.0':
        if x.shape[0]!=6:raise ValueError('Diagnostic measurement width differs')
        q=np.moveaxis(x,0,-1)
    else:
        mode = config['calibration']
        if mode == 'display': a = x.mean(0)
        elif mode == 'ct_hu': a = (x.mean(0).clip(-1000, 2000)+1000)/3000
        elif mode == 'optical_density': a = -np.log(x.clip(1/256, 1)).mean(0)/3
        else: raise ValueError('Unknown calibration')
        kernel = np.ones((3,)*dim)
        smooth = convolve(a, kernel, mode='constant') / convolve(np.ones_like(a), kernel, mode='constant')
        derivatives = list(np.gradient(smooth, *(record['spacing'] or [1.]*dim)))
        if dim == 2: derivatives = [np.zeros_like(a), *derivatives]
        q = np.stack([a, a-smooth, *derivatives,
                      np.sqrt(sum(v*v for v in derivatives)+1e-8)], -1)
    state = {k: np.asarray(v, float) for k, v in record['encoder_state'].items()}
    precision = np.logaddexp(0, state['log_precision'])+1e-3
    def radial(z):
        return np.exp(-.5*(((z[..., None, :]-state['centres'])**2)*precision).sum(-1))
    absolute = radial(((q-state['q_mean'])/state['q_scale']).clip(-6, 6))
    mean = q.mean(tuple(range(dim))); scale = q.std(tuple(range(dim))).clip(.03, None)
    relative = (((q-mean)/scale-state['relative_mean'])/state['relative_scale']).clip(-6, 6)
    alpha = 1/(1+np.exp(-state['mixture_logits']))
    response = alpha*absolute+(1-alpha)*radial(relative)
    reference_error = max(float(abs(mean-np.asarray(record['image_reference']['mean'])).max()),
                          float(abs(scale-np.asarray(record['image_reference']['scale'])).max()))
    omega = config['frequency_limit']*np.tanh(state['raw_wavevectors'])
    shape = tuple(record['shape']); bounds = np.asarray(record['support_bounds']); features = []
    for bound in bounds:
        global_axes = [(np.arange(lo, hi)+.5)/size*2-1
                       for (lo, hi), size in zip(bound, shape)]
        local_axes = [(np.arange(lo, hi)+.5-lo)/(hi-lo)*2-1 for lo, hi in bound]
        local = list(np.meshgrid(*local_axes, indexing='ij'))
        global_coords = list(np.meshgrid(*global_axes, indexing='ij'))
        if dim == 2:
            local = [np.zeros_like(local[0]), *local]
            global_coords = [np.zeros_like(global_coords[0]), *global_coords]
        local = np.stack(local, -1).reshape(-1, 3)
        global_coords = np.stack(global_coords, -1).reshape(-1, 3)
        values = response[tuple(slice(lo, hi) for lo, hi in bound)].reshape(-1, response.shape[-1])
        weights = values.sum(0); valid = weights > 1e-6
        centre = values.T @ local / weights.clip(1e-6, None)[:, None]
        covariance = np.stack([values.T@(local[:, i]*local[:, j])/weights.clip(1e-6, None)
                               -centre[:, i]*centre[:, j]
                               for i in range(3) for j in range(i, 3)], -1)
        centre[~valid] = 0; covariance[~valid] = 0
        midpoint = (bound[:, 0]+bound[:, 1])/np.asarray(shape)-1
        if dim == 2: midpoint = np.r_[0, midpoint]
        phase = (global_coords-midpoint)@omega.T
        harmonics = np.stack([np.cos(phase), np.sin(phase)], -1).reshape(len(phase), -1)
        normalizer = weights.clip(1e-6, None)[:, None] \
            if config['signature_normalization']=='response_mass' else len(values)
        signature = values.T@harmonics/normalizer
        features.append(np.c_[values.mean(0), centre, covariance, signature].ravel())
    f = np.stack(features); parent = np.maximum(np.asarray(record['parents']), 0)
    mass = np.prod(bounds[:, :, 1]-bounds[:, :, 0], 1)/np.prod(shape)
    depths = np.zeros(len(f)); index = 0; b = 2**dim
    for level in range(config['depth']+1):
        depths[index:index+b**level] = level/max(1, config['depth']); index += b**level
    phi = np.c_[f, f-f[parent], f-f[0], mass, depths, np.ones(len(f))].clip(-6, 6)
    return phi, reference_error


def verify_structural_input(record, prepared, atol=1e-3):
    phi, reference_error = structural_features(record, prepared)
    error = max(float(abs(phi[int(k)]-np.asarray(v)).max())
                for k, v in record['feature_registry'].items())
    if max(error, reference_error) > atol:
        raise ArithmeticError('Structural input/descriptor replay failed')
    return dict(passed=True, maximum_descriptor_error=error,
                maximum_reference_error=reference_error, atol=atol,
                scope='Prepared-input numeric replay, not clinical truth'), phi
