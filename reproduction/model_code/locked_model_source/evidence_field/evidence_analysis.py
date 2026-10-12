"""Native evidence accounting and reproducible intervention controls.

A spatial score allocation is a computational summary, not a lesion label.
Interventions alter prepared image pixels, including the shared references.
"""
import numpy as np

def signed_spatial_map(record, class_index):
    bounds = np.asarray(record['support_bounds'])
    output = np.zeros(record['shape'], dtype=np.float64)
    for unit in record['classes'][class_index]['units']:
        nodes = [unit['node']] if unit['kind'] == 'node' else unit['nodes']
        volume = sum(np.prod(bounds[v, :, 1] - bounds[v, :, 0]) for v in nodes)
        for v in nodes:
            slices = tuple(slice(lo, hi) for lo, hi in bounds[v])
            output[slices] += unit['score'] / volume
    expected = record['classes'][class_index]['original_logit'] - record['classes'][class_index]['bias']
    if not np.isclose(output.sum(), expected, atol=1e-4, rtol=0):
        raise ArithmeticError('Spatial allocation does not reconstruct signed units')
    return output

def support_keys(record, class_index):
    return {tuple([u['kind'], *([u['node']] if u['kind'] == 'node' else u['nodes'])])
            for u in record['classes'][class_index]['units']}

def rank_pixels(values, seed, low=False):
    strength = np.abs(np.asarray(values)).ravel()
    permutation = np.random.default_rng(seed).permutation(len(strength))
    # Common random tie-breaking makes uniform native maps the random control.
    priority = np.empty(len(strength), dtype=int); priority[permutation] = np.arange(len(strength))
    return np.lexsort((priority, strength if low else -strength))

def intervention_grid(image, baseline, order, fractions, insertion=False):
    a = np.asarray(image); b = np.asarray(baseline)
    if a.shape != b.shape or len(order) != a.size: raise ValueError('Intervention grids differ')
    if len(np.unique(order)) != a.size: raise ValueError('Pixel ranking is not a permutation')
    result = []
    for f in fractions:
        if not 0 <= f <= 1: raise ValueError('Pixel budget outside [0,1]')
        n = int(np.floor(f * a.size))
        start = b.ravel().copy() if insertion else a.ravel().copy()
        start[order[:n]] = (a if insertion else b).ravel()[order[:n]]
        result.append(start.reshape(a.shape))
    return np.stack(result)

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0
