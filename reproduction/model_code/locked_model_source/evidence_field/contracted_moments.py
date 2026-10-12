"""Batched tile contractions for the same learned regional moment equations.

No response-by-ten-basis product tensor is materialized. The spatial registry,
global image reference, parameters and moment definition remain unchanged.
This is an optional executor, not a new learned field or a default replacement.
"""
from collections import defaultdict
import numpy as np
import torch

def contracted_response(encoder, q, reference=None):
    """Quadratic expansion avoids a pixel x response x coordinate difference tensor."""
    precision = torch.nn.functional.softplus(encoder.log_precision) + 1e-3
    centre = encoder.centres
    def radial(z):
        quadratic = torch.einsum('...j,rj->...r', z.square(), precision)
        linear = torch.einsum('...j,rj->...r', z, precision * centre)
        constant = (precision * centre.square()).sum(-1)
        distance = (quadratic - 2 * linear + constant).clamp(min=0)
        return torch.exp(-.5 * distance).movedim(-1, 1)
    absolute = radial(((q.movedim(1, -1) - encoder.q_mean) / encoder.q_scale).clamp(-6, 6))
    if not hasattr(encoder, 'reference'):
        return absolute
    if reference is None:
        relative, _, _ = encoder.reference(q)
    else:
        mean, scale = reference; relative = (q - mean) / scale
    z = ((relative.movedim(1, -1) - encoder.relative_mean) / encoder.relative_scale).clamp(-6, 6)
    alpha = torch.sigmoid(encoder.mixture_logits).reshape(1, -1, *([1] * (q.ndim - 2)))
    return alpha * absolute + (1 - alpha) * radial(z)

def contracted_moments(encoder, q, g, voxel_budget=131072, checkpoint_groups=True):
    from torch.utils.checkpoint import checkpoint
    shape = tuple(q.shape[2:]); dim = len(shape); batch = len(q)
    if dim not in (2, 3) or shape != tuple(g['shape']) or voxel_budget < 1:
        raise ValueError('Valid 2D/3D grid and positive voxel budget required')
    groups = defaultdict(list)
    for node in g['levels'][-1]:
        bounds = g['bounds'][node]; width = bounds[:, 1] - bounds[:, 0]
        slab = max(1, int(voxel_budget // max(1, batch * np.prod(width[1:]))))
        for first in range(int(bounds[0, 0]), int(bounds[0, 1]), slab):
            starts = (first, *map(int, bounds[1:, 0]))
            stops = (min(first + slab, int(bounds[0, 1])), *map(int, bounds[1:, 1]))
            groups[tuple(b - a for a, b in zip(starts, stops))].append((int(node), starts, stops))
    reference = encoder.reference(q)[1:] if hasattr(encoder, 'reference') else None
    if reference is not None:
        reference = tuple(v.reshape(batch, 6, *([1] * (dim + 1))) for v in reference)
    chunks = defaultdict(list)
    for tile_shape, entries in groups.items():
        group_size = max(1, int(voxel_budget // max(1, batch * np.prod(tile_shape))))
        for offset in range(0, len(entries), group_size):
            selected = entries[offset:offset + group_size]
            tensors = [q[(slice(None), slice(None), *(slice(a, b) for a, b in zip(starts, stops)))]
                       for _, starts, stops in selected]
            packed = torch.stack(tensors, dim=2)
            def contraction(tensor, entries=tuple(selected)):
                bases = []
                for _, starts, stops in entries:
                    axes = [(torch.arange(a, b, device=q.device, dtype=q.dtype) + .5) / shape[j] * 2 - 1
                            for j, (a, b) in enumerate(zip(starts, stops))]
                    coords = list(torch.meshgrid(*axes, indexing='ij'))
                    if dim == 2:
                        coords = [torch.zeros_like(coords[0]), *coords]
                    basis = torch.stack([torch.ones_like(coords[0]), *coords,
                        *[coords[i] * coords[j] for i in range(3) for j in range(i, 3)]])
                    bases.append(basis.flatten(1))
                basis = torch.stack(bases, 0)
                response = contracted_response(encoder, tensor, reference=reference).flatten(3)
                return torch.einsum('nrgt,gct->ngrc', response, basis)
            value = checkpoint(contraction, packed, use_reentrant=False) \
                if checkpoint_groups and encoder.training and torch.is_grad_enabled() else contraction(packed)
            for index, (node, _, _) in enumerate(selected):
                chunks[node].append(value[:, index])
    leaves = [torch.stack(chunks[int(node)], 0).sum(0) for node in g['levels'][-1]]
    cur = torch.stack(leaves, 1); levels = [cur]
    for level in reversed(range(g['depth'])):
        cur = cur.reshape(batch, len(g['levels'][level]), 2 ** dim, encoder.responses_count, 10).sum(2)
        levels.append(cur)
    moments = torch.cat(list(reversed(levels)), 1)
    bounds = torch.as_tensor(g['bounds'], device=q.device)
    sizes = torch.tensor(shape, device=q.device, dtype=q.dtype)
    volume = (bounds[:, :, 1] - bounds[:, :, 0]).prod(1)
    midpoint = (bounds[:, :, 1] + bounds[:, :, 0]).to(q.dtype) / sizes - 1
    halfwidth = (bounds[:, :, 1] - bounds[:, :, 0]).to(q.dtype) / sizes
    if dim == 2:
        midpoint = torch.cat([torch.zeros_like(midpoint[:, :1]), midpoint], 1)
        halfwidth = torch.cat([torch.ones_like(halfwidth[:, :1]), halfwidth], 1)
    weight = moments[..., 0].clamp(min=1e-6)
    centre = moments[..., 1:4] / weight[..., None]
    covariance = torch.stack([(moments[..., 4 + index] / weight - centre[..., i] * centre[..., j]) /
        (halfwidth[None, :, None, i] * halfwidth[None, :, None, j])
        for index, (i, j) in enumerate(((i, j) for i in range(3) for j in range(i, 3)))], -1)
    valid = moments[..., 0] > 1e-6
    local = (centre - midpoint[None, :, None]) / halfwidth[None, :, None]
    local = torch.where(valid[..., None], local, torch.zeros_like(local))
    covariance = torch.where(valid[..., None], covariance, torch.zeros_like(covariance))
    return torch.cat([(moments[..., 0] / volume[None, :, None])[..., None], local, covariance], -1).flatten(2)
