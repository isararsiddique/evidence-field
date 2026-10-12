"""Experimental learned characteristic signatures inside the existing EF grammar.

The complex raw sums are associative. Centering phase is applied after aggregation.
No claim of Fourier features being new is made; this is a versioned EF extension.
Original model/encoder files and previously fitted checkpoint contracts are preserved.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .adaptive_encoder import AdaptiveFieldEncoder
from .encoder import measurements
from .model import EvidenceField, ModelConfig


@dataclass
class StructuralConfig(ModelConfig):
    frequencies: int = 4
    learned_frequencies: bool = True
    frequency_limit: float = 8 * np.pi
    signature_normalization: str = 'response_mass'
    core_version: str = 'ef-structural-development/0.2'


class StructuralFieldEncoder(AdaptiveFieldEncoder):
    def __init__(self, responses=16, seed=11, learned=True, frequencies=4,
                 learned_frequencies=True, frequency_limit=8*np.pi,
                 signature_normalization='response_mass'):
        super().__init__(responses, seed, learned)
        if frequencies < 1 or not np.isfinite(frequency_limit) or frequency_limit <= 0:
            raise ValueError('Positive finite frequency bank required')
        self.frequencies = frequencies
        self.frequency_limit = float(frequency_limit)
        if signature_normalization not in ('response_mass', 'voxel_count'):
            raise ValueError('Unknown characteristic signature normalization')
        self.signature_normalization = signature_normalization
        # This initial bank is fixed by the development protocol, not fitted to tests.
        directions = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 1], [1, 0, 0]], float)
        initial = np.stack([3*np.pi*directions[t % 4] / (1+t//4)
                            for t in range(frequencies)])
        initial = np.clip(initial / frequency_limit, -.95, .95)
        self.raw_wavevectors = nn.Parameter(torch.tensor(np.arctanh(initial),
                                          dtype=torch.float32),
                                          requires_grad=learned_frequencies)

    def wavevectors(self):
        return self.frequency_limit * self.raw_wavevectors.tanh()

    def signature_sums(self, q, g):
        shape = q.shape[2:]
        basis, bounds, volume, midpoint, halfwidth = self.constants(
            shape, q.device, q.dtype, g['bounds'])
        coordinates = basis[1:4].flatten(1).T
        phase = coordinates @ self.wavevectors().T
        harmonic = torch.stack([phase.cos(), phase.sin()], -1).flatten(1)
        response = self.response(q).flatten(2)
        # Dense response/basis matrices, no response-by-basis product allocation.
        pieces = []
        for v in g['levels'][-1]:
            b = g['bounds'][v]
            ids = torch.arange(int(np.prod(shape)), device=q.device).reshape(shape)
            ids = ids[tuple(slice(int(lo), int(hi)) for lo, hi in b)].flatten()
            pieces.append(torch.einsum('nrp,pc->nrc', response[:, :, ids], harmonic[ids]))
        cur = torch.stack(pieces, 1)
        totals = [cur]
        for level in reversed(range(g['depth'])):
            cur = cur.reshape(len(q), len(g['levels'][level]), 2**g['dim'],
                              self.responses_count, 2*self.frequencies).sum(2)
            totals.append(cur)
        return torch.cat(totals[::-1], 1)

    def forward(self, q, g):
        old = super().forward(q, g).reshape(len(q), len(g['mass']),
                                          self.responses_count, 10)
        sums = self.signature_sums(q, g).reshape(len(q), len(g['mass']),
                                                self.responses_count, self.frequencies, 2)
        _, bounds, volume, midpoint, _ = self.constants(q.shape[2:], q.device, q.dtype, g['bounds'])
        phase = midpoint @ self.wavevectors().T
        c, s = phase.cos()[None, :, None], phase.sin()[None, :, None]
        normalizer = (old[..., 0]*volume[None, :, None]).clamp(min=1e-6)[..., None] \
            if self.signature_normalization == 'response_mass' else volume[None, :, None, None]
        real = (sums[..., 0]*c+sums[..., 1]*s)/normalizer
        imag = (sums[..., 1]*c-sums[..., 0]*s)/normalizer
        signature = torch.stack([real, imag], -1).flatten(3)
        return torch.cat([old, signature], -1).flatten(2)


class StructuralEvidenceField(EvidenceField):
    def __init__(self, config: StructuralConfig):
        if not config.reference_conditioning:
            raise ValueError('This prototype specifies the dual-reference encoder')
        super().__init__(config)
        self.encoder = StructuralFieldEncoder(config.responses, config.seed,
            config.learned_encoder, config.frequencies, config.learned_frequencies,
            config.frequency_limit, config.signature_normalization)
        self.width = 3 * config.responses * (10+2*config.frequencies) + 3
        rng = np.random.default_rng(config.seed+29)
        self.theta = nn.Parameter(torch.tensor(rng.normal(0, .01,
            (config.classes, config.node_types, self.width)), dtype=torch.float32))
        if config.pair_types:
            self.pair = nn.Parameter(torch.tensor(rng.normal(0, .002,
                (config.classes, config.pair_types, 2*self.width-1)), dtype=torch.float32))
            self.pair.data[:, :, self.width-1] = -.01
            self.projection = nn.Parameter(torch.tensor(rng.normal(0, .02,
                (config.classes, config.pair_types, self.width-1)), dtype=torch.float32),
                requires_grad=config.bilinear)
            self.gamma = nn.Parameter(torch.full((config.classes, config.pair_types), .01),
                requires_grad=config.bilinear)

    def records(self, *args, **kwargs):
        logits, records = super().records(*args, **kwargs)
        for r in records:
            r['encoder_schema'] = 'evidence-field-characteristic-sums/0.2'
            r['descriptor_layout'] = dict(per_field=10+2*self.config.frequencies,
                frequencies=self.config.frequencies, phase_center='node global midpoint',
                signature_normalization=self.config.signature_normalization,
                bank='bounded learned global normalized-coordinate wavevectors')
            r['implementation_sha256']['structural_field.py'] = sha256(
                Path(__file__).read_bytes()).hexdigest()
        return logits, records


def descriptors_from_measurements(model, q):
    """Stage-isolated diagnostic interface, never presented as a clinical adapter."""
    g = model.layout(q.shape[2:])
    f = model.encoder(q, g)
    parent = torch.as_tensor(np.maximum(g['parent'], 0), device=q.device)
    mass = torch.as_tensor(g['mass'], device=q.device, dtype=q.dtype)
    depth = np.zeros(len(mass))
    for level, ids in enumerate(g['levels']):
        depth[ids] = level/max(1, g['depth'])
    topology = torch.stack([mass, torch.as_tensor(depth, device=q.device,
                            dtype=q.dtype), torch.ones_like(mass)], -1)
    phi = torch.cat([f, f-f[:, parent], f-f[:, :1],
                    topology[None].expand(len(q), -1, -1)], -1).clamp(-6, 6)
    return phi, g


def from_measurements(model, q, tau=0.):
    phi, g = descriptors_from_measurements(model, q)
    if model.config.readout == 'linear':
        return phi[:, 0] @ model.theta[:, 0].T + model.bias
    own, pair, _ = model.scored(phi, g, tau)
    return model.compose(own, pair, g, tau)
