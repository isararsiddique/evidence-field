"""Input-dependent, exact bounded matching executor; no training framework.

This prunes matching enumeration only. Encoder and candidate-score computation
remain dense. Reduced candidate counts need not reduce end-to-end latency.
"""
import math
import numpy as np

def bounded_program(own, pair, geometry):
    own = np.asarray(own, dtype=float)
    if not np.isfinite(own).all() or (pair is not None and not np.isfinite(pair).all()):
        raise ValueError('Finite signed scores required')
    values = own.copy(); routes = np.zeros_like(own, dtype=int)
    branch = 2 ** geometry['dim']; total = evaluated = pruned = 0
    for level in reversed(range(geometry['depth'])):
        ids = geometry['levels'][level]
        children = geometry['levels'][level + 1].reshape(-1, branch)
        for position, v in enumerate(ids):
            for k in range(own.shape[1]):
                child = values[children[position], k]
                candidates = geometry['matchings'] if pair is not None else [()]
                pv = None if pair is None else pair[geometry['pair_levels'][level][position], k]
                bound = child.copy()
                if pair is not None:
                    for e, (i, j) in enumerate(geometry['edges']):
                        bound[i] = max(bound[i], pv[e] / 2)
                        bound[j] = max(bound[j], pv[e] / 2)
                upper = math.fsum(bound)
                safety = 1e-12 * (1 + math.fsum(np.abs(bound)))
                total += len(candidates)
                if own[v, k] > upper + safety:
                    pruned += len(candidates); continue
                best = own[v, k]
                for m, edges in enumerate(candidates):
                    used = set(); terms = []
                    for e in edges:
                        i, j = geometry['edges'][e]; used.update([i, j]); terms.append(pv[e])
                    terms.extend(child[i] for i in range(branch) if i not in used)
                    value = math.fsum(terms); evaluated += 1
                    if value > best: best = value; routes[v, k] = m + 1
                values[v, k] = best
    return dict(root_scores=values[0], routes=routes, evaluated_matchings=evaluated,
                pruned_matchings=pruned, possible_matchings=total,
                pruning_fraction=pruned / total if total else 0.)
