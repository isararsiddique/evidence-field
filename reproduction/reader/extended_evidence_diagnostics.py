"""Framework-free decomposition and route checking for existing EF records.

This read-only helper adds an audit view of the unchanged trained computation.
It does not train, select models, alter evidence records, or certify diagnoses.
External authenticated checkpoint parameters are required; the certificate alone
does not serve as its own source of trust. Near-tied routes are reported separately
from an exact canonical route match because FP32 and FP64 can resolve ties differently.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.abc
import json
import sys
import types
import numpy as np


class ForbidTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('Extended independent evidence audit forbids Torch')


def reader_module(package_dir):
    if any(n == 'torch' or n.startswith('torch.') for n in sys.modules):
        raise RuntimeError('Use a fresh process without the training framework')
    sys.meta_path.insert(0, ForbidTorch())
    package = types.ModuleType('evidence_field')
    package.__path__ = [str(Path(package_dir) / 'evidence_field')]
    sys.modules['evidence_field'] = package
    return importlib.import_module('evidence_field.adaptive_replay')


def scored_components(record, parameters):
    """Split each typed correction into four explicit signed terms.

    Existing region descriptors already include parent/root contrasts. Consequently
    'local_affine' means the affine group feature block, not causal context isolation.
    'parent_affine' and 'parent_modulation' are the additional explicit context blocks.
    """
    g = record['geometry']; c = record['config']
    p = np.asarray(record['phi'], np.float64)
    a = {k: np.asarray(v, np.float64) for k, v in parameters.items()}
    mass = np.asarray(g['mass'], np.float64)
    typed_own = np.einsum('vd,kmd->vkm', p, a['theta']) * mass[:, None, None]
    own = typed_own.max(-1)
    components = {}; typed_groups = {}
    width = p.shape[1]
    for pi, production in enumerate(g['productions']):
        children = np.asarray(production['children'], int)
        context = p[production['parent']] if c['cross_scale'] else np.zeros(width)
        for gi, group in enumerate(g['groups']):
            ids = children[list(group)]; weights = mass[ids]; gm = weights.sum()
            members = p[ids]; mean = (members * weights[:, None]).sum(0) / gm
            spread = (np.abs(members[:, :-1] - mean[None, :-1]) * weights[:, None]).sum(0) / gm
            oi = len(group) - 2
            w = a['group_linear'][oi]
            local_affine = np.einsum('d,ktd->kt', np.r_[mean, spread], w[:, :, :2 * width - 1])
            parent_affine = np.einsum('d,ktd->kt', context, w[:, :, 2 * width - 1:])
            member_product = a['gamma'][oi] * np.tanh(
                np.einsum('gd,ktd->gkt', members[:, :-1], a['member_projection'][oi])).prod(0)
            parent_modulation = member_product * np.tanh(
                np.einsum('d,ktd->kt', context[:-1], a['parent_projection'][oi]))
            terms = {
                'local_affine': gm * local_affine,
                'parent_affine': gm * parent_affine,
                'member_product': gm * member_product,
                'parent_modulation': gm * parent_modulation,
            }
            components[pi, gi] = terms
            typed_groups[pi, gi] = own[ids].sum(0)[:, None] + sum(terms.values())
    return own, typed_own, typed_groups, components


def route_audit(record, own, typed_own, typed_groups, tolerance=1e-5):
    """Independently recompute reachable route choices and explicit tie status."""
    g = record['geometry']; k_count = record['config']['classes']
    values = own.copy(); routes = np.zeros_like(own, dtype=int); options_by_node = {}
    for h in reversed(range(g['depth'])):
        for vi, v in enumerate(g['levels'][h]):
            options = [own[v]]
            for alt in range(g['alternatives']):
                pi = g['production_levels'][h][vi * g['alternatives'] + alt]
                children = g['child_ids'][pi]
                for qi, packing in enumerate(g['packings']):
                    option = (values[children] * np.asarray(g['unmatched'][qi])[:, None]).sum(0)
                    for gi in packing:
                        option = option + typed_groups[pi, gi].max(-1)
                    options.append(option)
            candidates = np.asarray(options)
            routes[v] = candidates.argmax(0)
            values[v] = candidates.max(0)
            options_by_node[v] = candidates
    summaries = []
    for k in range(k_count):
        units = record['classes'][k]['units']
        node_atoms = {u['nodes'][0]: u for u in units if u['kind'] == 'node'}
        group_atoms = {(u['production'], u['group']): u for u in units if u['kind'] == 'group'}
        compared = []

        def parse(v):
            if v in node_atoms:
                unit = node_atoms[v]
                type_gap = float(typed_own[v, k].max() - typed_own[v, k, unit['type']])
                if type_gap > tolerance:
                    raise ArithmeticError('Recorded own type is not optimal')
                if v not in options_by_node:
                    return {id(unit)}
                choices = [0]
                consumed = {id(unit)}
            else:
                if v in g['levels'][-1]:
                    return None
                h = next(h for h, level in enumerate(g['levels']) if v in level)
                vi = g['levels'][h].index(v); choices = []; consumed = None
                for alt in range(g['alternatives']):
                    pi = g['production_levels'][h][vi * g['alternatives'] + alt]
                    selected = {gi for (production, gi) in group_atoms if production == pi}
                    for qi, packing in enumerate(g['packings']):
                        if set(packing) != selected:
                            continue
                        comparison_start = len(compared)
                        used = {id(group_atoms[pi, gi]) for gi in packing}; valid = True
                        for gi in packing:
                            atom = group_atoms[pi, gi]
                            score = typed_groups[pi, gi][k]
                            if score.max() - score[atom['type']] > tolerance:
                                raise ArithmeticError('Recorded group type is not optimal')
                        for ci in np.flatnonzero(g['unmatched'][qi]):
                            subtree = parse(g['child_ids'][pi][ci])
                            if subtree is None:
                                valid = False; break
                            used |= subtree
                        if valid:
                            choices.append(1 + alt * len(g['packings']) + qi)
                            consumed = used
                            break
                        del compared[comparison_start:]
                    if choices:
                        break
                if not choices:
                    return None
            choice = choices[0]
            candidates = options_by_node[v][:, k]
            gap = float(candidates.max() - candidates[choice])
            canonical = int(routes[v, k])
            exact = choice == canonical
            if gap > tolerance:
                raise ArithmeticError('Recorded reachable route is not independently optimal')
            compared.append(dict(node=int(v), recorded_route=int(choice),
                                 independently_selected_route=canonical,
                                 exact_route_match=bool(exact), score_gap=gap,
                                 near_tied_alternative=bool(not exact)))
            return consumed

        consumed = parse(0)
        if consumed != {id(u) for u in units}:
            raise ArithmeticError('Route reader did not consume the whole recorded program')
        summaries.append(dict(class_index=k, reachable_routes=compared,
                              exact_route_matches=sum(r['exact_route_match'] for r in compared),
                              near_tied_alternatives=sum(r['near_tied_alternative'] for r in compared)))
    return summaries, values[0]


def extended_audit(record, prepared, trust, reader):
    native = reader.verify_adaptive(record, prepared, trust['parameters'],
        trusted_config=trust['config'], trusted_model_id=trust['model_id'])
    own, typed_own, typed_groups, components = scored_components(record, trust['parameters'])
    routes, root = route_audit(record, own, typed_own, typed_groups)
    classes = []; correction_errors = []; logit_errors = []
    for k, row in enumerate(record['classes']):
        totals = dict(anchors=0., local_affine=0., parent_affine=0.,
                      member_product=0., parent_modulation=0.,
                      bias=float(trust['parameters']['bias'][k]))
        units = []
        for atom in row['units']:
            if atom['kind'] == 'node':
                anchor = float(own[atom['nodes'][0], k]); totals['anchors'] += anchor
                units.append(dict(kind='node', nodes=atom['nodes'], anchor=anchor))
            else:
                terms = components[atom['production'], atom['group']]
                selected = {name: float(value[k, atom['type']]) for name, value in terms.items()}
                anchor = float(own[atom['nodes'], k].sum()); totals['anchors'] += anchor
                for name, value in selected.items(): totals[name] += value
                correction_errors.append(abs(sum(selected.values()) - atom['correction']))
                units.append(dict(kind='group', nodes=atom['nodes'], parent=atom['parent'],
                                  production=atom['production'], group=atom['group'],
                                  type=atom['type'], anchors=anchor, **selected))
        logit = sum(totals.values()); logit_errors.append(abs(logit - row['logit']))
        classes.append(dict(class_index=k, signed_totals=totals,
                            reconstructed_logit=logit, recorded_logit=row['logit'], units=units))
    error = max(logit_errors + correction_errors + [0.])
    if error > 1e-4:
        raise ArithmeticError('Cross-scale decomposition did not reconstruct the original program')
    direct_phi = reader.independent_phi(record, prepared)
    width = record['config']['responses'] * (10 + 2 * record['config']['frequencies'])
    per_response = 10 + 2 * record['config']['frequencies']
    arranged = direct_phi[:, :width].reshape(len(direct_phi), record['config']['responses'], per_response)
    signature = arranged[..., 10:]
    return dict(passed=True, native_reader=native, classes=classes, routes=routes,
                maximum_decomposition_error=error,
                maximum_root_optimum_error=float(np.abs(root + np.asarray(trust['parameters']['bias']) - record['logits']).max()),
                structural_signature=dict(frequencies=record['config']['frequencies'],
                    learned_frequencies=record['config']['learned_frequencies'],
                    independent_signature_count=int(signature.size),
                    maximum_absolute_signature=float(np.abs(signature).max())),
                training_framework_imported=False,
                scope='Unchanged original-program accounting and independent reachable-route audit; not clinical validation',
                context_note='Region descriptors contain parent/root contrasts. Explicit context terms are additional computational contributions, not causal biological effects.')


def run(args):
    reader = reader_module(args.package_dir)
    trust_bytes = Path(args.trust_json).read_bytes()
    if hashlib.sha256(trust_bytes).hexdigest() != args.trust_sha256:
        raise ValueError('External trust-file digest mismatch')
    trust = json.loads(trust_bytes); record = json.loads(Path(args.record).read_text())
    prepared = np.load(args.prepared, allow_pickle=False)
    result = extended_audit(record, prepared, trust, reader)
    result.update(record_sha256=hashlib.sha256(Path(args.record).read_bytes()).hexdigest(),
                  trust_sha256=args.trust_sha256, checkpoint_sha256=trust['checkpoint_sha256'],
                  input_sha256=record['input_sha256'], helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['package-dir', 'trust-json', 'trust-sha256', 'record', 'prepared', 'output']:
        p.add_argument('--' + name, required=True)
    run(p.parse_args())
