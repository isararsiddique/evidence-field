"""Visible, score-preserving reports from independently checked EF programs."""
from pathlib import Path
import argparse
import base64
import gzip
import hashlib
import html
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(57, 139, 192), (242, 179, 102), (100, 181, 150),
           (180, 143, 196), (238, 146, 152), (139, 177, 205),
           (203, 194, 125), (145, 186, 169)]
TERMS = ['anchors', 'local_affine', 'parent_affine', 'member_product', 'parent_modulation', 'bias']
TERM_LABELS = ['Anchors', 'Local affine', 'Parent affine', 'Member product', 'Parent modulation', 'Bias']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    return ImageFont.load_default(size=size)


def image_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def create_report(directory):
    out = Path(directory)
    prediction = json.loads((out / 'prediction.json').read_text())
    verification = json.loads((out / 'independent_evidence_verification.json').read_text())
    if not verification['passed'] or verification['class_programs'] != 6:
        raise ArithmeticError('Visible evidence requires successful six-program reconstruction.')
    rgb = np.array(Image.open(out / 'input.png').convert('RGB'))
    members = []
    for member, audit in zip(prediction['members'], verification['members']):
        assert member['seed'] == audit['seed']
        assert sha(out / member['evidence_file']) == member['evidence_sha256']
        with gzip.open(out / member['evidence_file'], 'rt') as stream:
            record = json.load(stream)
        bounds = np.array(record['geometry']['bounds'], dtype=int)
        fields = np.zeros((2, 96, 96))
        covers = np.zeros((2, 96, 96), dtype=int)
        regions = []
        for k, row in enumerate(record['classes']):
            for unit_index, unit in enumerate(row['units']):
                area = sum(int(np.prod(bounds[node, :, 1] - bounds[node, :, 0])) for node in unit['nodes'])
                for node in unit['nodes']:
                    (yl, yh), (xl, xh) = bounds[node]
                    fields[k, yl:yh, xl:xh] += unit['score'] / area
                    covers[k, yl:yh, xl:xh] += 1
                    regions.append(dict(class_index=k, unit=unit_index, kind=unit['kind'],
                        node=node, bounds_yx=bounds[node].tolist(), score=unit['score'],
                        unit_area=area, displayed_score_density=unit['score'] / area))
        assert np.all(covers == 1)
        totals = audit['signed_class_totals']
        contrasts = {name: totals[1][name] - totals[0][name] for name in TERMS}
        margin = fields[1] - fields[0]
        bias_margin = contrasts['bias']
        reconstructed_margin = float(margin.sum() + bias_margin)
        actual_margin = member['logits'][1] - member['logits'][0]
        assert abs(reconstructed_margin - actual_margin) <= 2e-5
        assert abs(sum(contrasts.values()) - actual_margin) <= 2e-5
        cover = Image.fromarray(rgb).convert('RGBA')
        overlay = Image.new('RGBA', cover.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for unit_index, unit in enumerate(record['classes'][1]['units']):
            color = PALETTE[unit_index % len(PALETTE)]
            for node in unit['nodes']:
                (yl, yh), (xl, xh) = bounds[node]
                draw.rectangle((int(xl), int(yl), int(xh - 1), int(yh - 1)),
                               fill=(*color, 100), outline=(32, 50, 68, 255), width=1)
        Image.alpha_composite(cover, overlay).convert('RGB').save(out / f'cover_seed_{member["seed"]}.png')
        # Colour the exact hard-program score allocation. No learned saliency map.
        displayed = margin * 9216
        limit = max(float(np.abs(displayed).max()), 1e-12)
        strength = np.abs(displayed / limit)[..., None]
        colors = np.where((displayed >= 0)[..., None], np.array([194, 68, 58]), np.array([43, 103, 161]))
        heatmap = np.rint(255 * (1 - strength) + colors * strength).astype(np.uint8)
        Image.fromarray(heatmap).save(out / f'margin_seed_{member["seed"]}.png')
        np.savez_compressed(out / f'signed_fields_seed_{member["seed"]}.npz',
                           class_fields=fields, tumour_minus_normal=margin, bias_margin=bias_margin)
        members.append(dict(seed=member['seed'], class_scores=member['logits'],
            signed_class_totals=totals, tumour_minus_normal_terms=contrasts,
            reconstructed_margin=reconstructed_margin, actual_margin=actual_margin,
            margin_reconstruction_error=abs(reconstructed_margin - actual_margin),
            tumour_cover_units=len(record['classes'][1]['units']), onefold_cover=True,
            display_range=[-limit, limit], regions=regions,
            member_raw_probability=member['raw_probability'],
            member_calibrated_probability=member['calibrated_probability']))
    member_labels = [int(np.argmax(m['member_raw_probability'])) for m in members]
    visible = dict(passed=True, input_image_sha256=sha(out / 'input.png'),
        predicted_label=prediction['predicted_label'], raw_probability=prediction['raw_probability'],
        source_calibrated_probability=prediction['source_calibrated_probability'],
        members_disagree=len(set(member_labels)) > 1,
        members=members, all_six_class_programs_replayed=True,
        field_definition='Each hard-program unit score is allocated uniformly over its selected support. Field integral plus bias reconstructs each member logit; class difference reconstructs the member margin.',
        ensemble_definition='Apply softmax to each member, then the registered weighted probability mixture; fields are not an additive ensemble-probability explanation.',
        scope='Computational evidence; clinical localization and review policy require separate validation.')
    (out / 'visible_evidence.json').write_text(json.dumps(visible, indent=2) + '\n')
    write_overview(out, prediction, verification, members)
    write_html(out, prediction, verification, visible)
    return visible


def write_overview(out, prediction, verification, members):
    canvas = Image.new('RGB', (1480, 990), '#f6f8fb')
    d = ImageDraw.Draw(canvas)
    d.text((34, 22), 'Evidence Field | Prediction and evidence', font=font(31), fill='#193148')
    d.text((34, 72), f"Prediction: {prediction['predicted_label']}   |   Raw P(tumour): {prediction['raw_probability'][1]:.2%}",
           font=font(23), fill='#193148')
    d.text((34, 110), 'Independent replay: 6/6 class programs passed. Native 96 x 96 pixels enlarged for display.',
           font=font(18), fill='#516779')
    positions = [34, 394, 754, 1114]
    titles = ['Actual input / label window'] + [f"Seed {m['seed']} | tumour cover" for m in members]
    for x, title in zip(positions, titles):
        d.rounded_rectangle((x, 163, x + 330, 549), 14, fill='white', outline='#d4dce4', width=2)
        d.text((x + 14, 179), title, font=font(19), fill='#193148')
    original = Image.open(out / 'input.png').convert('RGB').resize((300, 300), Image.Resampling.NEAREST)
    image_draw = ImageDraw.Draw(original)
    image_draw.rectangle((100, 100, 199, 199), outline='white', width=3)
    canvas.paste(original, (49, 219))
    for x, member in zip(positions[1:], members):
        picture = Image.open(out / f"cover_seed_{member['seed']}.png").resize((300, 300), Image.Resampling.NEAREST)
        canvas.paste(picture, (x + 15, 219))
    d.rounded_rectangle((34, 574, 364, 913), 14, fill='white', outline='#d4dce4', width=2)
    d.text((49, 590), 'Member probabilities', font=font(22), fill='#193148')
    for i, member in enumerate(members):
        d.text((49, 638 + i * 55), f"Seed {member['seed']}: P(tumour) {member['member_raw_probability'][1]:.2%}",
               font=font(19), fill='#193148')
    d.text((49, 817), 'Red: positive score margin', font=font(17), fill='#a63831')
    d.text((49, 846), 'Blue: negative score margin', font=font(17), fill='#28629b')
    for x, member in zip(positions[1:], members):
        d.rounded_rectangle((x, 574, x + 330, 913), 14, fill='white', outline='#d4dce4', width=2)
        d.text((x + 14, 590), f"Seed {member['seed']} | signed margin", font=font(19), fill='#193148')
        d.text((x + 14, 611), f"Scale ±{member['display_range'][1]:.3g} (9216 x density)", font=font(12), fill='#516779')
        picture = Image.open(out / f"margin_seed_{member['seed']}.png").resize((268, 268), Image.Resampling.NEAREST)
        canvas.paste(picture, (x + 31, 628))
    d.text((34, 942), 'Maps allocate selected program scores across their supports. Member scores reconstruct predictions; colours do not identify cells.',
           font=font(17), fill='#516779')
    canvas.save(out / 'evidence_overview.png')


def write_html(out, prediction, verification, visible):
    escaped = html.escape
    mode = prediction['selected_probability_mode']
    p = prediction['source_calibrated_probability'] if mode == 'source_calibrated' else prediction['raw_probability']
    cards = []
    for member in visible['members']:
        seed = member['seed']
        rows = ''.join(f'<tr><td>{label}</td><td>{member["tumour_minus_normal_terms"][term]:+.6f}</td></tr>'
                       for term, label in zip(TERMS, TERM_LABELS))
        cards.append(f'''<section class="card"><h2>Member {seed}</h2>
          <p>Raw tumour probability <strong>{member['member_raw_probability'][1]:.2%}</strong></p>
          <div class="images"><figure><img src="{image_uri(out / f'cover_seed_{seed}.png')}" alt="Selected tumour-class covering"><figcaption>Selected tumour-class cover · {member['tumour_cover_units']} units</figcaption></figure>
          <figure><img src="{image_uri(out / f'margin_seed_{seed}.png')}" alt="Signed program margin"><figcaption>Signed tumour − normal score field<br>Display scale ±{member['display_range'][1]:.3g} (9216 × density)</figcaption></figure></div>
          <table><thead><tr><th>Score contribution</th><th>Tumour − normal</th></tr></thead><tbody>{rows}
          <tr class="total"><td>Sum = member logit margin</td><td>{member['actual_margin']:+.6f}</td></tr></tbody></table>
          <p class="note">Field integral + bias = {member['reconstructed_margin']:+.6f}; reconstruction error {member['margin_reconstruction_error']:.2e}.</p></section>''')
    disagreement = ('Members predict different classes; their separate programs and probabilities are shown below.'
                    if visible['members_disagree'] else 'All three members predict the same class.')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Evidence Field image result</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f4f7fa;color:#183047;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1480px;margin:auto;padding:28px}}
    h1{{font-size:32px;margin:0 0 12px}}h2{{font-size:22px;margin:0 0 12px}}p{{margin:8px 0}}.card{{background:white;border:1px solid #d6e0e9;border-radius:16px;padding:22px;min-width:0}}
    .summary{{display:grid;grid-template-columns:240px 1fr;gap:26px;align-items:center}}.summary img{{width:100%;image-rendering:pixelated;border-radius:8px}}
    .prediction{{font-size:25px;font-weight:700}}.badge{{display:inline-block;background:#e5f5ec;color:#195d3b;border-radius:8px;padding:5px 11px;font-weight:600}}
    .grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin:24px 0}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:12px 0;min-width:0}}figure img{{width:100%;image-rendering:pixelated;border:1px solid #d6e0e9}}figcaption{{font-size:13px;color:#526779;min-height:42px}}
    table{{width:100%;border-collapse:collapse;font-size:14px;table-layout:fixed}}th,td{{padding:9px 4px;text-align:left;border-bottom:1px solid #e3e9ef;overflow-wrap:anywhere}}th:last-child,td:last-child{{text-align:right}}.total{{font-weight:700}}.note{{color:#526779;font-size:14px}}a{{color:#245d90}}
    @media(max-width:1100px){{.grid{{grid-template-columns:1fr}}.images{{max-width:680px}}}}@media(max-width:580px){{main{{padding:14px}}.summary{{grid-template-columns:1fr}}.summary img{{max-width:240px}}}}
    </style><main><h1>Evidence Field · Image prediction</h1><section class="card summary"><img src="{image_uri(out / 'input.png')}" alt="Actual input pathology image">
    <div><p class="prediction">{escaped(prediction['predicted_label'].replace('_',' '))} · {p[1 if prediction['predicted_class_index']==1 else 0]:.2%}</p>
    <p>Probability mode: {escaped(mode.replace('_',' '))}. Raw P(tumour): {prediction['raw_probability'][1]:.2%}; source-calibrated: {prediction['source_calibrated_probability'][1]:.2%}.</p>
    <p><span class="badge">Independent replay passed · 6/6 class programs</span></p><p>{escaped(disagreement)}</p>
    <p class="note">Task: {escaped(prediction['task'])}. Input {escaped(prediction['input_file'])}. Each member uses its original authenticated checkpoint.</p></div></section>
    <div class="grid">{''.join(cards)}</div><section class="card"><h2>How to read this evidence</h2>
    <p>Cover colours distinguish selected computational units. In the margin maps, red is a positive tumour-versus-normal contribution and blue is an opposing contribution. Each unit score is distributed uniformly over its selected support for display. Its integral plus bias reconstructs the member score margin.</p>
    <p>Each member’s two class scores pass through softmax, then the registered weights combine probabilities. The displayed score fields explain individual member programs; they are not an additive decomposition of the ensemble probability.</p>
    <p>The independent reader recomputed descriptors, covering supports, selected routes, signed terms and optima without importing Torch. Maximum probability reconstruction error: {verification['maximum_probability_error']:.2e}.</p>
    <p class="note">This is auditable computational evidence. Clinical localization and a hospital review policy require separate validation.</p>
    <p><a href="prediction.json">Prediction JSON</a> · <a href="visible_evidence.json">Selected supports and signed scores</a> · <a href="independent_evidence_verification.json">Independent replay record</a></p></section></main></html>'''
    (out / 'report.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    create_report(parser.parse_args().output)
