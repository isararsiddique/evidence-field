"""End-to-end inference with the three original, authenticated histology models."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Native 96x96 RGB PNG, JPEG or TIFF patch')
    parser.add_argument('--output', required=True, help='New result directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--calibrated', action='store_true', help='Select the unchanged source-calibrated result; both modes are saved')
    args = parser.parse_args()
    inventory = json.loads((HERE / 'Package_Inventory.json').read_text())
    for row in inventory['files']:
        path = (HERE / row['file']).resolve()
        if not path.is_relative_to(HERE.resolve()) or sha(path) != row['sha256']:
            raise RuntimeError('Package file failed authentication: ' + row['file'])
    source = Path(args.input).resolve()
    with Image.open(source) as image:
        if image.size != (96, 96) or image.mode != 'RGB':
            raise ValueError('This trained model requires a native 96x96 RGB patch. No silent resizing or whole-slide diagnosis is performed.')
        rgb = np.asarray(image, dtype=np.uint8).copy()
    prepared = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255)
    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(HERE / 'implementation/locked_model_source'))
    from evidence_field.dense_program import DenseOpticalProgram
    from evidence_field.stable_program import StableProgramConfig
    settings = json.loads((HERE / 'checkpoints/Ensemble_Settings.json').read_text())
    assert all(m['temperature'] > 0 and m['weight'] >= 0 for m in settings['members'])
    assert abs(sum(m['weight'] for m in settings['members']) - 1) < 1e-12
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory to preserve existing results.')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'prepared_input.npy', prepared, allow_pickle=False)
    Image.fromarray(rgb).save(out / 'input.png')
    x = torch.from_numpy(prepared)[None].to(args.device)
    start = time.perf_counter()
    members = []
    for member in settings['members']:
        seed = member['seed']
        path = HERE / 'checkpoints' / member['file']
        assert sha(path) == member['sha256']
        config = settings['model_config_by_seed'][str(seed)]
        model = DenseOpticalProgram(StableProgramConfig(**config))
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        model = model.to(args.device).eval()
        with torch.no_grad():
            phi, geometry = model.descriptors(x)
            own, groups, _ = model.scored(phi, geometry, 0.)
            logits = model.compose(own.double(), groups.double(), geometry, 0.)[0]
            raw = torch.softmax(logits.double(), -1).cpu().numpy()
            calibrated = torch.softmax(logits.double() / member['temperature'], -1).cpu().numpy()
            record = model.certificate(x)
        if np.max(np.abs(np.asarray(record['logits']) - logits.cpu().numpy())) > 2e-5:
            raise ArithmeticError('The exported evidence does not reconstruct prediction logits.')
        trust = dict(config=config, parameters=record['parameters'], model_id=record['model_id'],
                     checkpoint_sha256=member['sha256'], temperature=member['temperature'], weight=member['weight'])
        with gzip.open(out / f'evidence_seed_{seed}.json.gz', 'wt') as stream:
            json.dump(record, stream, separators=(',', ':'))
        (out / f'trust_seed_{seed}.json').write_text(json.dumps(trust, separators=(',', ':')) + '\n')
        members.append(dict(seed=seed, checkpoint_sha256=member['sha256'],
            model_id=record['model_id'], temperature=member['temperature'], weight=member['weight'],
            logits=logits.cpu().numpy().tolist(), raw_probability=raw.tolist(),
            calibrated_probability=calibrated.tolist(),
            evidence_file=f'evidence_seed_{seed}.json.gz', evidence_sha256=sha(out / f'evidence_seed_{seed}.json.gz'),
            trust_file=f'trust_seed_{seed}.json', trust_sha256=sha(out / f'trust_seed_{seed}.json')))
    raw = sum(m['weight'] * np.asarray(m['raw_probability']) for m in members)
    calibrated = sum(m['weight'] * np.asarray(m['calibrated_probability']) for m in members)
    selected = calibrated if args.calibrated else raw
    summary = dict(schema='EF-histopathology-image-inference/1',
        input_file=source.name, input_file_sha256=sha(source), input_shape=[3, 96, 96],
        prepared_input_file_sha256=sha(out / 'prepared_input.npy'),
        preprocessing='Native uint8 RGB to float32 CHW divided by 255; original optical encoder follows',
        class_labels=settings['class_labels'], raw_probability=raw.tolist(),
        source_calibrated_probability=calibrated.tolist(),
        selected_probability_mode='source_calibrated' if args.calibrated else 'raw',
        predicted_class_index=int(selected.argmax()), predicted_label=settings['class_labels'][int(selected.argmax())],
        members=members, device=args.device, torch_version=torch.__version__,
        elapsed_seconds=time.perf_counter() - start,
        task='Tumour in the central 32x32 window of a 96x96 lymph-node patch',
        evidence_interpretation='Signed computation and selected covering supports, not an independently validated clinical localization',
        optimizer_steps=0)
    (out / 'prediction.json').write_text(json.dumps(summary, indent=2) + '\n')
    subprocess.run([sys.executable, str(HERE / 'verify_evidence.py'), '--output', str(out)], check=True)
    verified = json.loads((out / 'independent_evidence_verification.json').read_text())
    assert verified['passed']
    from render_evidence import create_report
    create_report(out)
    print(json.dumps(dict(predicted_label=summary['predicted_label'], raw_probability=raw.tolist(),
        source_calibrated_probability=calibrated.tolist(), independent_replay_passed=True,
        selected_probability_mode=summary['selected_probability_mode'], output=str(out),
        visible_report=str(out / 'report.html'), evidence_image=str(out / 'evidence_overview.png')), indent=2))


if __name__ == '__main__':
    main()
