"""Download/decode and authenticate the complete prediction/evidence application."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='main')
    parser.add_argument('--output', default=str(HERE))
    args = parser.parse_args()
    root = Path(args.output).resolve()
    base = 'https://raw.githubusercontent.com/isararsiddique/evidence-field/' + args.ref + '/'

    def read(name):
        local = HERE / name
        return local.read_bytes() if local.is_file() else urllib.request.urlopen(base + name, timeout=60).read()

    manifest = json.loads(read('application_artifacts/Manifest.json'))
    parts = []
    for item in manifest['parts']:
        data = read(item['file'])
        assert len(data) == item['bytes'] and digest(data) == item['sha256'], item['file']
        parts.append(data.strip())
    archive = base64.b64decode(b''.join(parts), validate=True)
    assert len(archive) == manifest['archive_bytes'] and digest(archive) == manifest['archive_sha256']
    target = root / 'Histopathology_EF'
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Choose a new --output directory to preserve the existing application.')
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in bundle.namelist():
            assert (root / name).resolve().is_relative_to(root), name
        bundle.extractall(root)
    inventory_bytes = (target / 'Package_Inventory.json').read_bytes()
    assert digest(inventory_bytes) == manifest['payload_inventory_sha256']
    inventory = json.loads(inventory_bytes)
    for item in inventory['files']:
        path = (target / item['file']).resolve()
        assert path.is_relative_to(target.resolve()), item['file']
        data = path.read_bytes()
        assert len(data) == item['bytes'] and digest(data) == item['sha256'], item['file']
    print('Verified complete application:', len(inventory['files']), 'payloads in', target)


if __name__ == '__main__':
    main()
