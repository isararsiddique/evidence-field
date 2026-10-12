"""Decode authenticated compact artifacts from a clone or public raw GitHub."""
from pathlib import Path
import argparse,base64,hashlib,json,urllib.request,zipfile,io
P=Path(__file__).resolve().parent
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--ref',default='main');args=a.parse_args()
    base='https://raw.githubusercontent.com/isararsiddique/evidence-field/'+args.ref+'/'
    def read(name):
        f=P/name
        return f.read_bytes() if f.is_file() else urllib.request.urlopen(base+name,timeout=60).read()
    m=json.loads(read('artifacts/Manifest.json'));parts=[]
    for r in m['parts']:
        b=read(r['file']);assert len(b)==r['bytes'] and digest(b)==r['sha256'],r['file'];parts.append(b.strip())
    data=base64.b64decode(b''.join(parts),validate=True)
    assert len(data)==m['archive_bytes'] and digest(data)==m['archive_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            dest=(P/name).resolve();assert dest.is_relative_to(P.resolve()),name
        z.extractall(P)
    q=P/'Evidence_Field_Reproduction'
    ib=(q/'Reproduction_Inventory.json').read_bytes();assert digest(ib)==m['payload_inventory_sha256']
    inv=json.loads(ib)
    for r in inv['files']:
        f=(q/r['file']).resolve();assert f.is_relative_to(q.resolve())
        b=f.read_bytes();assert len(b)==r['bytes'] and digest(b)==r['sha256'],r['file']
    print('Verified compact ZIP and',len(inv['files']),'payloads in',q)
if __name__=='__main__':main()
