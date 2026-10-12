"""Local image-upload interface for the complete EF prediction/evidence pipeline."""
from pathlib import Path
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import html
import mimetypes
import subprocess
import sys
import threading
from urllib.parse import unquote, urlparse
import uuid

HERE = Path(__file__).resolve().parent
LOCK = threading.Semaphore(1)

HOME = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence Field · Image prediction</title><style>*{box-sizing:border-box}body{margin:0;background:#f4f7fa;font:17px/1.6 system-ui,sans-serif;color:#183047}main{max-width:820px;margin:65px auto;padding:28px}section{background:white;border:1px solid #d7e0e8;border-radius:18px;padding:30px}h1{font-size:36px;line-height:1.2;margin:0 0 14px}label{display:block;margin:18px 0 8px}input[type=file]{width:100%;padding:20px;border:2px dashed #9fb5c7;border-radius:12px}button{background:#225e86;color:white;border:0;border-radius:10px;padding:14px 24px;font-size:17px;cursor:pointer;margin-top:24px}.note{color:#526779;font-size:15px}a{color:#225e86}.check{display:flex;gap:10px;align-items:center}#status{margin-top:14px}@media(max-width:600px){main{margin:20px auto;padding:14px}section{padding:22px}}</style>
<main><h1>Evidence Field</h1><p>Image prediction with signed, independently replayed evidence.</p><section>
<form action="/predict" method="post" enctype="multipart/form-data" onsubmit="document.getElementById('run').disabled=true;document.getElementById('status').textContent='Computing prediction and independently reconstructing all six class programs…';">
<label for="image">Upload a native 96 × 96 RGB lymph-node pathology patch</label><input id="image" name="image" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" required>
<label class="check"><input type="checkbox" name="calibrated" value="yes"> Use the recorded source-calibrated probability</label>
<button id="run" type="submit">Predict and show evidence</button><p id="status" aria-live="polite"></p></form>
<p class="note">Outputs include the predicted class, member probabilities, selected covers, supporting/opposing score terms and an independent replay report. Both probability modes are saved. Images are processed on this computer.</p>
<p class="note">Trained task: tumour in the central 32 × 32 window of a 96 × 96 lymph-node patch. Whole-slide preparation and other image modalities require their own validated pipelines.</p>
<p><a href="/demo/positive">Run a real positive example</a> · <a href="/demo/negative">Run a real negative example</a> · <a href="/demo/error">Inspect a real incorrect prediction</a></p></section></main></html>'''


def handler_for(root):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, payload, mime='text/html; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def fail(self, message):
            self.send_bytes(400, ('<main style="font:18px system-ui;padding:35px"><h1>Image could not be processed</h1><p>'
                                 + html.escape(message) + '</p><a href="/">Return to image upload</a></main>').encode())

        def inference(self, source, calibrated=False, run_id=None):
            run_id = run_id or uuid.uuid4().hex
            result = root / run_id / 'result'
            cmd = [sys.executable, str(HERE / 'predict_histopathology.py'), '--input', str(source), '--output', str(result)]
            if calibrated:
                cmd.append('--calibrated')
            try:
                with LOCK:
                    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if completed.returncode:
                    detail = next((line for line in reversed(completed.stderr.splitlines()) if line.strip()), 'Inference failed')
                    self.fail(detail)
                    return
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.fail(str(exc))
                return
            self.send_response(303)
            self.send_header('Location', f'/result/{run_id}/report.html')
            self.end_headers()

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/':
                self.send_bytes(200, HOME.encode())
                return
            if path.startswith('/demo/'):
                name = path[len('/demo/'):]
                if name not in ('negative', 'positive', 'error'):
                    self.send_error(404)
                    return
                self.inference(HERE / 'examples' / f'example_{name}.png')
                return
            if path.startswith('/result/'):
                parts = path[len('/result/'):].split('/')
                if len(parts) != 2 or len(parts[0]) != 32 or any(ch not in '0123456789abcdef' for ch in parts[0]):
                    self.send_error(404)
                    return
                directory = (root / parts[0] / 'result').resolve()
                file = (directory / parts[1]).resolve()
                if not file.is_relative_to(directory) or not file.is_file():
                    self.send_error(404)
                    return
                self.send_bytes(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                return
            self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != '/predict':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 10 * 1024 * 1024:
                    self.fail('Provide an image upload no larger than 10 MB.')
                    return
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    self.fail('Use the image-upload form.')
                    return
                message = BytesParser(policy=default).parsebytes(
                    ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + self.rfile.read(length))
                image = None
                calibrated = False
                for part in message.iter_parts():
                    name = part.get_param('name', header='content-disposition')
                    if name == 'image':
                        image = part.get_payload(decode=True)
                    elif name == 'calibrated':
                        calibrated = part.get_payload(decode=True) == b'yes'
                if not image:
                    self.fail('Select a PNG, JPEG or TIFF patch.')
                    return
                run_id = uuid.uuid4().hex
                directory = root / run_id
                directory.mkdir(parents=True)
                source = directory / 'uploaded_image'
                source.write_bytes(image)
                self.inference(source, calibrated=calibrated, run_id=run_id)
            except (ValueError, TypeError, OSError) as exc:
                self.fail(str(exc))

    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--output-root', default='image_results')
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(root))
    print(f'Evidence Field image interface: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
