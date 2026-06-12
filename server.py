from flask import Flask, request, jsonify, make_response, send_from_directory
import subprocess, base64, tempfile, os, uuid, sys, threading, traceback
import numpy as np
import cv2

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, 'Wav2Lip')
CHECKPOINT  = os.path.join(WAV2LIP_DIR, 'checkpoints', 'Wav2Lip-SD-GAN.pt')
AVATAR_DIR  = os.path.join(BASE_DIR, 'uploads')
TEMP_DIR    = os.path.join(WAV2LIP_DIR, 'temp')
os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# Add Wav2Lip to path so we can import its modules directly
sys.path.insert(0, WAV2LIP_DIR)
import torch
from models import Wav2Lip as Wav2LipModel
import audio as wav2lip_audio

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

stored_face = {'path': None, 'is_video': False, 'box': None}
for _ext, _vid in [('mp4', True), ('png', False)]:
    _p = os.path.join(AVATAR_DIR, f'current.{_ext}')
    if os.path.exists(_p):
        stored_face = {'path': _p, 'is_video': _vid, 'box': None}
        print(f'[startup] restored face: {_p}')
        break

jobs       = {}
jobs_lock  = threading.Lock()
infer_lock = threading.Lock()   # serialise CPU inference

VOICE_MAP = {
    'en-US': 'en-US-JennyNeural', 'en-GB': 'en-GB-SoniaNeural',
    'hi-IN': 'hi-IN-SwaraNeural', 'es-ES': 'es-ES-ElviraNeural',
    'fr-FR': 'fr-FR-DeniseNeural', 'de-DE': 'de-DE-KatjaNeural',
    'ja-JP': 'ja-JP-NanamiNeural',
}

# ── Load model ONCE at startup — eliminates the 20-30 s per-request overhead ─

def _load_checkpoint():
    print('[startup] Loading Wav2Lip model...')
    loc = torch.device('cpu')
    try:
        m = torch.jit.load(CHECKPOINT, map_location=loc)
        print('[startup] TorchScript checkpoint loaded.')
        return m.eval()
    except Exception:
        ckpt = torch.load(CHECKPOINT, map_location=loc, weights_only=False)
        m = Wav2LipModel()
        s = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        m.load_state_dict(s)
        print('[startup] State-dict checkpoint loaded.')
        return m.cpu().eval()

try:
    WAV2LIP_MODEL = _load_checkpoint()
except Exception as _e:
    print(f'[startup] WARNING — model not loaded: {_e}')
    WAV2LIP_MODEL = None

# ── CORS ──────────────────────────────────────────────────────────────────────

def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return r

@app.after_request
def after(r): return cors(r)

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/config', methods=['GET'])
def config():
    # Groq key lives in Render env var — never in code or HTML
    return jsonify({'groq_key': os.environ.get('GROQ_API_KEY', '')})

@app.route('/health', methods=['GET'])
def health():
    return (f'OK | model:{WAV2LIP_MODEL is not None} '
            f'| ckpt:{os.path.exists(CHECKPOINT)} '
            f'| face:{stored_face["path"]}')

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({
        'model_loaded': WAV2LIP_MODEL is not None,
        'checkpoint':   os.path.exists(CHECKPOINT),
        'stored_face':  stored_face,
        'torch':        torch.__version__,
        'cv2':          cv2.__version__,
    })

# ── Media helpers ─────────────────────────────────────────────────────────────

def save_face_media(avatar_b64, out_dir, name):
    from PIL import Image
    import io as _io

    if avatar_b64 and avatar_b64.startswith('data:'):
        mime = avatar_b64.split(';')[0].split(':')[1]
        data = base64.b64decode(avatar_b64.split(',')[1])

        if mime.startswith('video/'):
            orig_ext  = mime.split('/')[1].split(';')[0]
            orig_path = os.path.join(out_dir, f'{name}_raw.{orig_ext}')
            mp4_path  = os.path.join(out_dir, f'{name}.mp4')
            with open(orig_path, 'wb') as f: f.write(data)
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', orig_path,
                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', mp4_path],
                capture_output=True, text=True)
            try: os.remove(orig_path)
            except: pass
            if r.returncode != 0 or not os.path.exists(mp4_path):
                raise RuntimeError(f'ffmpeg video convert failed: {r.stderr[-300:]}')
            return mp4_path, True

        img = Image.open(_io.BytesIO(data))
    elif os.path.exists(os.path.join(BASE_DIR, 'avatar.png')):
        img = Image.open(os.path.join(BASE_DIR, 'avatar.png'))
    else:
        raise ValueError('No avatar provided')

    if img.mode != 'RGB': img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > 720:
        s = 720 / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    path = os.path.join(out_dir, f'{name}.png')
    img.save(path, 'PNG')
    return path, False


def predetect_face_box(face_path, is_video):
    try:
        import json as _j
        r = subprocess.run(
            [sys.executable, 'detect_face.py', face_path, str(is_video)],
            cwd=WAV2LIP_DIR, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            box = _j.loads(r.stdout).get('box')
            if box:
                print(f'[upload] cached box: {box}')
                return box
        print(f'[upload] face detect: {r.stderr[:200]}')
    except Exception as e:
        print(f'[upload] face detect error: {e}')
    return None

# ── In-process Wav2Lip inference ──────────────────────────────────────────────

def _run_batch(img_list, mel_list, frame_list, coord_list):
    IMG_SIZE = 96
    ib = np.asarray(img_list, dtype=np.float32)
    mb = np.asarray(mel_list, dtype=np.float32)
    masked = ib.copy(); masked[:, IMG_SIZE//2:] = 0
    ib_in = np.concatenate((masked, ib), axis=3) / 255.
    mb_in = mb.reshape(len(mb), mb.shape[1], mb.shape[2], 1)

    with torch.no_grad():
        pred = WAV2LIP_MODEL(
            torch.FloatTensor(mb_in.transpose(0, 3, 1, 2)),
            torch.FloatTensor(ib_in.transpose(0, 3, 1, 2)),
        )
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

    out = []
    for p, f, (y1, y2, x1, x2) in zip(pred, frame_list, coord_list):
        fh, fw = y2-y1, x2-x1
        # Lanczos upscale → sharper than bilinear
        p = cv2.resize(p.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        # Unsharp mask to recover detail lost at 96×96
        blurred = cv2.GaussianBlur(p, (0, 0), 2.5)
        p = np.clip(cv2.addWeighted(p, 1.4, blurred, -0.4, 0), 0, 255).astype(np.uint8)
        # Feathered blend so paste boundary is invisible
        fe = max(4, min(fh, fw)//10)
        mask = np.ones((fh, fw), np.float32)
        mask[:fe,  :] *= np.linspace(0, 1, fe)[:, None]
        mask[-fe:, :] *= np.linspace(1, 0, fe)[:, None]
        mask[:, :fe]  *= np.linspace(0, 1, fe)[None, :]
        mask[:, -fe:] *= np.linspace(1, 0, fe)[None, :]
        m3 = mask[:, :, None]
        f[y1:y2, x1:x2] = (m3*p + (1-m3)*f[y1:y2, x1:x2]).astype(np.uint8)
        out.append(f)
    return out


def wav2lip_infer(face_path, wav_path, box, is_video):
    IMG_SIZE = 96
    BATCH    = 20
    MEL_STEP = 16

    TARGET_FPS = 10.0   # fixed rate keeps inference fast for both image and video
    if is_video:
        cap      = cv2.VideoCapture(face_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step     = max(1, round(orig_fps / TARGET_FPS))   # subsample to TARGET_FPS
        frames, idx = [], 0
        while True:
            ret, frm = cap.read()
            if not ret: break
            if idx % step == 0:
                frames.append(frm)
            idx += 1
        cap.release()
    else:
        frames = [cv2.imread(face_path)]
    fps = TARGET_FPS

    if not frames:
        raise ValueError('No frames from face source')

    wav = wav2lip_audio.load_wav(wav_path, 16000)
    mel = wav2lip_audio.melspectrogram(wav)
    if np.isnan(mel).any():
        raise ValueError('Mel spectrogram has NaN')

    mel_mult = 80.0 / fps
    mel_chunks, i = [], 0
    while True:
        s = int(i * mel_mult)
        if s + MEL_STEP > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1]-MEL_STEP:]); break
        mel_chunks.append(mel[:, s:s+MEL_STEP])
        i += 1

    frames    = frames[:len(mel_chunks)]
    is_static = len(frames) == 1

    if box and len(box) == 4:
        y1, y2, x1, x2 = box
        face_regions = [(frm[y1:y2, x1:x2].copy(), (y1,y2,x1,x2)) for frm in frames]
    else:
        import face_detection as fd_mod
        det   = fd_mod.FaceAlignment(fd_mod.LandmarksType._2D, flip_input=False, device='cpu')
        rgb   = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        preds = det.get_detections_for_batch(np.array([rgb]))
        del det
        if not preds or preds[0] is None:
            raise ValueError('No face detected')
        rx1, ry1, rx2, ry2 = [int(v) for v in preds[0]]
        ry2 = min(frames[0].shape[0], ry2+10)
        face_regions = [(frm[ry1:ry2, rx1:rx2].copy(), (ry1,ry2,rx1,rx2)) for frm in frames]

    out_frames = []
    ib, mb, fb, cb = [], [], [], []
    for i, m_chunk in enumerate(mel_chunks):
        idx  = 0 if is_static else i % len(frames)
        face, coords = face_regions[idx]
        ib.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        mb.append(m_chunk)
        fb.append(frames[idx].copy())
        cb.append(coords)
        if len(ib) >= BATCH:
            out_frames.extend(_run_batch(ib, mb, fb, cb))
            ib, mb, fb, cb = [], [], [], []
    if ib:
        out_frames.extend(_run_batch(ib, mb, fb, cb))

    return out_frames, fps

# ── Background lip-sync job ───────────────────────────────────────────────────

def run_lipsync_job(job_id, text, face_path, is_video, box, lang, rate_val):
    try:
        tmp      = tempfile.mkdtemp()
        voice    = VOICE_MAP.get(lang, 'en-US-JennyNeural')
        rate_pct = int((rate_val - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        audio_path = os.path.join(tmp, f'{job_id}_speech.mp3')

        # Full TTS audio
        r = subprocess.run(
            ['edge-tts', '--voice', voice, '--rate', rate_str,
             '--text', text, '--write-media', audio_path],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(audio_path):
            raise RuntimeError(r.stderr or 'edge-tts failed')

        # MP3 → 16 kHz mono WAV (full duration, so lip sync covers whole answer)
        wav_path = os.path.join(tmp, f'{job_id}.wav')
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', wav_path],
                       capture_output=True)
        if not os.path.exists(wav_path):
            raise RuntimeError('WAV conversion failed')

        # In-process inference on full audio — model already in RAM
        with infer_lock:
            out_frames, fps = wav2lip_infer(face_path, wav_path, box, is_video)

        # Write AVI
        fh, fw = out_frames[0].shape[:2]
        avi_path = os.path.join(tmp, f'{job_id}.avi')
        vw = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (fw, fh))
        for frm in out_frames: vw.write(frm)
        vw.release()

        # Mux full audio into MP4 — video and audio are now in perfect sync
        out_path = os.path.join(tmp, f'{job_id}_out.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-i', avi_path, '-i', audio_path,
             '-map', '0:v', '-map', '1:a',
             '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-c:a', 'aac',
             out_path],
            capture_output=True)
        if not os.path.exists(out_path):
            raise RuntimeError('ffmpeg mux failed')

        with open(out_path, 'rb') as f:
            video_b64 = 'data:video/mp4;base64,' + base64.b64encode(f.read()).decode()

        with jobs_lock:
            jobs[job_id] = {'status': 'done', 'videoUrl': video_b64}
        print(f'[job {job_id}] done')

    except Exception as e:
        print(traceback.format_exc())
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(e)}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_avatar():
    global stored_face
    if request.method == 'OPTIONS': return make_response('OK', 200)
    try:
        data       = request.get_json(force=True, silent=True) or {}
        avatar_b64 = data.get('avatar', '').strip()
        if not avatar_b64: return jsonify({'error': 'No avatar data'}), 400
        face_path, is_video = save_face_media(avatar_b64, AVATAR_DIR, 'current')
        box = predetect_face_box(face_path, is_video)
        stored_face = {'path': face_path, 'is_video': is_video, 'box': box}
        return jsonify({'ok': True, 'is_video': is_video, 'box': box})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/wav2p', methods=['GET', 'POST', 'OPTIONS'])
def wav2p():
    if request.method in ('OPTIONS', 'GET'): return make_response('OK', 200)
    try:
        data     = request.get_json(force=True, silent=True) or {}
        text     = data.get('text', 'Hello').strip() or 'Hello'
        lang     = data.get('lang', 'en-US')
        rate_val = float(data.get('rate', 1.0))

        if stored_face['path'] and os.path.exists(stored_face['path']):
            face_path, is_video, box = stored_face['path'], stored_face['is_video'], stored_face.get('box')
        else:
            try:
                face_path, is_video = save_face_media(
                    data.get('avatar','').strip(), tempfile.mkdtemp(), str(uuid.uuid4())[:8])
                box = None
            except Exception as e:
                return jsonify({'error': f'Media error: {e}'}), 400

        job_id = str(uuid.uuid4())[:8]
        with jobs_lock:
            jobs[job_id] = {'status': 'processing'}
            if len(jobs) > 20:
                del jobs[list(jobs.keys())[0]]

        threading.Thread(
            target=run_lipsync_job,
            args=(job_id, text, face_path, is_video, box, lang, rate_val),
            daemon=True).start()

        return jsonify({'job_id': job_id, 'status': 'processing'})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>', methods=['GET', 'OPTIONS'])
def job_status(job_id):
    if request.method == 'OPTIONS': return make_response('OK', 200)
    with jobs_lock:
        job = dict(jobs.get(job_id, {'status': 'not_found'}))
        if job.get('status') == 'done' and 'videoUrl' in job:
            jobs[job_id] = {'status': 'fetched'}
    return jsonify(job)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)),
            debug=False, use_reloader=False, threaded=True)
