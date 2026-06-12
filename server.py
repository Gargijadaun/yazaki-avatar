from flask import Flask, request, jsonify, make_response, send_from_directory
import subprocess, base64, tempfile, os, uuid, sys, threading, traceback, shutil, gc
from datetime import datetime, timezone
import numpy as np
import cv2

# ── MongoDB (optional — server runs fine without it) ──────────────────────────
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    _MONGO_URI  = os.environ.get('MONGODB_URI', '')
    _MONGO_DB   = os.environ.get('MONGODB_DB', 'yazaki_avatar')
    if _MONGO_URI:
        _mc = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=8000,
                          connectTimeoutMS=8000, socketTimeoutMS=10000)
        _mc.admin.command('ping')
        print(f'[startup] MongoDB ping OK, using db={_MONGO_DB}')
        _mdb   = _mc[_MONGO_DB]
        _chats = _mdb['chats']
        # Test write access before accepting traffic
        _chats.insert_one({'_startup_test': True}).inserted_id
        _chats.delete_one({'_startup_test': True})
        try:
            _chats.create_index([('session_id', ASCENDING), ('timestamp', DESCENDING)])
        except Exception:
            pass  # index creation is optional
        print('[startup] MongoDB write access confirmed')
    else:
        _chats = None
        print('[startup] MONGODB_URI not set — chat saving disabled')
except Exception as _me:
    print(f'[startup] MongoDB error: {_me}')
    _chats = None

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, 'Wav2Lip')
CHECKPOINT  = os.path.join(WAV2LIP_DIR, 'checkpoints', 'Wav2Lip-SD-GAN.pt')
AVATAR_DIR  = os.path.join(BASE_DIR, 'uploads')
TEMP_DIR    = os.path.join(WAV2LIP_DIR, 'temp')
os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# ── Wav2Lip imports (wrapped so server starts even if torch is missing) ────────
sys.path.insert(0, WAV2LIP_DIR)
try:
    import torch
    from models import Wav2Lip as Wav2LipModel
    import audio as wav2lip_audio
    _IMPORTS_OK = True
except Exception as _ie:
    print(f'[startup] Wav2Lip import error: {_ie}')
    _IMPORTS_OK = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ── Default avatar: use new.mp4 if no upload exists yet ───────────────────────
stored_face = {'path': None, 'is_video': False, 'box': None}
for _ext, _vid in [('mp4', True), ('png', False)]:
    _p = os.path.join(AVATAR_DIR, f'current.{_ext}')
    if os.path.exists(_p):
        stored_face = {'path': _p, 'is_video': _vid, 'box': None}
        print(f'[startup] restored face: {_p}')
        break

if not stored_face['path']:
    for _default in ['new.mp4', 'avatar.png']:
        _src = os.path.join(BASE_DIR, _default)
        if os.path.exists(_src):
            _is_vid = _default.endswith('.mp4')
            _dst = os.path.join(AVATAR_DIR, 'current.' + ('mp4' if _is_vid else 'png'))
            shutil.copy2(_src, _dst)
            stored_face = {'path': _dst, 'is_video': _is_vid, 'box': None}
            print(f'[startup] default face set from {_default}')
            break

jobs       = {}
jobs_lock  = threading.Lock()
infer_lock = threading.Lock()

VOICE_MAP = {
    'en-US': 'en-US-JennyNeural', 'en-GB': 'en-GB-SoniaNeural',
    'hi-IN': 'hi-IN-SwaraNeural', 'es-ES': 'es-ES-ElviraNeural',
    'fr-FR': 'fr-FR-DeniseNeural', 'de-DE': 'de-DE-KatjaNeural',
    'ja-JP': 'ja-JP-NanamiNeural',
}

# ── Load model ONCE at startup ─────────────────────────────────────────────────
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

WAV2LIP_MODEL = None
if _IMPORTS_OK:
    try:
        WAV2LIP_MODEL = _load_checkpoint()
    except Exception as _e:
        print(f'[startup] Model load failed: {_e}')

# ── CORS ───────────────────────────────────────────────────────────────────────
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return r

@app.after_request
def after(r): return cors(r)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/config')
def config():
    return jsonify({'groq_key': os.environ.get('GROQ_API_KEY', '')})

@app.route('/save-chat', methods=['POST', 'OPTIONS'])
def save_chat():
    if request.method == 'OPTIONS': return make_response('OK', 200)
    if not _chats:
        return jsonify({'ok': False, 'error': 'MongoDB not configured'}), 503
    try:
        data = request.get_json(force=True, silent=True) or {}
        doc = {
            'session_id': str(data.get('session_id', ''))[:64],
            'timestamp':  datetime.now(timezone.utc),
            'user':       str(data.get('user', ''))[:2000],
            'ai':         str(data.get('ai',   ''))[:2000],
            'lang':       str(data.get('lang', 'en-US'))[:16],
        }
        result = _chats.insert_one(doc)
        return jsonify({'ok': True, 'id': str(result.inserted_id)})
    except Exception as e:
        print(f'[save-chat] error: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def chat_history():
    if not _chats:
        return jsonify({'ok': False, 'chats': []})
    try:
        session_id = request.args.get('session_id', '').strip()
        limit      = min(int(request.args.get('limit', 50)), 200)
        query      = {'session_id': session_id} if session_id else {}
        docs = list(_chats.find(query, {'_id': 0}).sort('timestamp', ASCENDING).limit(limit))
        for d in docs:
            if isinstance(d.get('timestamp'), datetime):
                d['timestamp'] = d['timestamp'].isoformat()
        return jsonify({'ok': True, 'chats': docs})
    except Exception as e:
        print(f'[chat-history] error: {e}')
        return jsonify({'ok': False, 'chats': [], 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'ok': True, 'model': WAV2LIP_MODEL is not None,
                    'face': stored_face['path']})

@app.route('/<path:filename>')
def static_files(filename):
    blocked = {'.py', '.env', '.yaml', '.yml', '.sh'}
    if os.path.splitext(filename)[1].lower() in blocked or '..' in filename:
        return '', 404
    try:
        return send_from_directory(BASE_DIR, filename)
    except Exception:
        return '', 404

# ── Media helpers ──────────────────────────────────────────────────────────────
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
                raise RuntimeError(f'ffmpeg convert failed: {r.stderr[-300:]}')
            return mp4_path, True

        img = Image.open(_io.BytesIO(data))
    elif os.path.exists(os.path.join(BASE_DIR, 'avatar.png')):
        img = Image.open(os.path.join(BASE_DIR, 'avatar.png'))
    else:
        raise ValueError('No avatar provided')

    if img.mode != 'RGB': img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > 480:
        s = 480 / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    path = os.path.join(out_dir, f'{name}.png')
    img.save(path, 'PNG')
    return path, False


# ── In-process inference ───────────────────────────────────────────────────────
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
        p = cv2.resize(p.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        blurred = cv2.GaussianBlur(p, (0, 0), 2.5)
        p = np.clip(cv2.addWeighted(p, 1.4, blurred, -0.4, 0), 0, 255).astype(np.uint8)
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


def _shrink(frame, max_dim=480):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    s = max_dim / max(h, w)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)


def wav2lip_infer(face_path, wav_path, box, is_video, out_avi_path):
    """Run inference, write frames directly to out_avi_path (no frame list in RAM)."""
    IMG_SIZE = 96; BATCH = 8; MEL_STEP = 16
    TARGET_FPS = 10.0; MAX_FRAMES = 100  # cap at 10 s to stay within 2 GB RAM

    if is_video:
        cap = cv2.VideoCapture(face_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(orig_fps / TARGET_FPS))
        frames, idx = [], 0
        while len(frames) < MAX_FRAMES:
            ret, frm = cap.read()
            if not ret: break
            if idx % step == 0:
                frames.append(_shrink(frm))
            idx += 1
        cap.release()
    else:
        frm = cv2.imread(face_path)
        frames = [_shrink(frm)]

    if not frames: raise ValueError('No frames from face source')

    wav = wav2lip_audio.load_wav(wav_path, 16000)
    mel = wav2lip_audio.melspectrogram(wav)
    if np.isnan(mel).any(): raise ValueError('Mel has NaN')

    mel_mult = 80.0 / TARGET_FPS
    mel_chunks, i = [], 0
    while len(mel_chunks) < MAX_FRAMES:
        s = int(i * mel_mult)
        if s + MEL_STEP > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1]-MEL_STEP:]); break
        mel_chunks.append(mel[:, s:s+MEL_STEP]); i += 1

    n = min(len(frames), len(mel_chunks))
    mel_chunks = mel_chunks[:n]
    is_static = len(frames) == 1

    if box and len(box) == 4:
        y1, y2, x1, x2 = [int(v) for v in box]
        face_regions = [(frm[y1:y2, x1:x2].copy(), (y1,y2,x1,x2)) for frm in frames]
    else:
        import face_detection as fd_mod
        det = fd_mod.FaceAlignment(fd_mod.LandmarksType._2D, flip_input=False, device='cpu')
        rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        preds = det.get_detections_for_batch(np.array([rgb]))
        del det; gc.collect()
        if not preds or preds[0] is None: raise ValueError('No face detected')
        rx1, ry1, rx2, ry2 = [int(v) for v in preds[0]]
        ry2 = min(frames[0].shape[0], ry2+10)
        face_regions = [(frm[ry1:ry2, rx1:rx2].copy(), (ry1,ry2,rx1,rx2)) for frm in frames]

    fh, fw = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_avi_path, cv2.VideoWriter_fourcc(*'DIVX'), TARGET_FPS, (fw, fh))

    ib, mb, fb, cb = [], [], [], []
    for i, m_chunk in enumerate(mel_chunks):
        fi = 0 if is_static else i % len(frames)
        face, coords = face_regions[fi]
        ib.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        mb.append(m_chunk); fb.append(frames[fi].copy()); cb.append(coords)
        if len(ib) >= BATCH:
            for frm in _run_batch(ib, mb, fb, cb): vw.write(frm)
            ib, mb, fb, cb = [], [], [], []
    if ib:
        for frm in _run_batch(ib, mb, fb, cb): vw.write(frm)

    vw.release()
    return TARGET_FPS

# ── Background lip-sync job ────────────────────────────────────────────────────
def run_lipsync_job(job_id, text, face_path, is_video, box, lang, rate_val):
    try:
        if not WAV2LIP_MODEL:
            raise RuntimeError('Wav2Lip model not loaded — check server logs')

        tmp      = tempfile.mkdtemp()
        voice    = VOICE_MAP.get(lang, 'en-US-JennyNeural')
        rate_pct = int((rate_val - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        audio_path = os.path.join(tmp, f'{job_id}_speech.mp3')

        r = subprocess.run(
            ['edge-tts', '--voice', voice, '--rate', rate_str,
             '--text', text, '--write-media', audio_path],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(audio_path):
            raise RuntimeError(r.stderr or 'edge-tts failed')

        wav_path = os.path.join(tmp, f'{job_id}.wav')
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', wav_path],
                       capture_output=True)
        if not os.path.exists(wav_path):
            raise RuntimeError('WAV conversion failed')

        # Trim text so TTS stays under ~10 s (prevent OOM from very long responses)
        text = text[:300]

        avi_path = os.path.join(tmp, f'{job_id}.avi')
        with infer_lock:
            wav2lip_infer(face_path, wav_path, box, is_video, avi_path)
        gc.collect()

        out_path = os.path.join(tmp, f'{job_id}_out.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-i', avi_path, '-i', audio_path,
             '-map', '0:v', '-map', '1:a',
             '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-c:a', 'aac',
             out_path], capture_output=True)
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


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_avatar():
    global stored_face
    if request.method == 'OPTIONS': return make_response('OK', 200)
    try:
        data       = request.get_json(force=True, silent=True) or {}
        avatar_b64 = data.get('avatar', '').strip()
        if not avatar_b64: return jsonify({'error': 'No avatar data'}), 400
        face_path, is_video = save_face_media(avatar_b64, AVATAR_DIR, 'current')
        stored_face = {'path': face_path, 'is_video': is_video, 'box': None}
        return jsonify({'ok': True, 'is_video': is_video})
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
    port = int(os.environ.get('PORT', 5001))
    print(f'[startup] Listening on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
