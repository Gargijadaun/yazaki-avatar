"""
Run once at upload time to cache the face bounding box.
Passing --box to inference.py skips S3FD detection on every request (the main bottleneck).
"""
import sys, os, json
import cv2, numpy as np
import face_detection

def detect(media_path, is_video):
    if is_video:
        cap = cv2.VideoCapture(media_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
    else:
        frame = cv2.imread(media_path)
        if frame is None:
            return None

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device='cpu')
    preds = detector.get_detections_for_batch(np.array([frame_rgb]))
    del detector

    if not preds or preds[0] is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in preds[0]]
    # Match Wav2Lip default pads (top=0, bottom=10, left=0, right=0)
    y1 = max(0, y1)
    y2 = min(h, y2 + 10)
    x1 = max(0, x1)
    x2 = min(w, x2)
    return [y1, y2, x1, x2]   # format: top, bottom, left, right

if __name__ == '__main__':
    media_path = sys.argv[1]
    is_video   = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    try:
        box = detect(media_path, is_video)
        print(json.dumps({'box': box}))
    except Exception as e:
        print(json.dumps({'box': None, 'error': str(e)}))
