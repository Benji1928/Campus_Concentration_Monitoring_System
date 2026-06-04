import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"


def _ensure_model():
    if _MODEL_PATH.exists():
        return
    _MODEL_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading face landmark model (~30 MB) to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Model downloaded.")


class FaceMesh:
    def __init__(self, max_faces=1, det_conf=0.5, presence_conf=0.5, track_conf=0.5):
        _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_faces=max_faces,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Returns list of (x_px, y_px, z_norm) for the primary face, or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return [(lm.x * w, lm.y * h, lm.z) for lm in result.face_landmarks[0]]

    def close(self):
        self._landmarker.close()
