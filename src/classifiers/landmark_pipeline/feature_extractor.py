import time
from collections import deque

import cv2
import numpy as np

# ── Landmark index sets ───────────────────────────────────────────────────────
# 6-point EAR indices per eye: [outer, top-outer, top-inner, inner, bot-inner, bot-outer]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# 6-point MAR indices (mirrors EAR geometry): [left, top-l, top-r, right, bot-r, bot-l]
MOUTH = [61, 82, 312, 291, 317, 87]

# 6 landmarks used for solvePnP head-pose estimation
HEAD_POSE_LM = [1, 152, 263, 33, 287, 57]

# Ordered list of feature names — order matters for MLP input
FEATURE_COLS = [
    'ear_left', 'ear_right', 'ear_avg',
    'mar',
    'pitch', 'yaw', 'roll',
    'perclos',
    'blink_rate',
]

# ── 3-D face model (mm, face-centred) matching HEAD_POSE_LM order ─────────────
_MODEL_3D = np.array([
    [0.0,    0.0,    0.0],      # nose tip       (1)
    [0.0,  -330.0,  -65.0],     # chin           (152)
    [-225.0, 170.0, -135.0],    # left eye outer (263)
    [225.0,  170.0, -135.0],    # right eye outer(33)
    [-150.0,-150.0, -125.0],    # left mouth     (287)
    [150.0, -150.0, -125.0],    # right mouth    (57)
], dtype=np.float64)

EAR_CLOSED_THRESHOLD = 0.22
PERCLOS_WINDOW = 90   # frames (~3 s at 30 fps)
BLINK_RATE_WINDOW = 60.0  # seconds


def _aspect_ratio(all_lm: np.ndarray, indices: list) -> float:
    """Generic 6-point aspect ratio (EAR / MAR formula)."""
    pts = np.array([[all_lm[i, 0], all_lm[i, 1]] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def _head_pose(all_lm: np.ndarray, cam_matrix: np.ndarray, dist: np.ndarray):
    """Returns (pitch, yaw, roll) in degrees, or (0, 0, 0) on failure."""
    img_pts = np.array(
        [[all_lm[i, 0], all_lm[i, 1]] for i in HEAD_POSE_LM], dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_3D, img_pts, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0
    return pitch, yaw, roll


class FeatureExtractor:
    def __init__(self, frame_w: int, frame_h: int, perclos_window: int = PERCLOS_WINDOW):
        self._cam_matrix = np.array([
            [frame_w, 0,       frame_w / 2],
            [0,       frame_w, frame_h / 2],
            [0,       0,       1],
        ], dtype=np.float64)
        self._dist = np.zeros((4, 1))

        self._perclos_buf = deque(maxlen=perclos_window)
        self._blink_times: deque = deque()
        self._eye_was_closed = False
        self._start = time.time()

    def extract(self, landmarks: list) -> dict:
        """
        landmarks: list of (x_px, y_px, z_norm) returned by FaceMesh.process().
        Returns a dict with keys matching FEATURE_COLS.
        """
        lm = np.array(landmarks)  # (468, 3)

        ear_l = _aspect_ratio(lm, LEFT_EYE)
        ear_r = _aspect_ratio(lm, RIGHT_EYE)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = _aspect_ratio(lm, MOUTH)

        # PERCLOS — rolling fraction of closed-eye frames
        self._perclos_buf.append(1 if ear_avg < EAR_CLOSED_THRESHOLD else 0)
        perclos = sum(self._perclos_buf) / len(self._perclos_buf)

        # Blink rate (blinks per minute over a rolling window)
        now = time.time()
        is_closed = ear_avg < EAR_CLOSED_THRESHOLD
        if not is_closed and self._eye_was_closed:
            self._blink_times.append(now)
        self._eye_was_closed = is_closed

        cutoff = now - BLINK_RATE_WINDOW
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        elapsed = min(now - self._start, BLINK_RATE_WINDOW)
        blink_rate = (len(self._blink_times) / elapsed * 60.0) if elapsed > 0 else 0.0

        pitch, yaw, roll = _head_pose(lm, self._cam_matrix, self._dist)

        return {
            'ear_left':   ear_l,
            'ear_right':  ear_r,
            'ear_avg':    ear_avg,
            'mar':        mar,
            'pitch':      pitch,
            'yaw':        yaw,
            'roll':       roll,
            'perclos':    perclos,
            'blink_rate': blink_rate,
        }

    def reset(self):
        self._perclos_buf.clear()
        self._blink_times.clear()
        self._eye_was_closed = False
        self._start = time.time()
