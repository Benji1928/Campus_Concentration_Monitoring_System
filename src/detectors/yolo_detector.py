from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class FaceDetection:
    bbox: tuple  # (x1, y1, x2, y2) int pixels
    confidence: float


class YOLOFaceDetector:
    def __init__(self, model_path: str, conf: float = 0.5):
        self._model = YOLO(model_path)
        self._conf = conf

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        results = self._model(frame, conf=self._conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(FaceDetection(bbox=(x1, y1, x2, y2), confidence=float(box.conf[0])))
        return detections
