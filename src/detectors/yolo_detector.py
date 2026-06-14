from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class FaceDetection:
    bbox: tuple  # (x1, y1, x2, y2) int pixels
    confidence: float


class YOLOFaceDetector:
    def __init__(self, model_path: str, conf: float = 0.5):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = YOLO(model_path)
        self._model.to(self._device)
        self._conf = conf

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        results = self._model(frame, conf=self._conf, verbose=False, device=self._device)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(FaceDetection(bbox=(x1, y1, x2, y2), confidence=float(box.conf[0])))
        return detections
