"""
Orchestrates: YOLOv8 face detection → optional landmark extraction → classifier.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.classifiers.base import BaseAttentionClassifier, ClassifierResult, LABEL_NAMES
from src.classifiers.landmark_pipeline.rule_based import RuleBasedClassifier
from src.classifiers.landmark_pipeline.mlp_classifier import MLPAttentionClassifier
from src.classifiers.dl_classifier import MobileNetV3Classifier, EfficientNetV2Classifier, DeiTTinyClassifier, MobileViTClassifier
from src.detectors.yolo_detector import YOLOFaceDetector, FaceDetection
from src.classifiers.landmark_pipeline.face_mesh import FaceMesh
from src.classifiers.landmark_pipeline.feature_extractor import FeatureExtractor

MODELS_DIR = ROOT / "models"


@dataclass
class PipelineResult:
    faces: list = field(default_factory=list)   # list[FaceDetection]
    classifier_result: ClassifierResult | None = None
    features: dict | None = None
    active_classifier_name: str = ""


# ── Adapters for existing classifiers ────────────────────────────────────────

class _RuleBasedAdapter(BaseAttentionClassifier):
    name = "Rule-based"
    needs_landmarks = True

    def __init__(self):
        self._clf = RuleBasedClassifier()

    def predict(self, face_crop: np.ndarray, features: dict | None) -> ClassifierResult:
        label, label_int = self._clf.predict(features)
        probs = np.zeros(3, dtype=np.float32)
        probs[label_int] = 1.0
        return ClassifierResult(
            label=label, label_int=label_int, probabilities=probs, confidence=1.0
        )


class _MLPAdapter(BaseAttentionClassifier):
    name = "MLP"
    needs_landmarks = True

    def __init__(self, mlp: MLPAttentionClassifier):
        self._clf = mlp

    def predict(self, face_crop: np.ndarray, features: dict | None) -> ClassifierResult:
        label, label_int = self._clf.predict(features)
        probs = self._clf.predict_proba(features).astype(np.float32)
        return ClassifierResult(
            label=label, label_int=label_int, probabilities=probs,
            confidence=float(probs[label_int]),
        )


# ── Pipeline ─────────────────────────────────────────────────────────────────

class InferencePipeline:
    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._detector = YOLOFaceDetector(str(MODELS_DIR / "face_detection.pt"), conf=0.5)
        self._face_mesh = FaceMesh()
        self._extractor = FeatureExtractor(frame_w, frame_h)
        self._clf: BaseAttentionClassifier | None = None

    def set_classifier(self, clf: BaseAttentionClassifier) -> None:
        self._clf = clf

    def reset(self) -> None:
        self._extractor.reset()

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        faces = self._detector.detect(frame)

        if not faces or self._clf is None:
            return PipelineResult(faces=faces)

        # Use the largest detected face (by area)
        primary = max(faces, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        x1, y1, x2, y2 = primary.bbox

        features = None
        if self._clf.needs_landmarks:
            landmarks = self._face_mesh.process(frame)
            if landmarks is not None:
                features = self._extractor.extract(landmarks)
            else:
                return PipelineResult(faces=faces, active_classifier_name=self._clf.name)

        # Crop face with a small padding, clamped to frame bounds
        h, w = frame.shape[:2]
        pad = max(10, int((x2 - x1) * 0.1))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        face_crop = frame[cy1:cy2, cx1:cx2]

        try:
            result = self._clf.predict(face_crop, features)
        except Exception:
            result = None

        return PipelineResult(
            faces=faces,
            classifier_result=result,
            features=features,
            active_classifier_name=self._clf.name,
        )

    def close(self) -> None:
        self._face_mesh.close()


# ── Classifier registry ───────────────────────────────────────────────────────

def build_classifier_registry(models_dir: Path = MODELS_DIR) -> dict[str, Callable[[], BaseAttentionClassifier]]:
    registry: dict[str, Callable[[], BaseAttentionClassifier]] = {
        "Rule-based": _RuleBasedAdapter,
    }

    mlp_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    if mlp_path.exists() and scaler_path.exists():
        registry["MLP"] = lambda: _MLPAdapter(
            MLPAttentionClassifier.load(str(mlp_path), str(scaler_path))
        )

    mobilenet_path = models_dir / "best_mobilenetv3_with.pth"
    if mobilenet_path.exists():
        registry["MobileNetV3"] = lambda: MobileNetV3Classifier(str(mobilenet_path))

    effnet_path = models_dir / "best_efficientnetv2_with.pth"
    if effnet_path.exists():
        registry["EfficientNetV2"] = lambda: EfficientNetV2Classifier(str(effnet_path))

    deit_path = models_dir / "best_deit_tiny_with.pth"
    if deit_path.exists():
        registry["DeiT-Tiny"] = lambda: DeiTTinyClassifier(str(deit_path))

    mobilevit_path = models_dir / "best_mobilevit_xxs_with.pth"
    if mobilevit_path.exists():
        registry["MobileViT-XXS"] = lambda: MobileViTClassifier(str(mobilevit_path))

    return registry
