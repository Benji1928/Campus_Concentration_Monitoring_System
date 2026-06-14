from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

LABEL_NAMES = {0: "ATTENTIVE", 1: "DROWSY", 2: "DISTRACTED"}


@dataclass
class ClassifierResult:
    label: str
    label_int: int
    probabilities: np.ndarray  # shape (3,) — [attentive, drowsy, distracted]
    confidence: float


class BaseAttentionClassifier(ABC):
    name: str
    needs_landmarks: bool  # True -> pipeline must supply features dict before predict

    @abstractmethod
    def predict(self, face_crop: np.ndarray, features: dict | None) -> ClassifierResult:
        """
        face_crop: BGR ndarray of the detected face region (any size)
        features:  dict from FeatureExtractor.extract(), or None for DL classifiers
        """
