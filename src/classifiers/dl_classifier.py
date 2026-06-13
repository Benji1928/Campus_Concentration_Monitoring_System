import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

from src.classifiers.base import BaseAttentionClassifier, ClassifierResult, LABEL_NAMES

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

NUM_CLASSES = 3
_HIDDEN = 128


def _build_timm_model(model_name: str) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, _HIDDEN),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(_HIDDEN, NUM_CLASSES),
    )
    return model


def _load_weights(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                model.load_state_dict(checkpoint[key])
                return model
        model.load_state_dict(checkpoint)
        return model
    raise ValueError(f"Unrecognised checkpoint format: {type(checkpoint)}")


class MobileNetV3Classifier(BaseAttentionClassifier):
    name = "MobileNetV3"
    needs_landmarks = False

    def __init__(self, model_path: str):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_timm_model("mobilenetv3_large_100")
        self._model = _load_weights(model, model_path, self._device).to(self._device)
        self._model.eval()

    def predict(self, face_crop: np.ndarray, features: dict | None) -> ClassifierResult:
        rgb = face_crop[:, :, ::-1].copy()  # BGR → RGB
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        label_int = int(probs.argmax())
        return ClassifierResult(
            label=LABEL_NAMES[label_int],
            label_int=label_int,
            probabilities=probs,
            confidence=float(probs[label_int]),
        )


class EfficientNetV2Classifier(BaseAttentionClassifier):
    name = "EfficientNetV2"
    needs_landmarks = False

    def __init__(self, model_path: str):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_timm_model("tf_efficientnetv2_b0")
        self._model = _load_weights(model, model_path, self._device).to(self._device)
        self._model.eval()

    def predict(self, face_crop: np.ndarray, features: dict | None) -> ClassifierResult:
        rgb = face_crop[:, :, ::-1].copy()  # BGR → RGB
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        label_int = int(probs.argmax())
        return ClassifierResult(
            label=LABEL_NAMES[label_int],
            label_int=label_int,
            probabilities=probs,
            confidence=float(probs[label_int]),
        )
