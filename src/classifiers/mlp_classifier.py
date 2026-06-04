import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.landmark.feature_extractor import FEATURE_COLS

LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}


class MLPAttentionClassifier:
    def __init__(self):
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
        )
        self._trained = False

    def _to_array(self, features: dict) -> np.ndarray:
        return np.array([[features[c] for c in FEATURE_COLS]])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y)
        self._trained = True

    def predict(self, features: dict) -> tuple:
        """Returns (label_str, label_int)."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        x = self._scaler.transform(self._to_array(features))
        label_int = int(self._model.predict(x)[0])
        return LABEL_NAMES[label_int], label_int

    def predict_proba(self, features: dict) -> np.ndarray:
        """Returns probability array [attentive, sleepy, distracted]."""
        x = self._scaler.transform(self._to_array(features))
        return self._model.predict_proba(x)[0]

    def save(self, model_path: str, scaler_path: str):
        joblib.dump(self._model, model_path)
        joblib.dump(self._scaler, scaler_path)

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'MLPAttentionClassifier':
        obj = cls()
        obj._model = joblib.load(model_path)
        obj._scaler = joblib.load(scaler_path)
        obj._trained = True
        return obj
