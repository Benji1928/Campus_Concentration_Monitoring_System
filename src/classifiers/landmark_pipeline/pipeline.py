"""
Real-time attention monitoring pipeline.
Usage:
    python src/classifiers/landmark_pipeline/pipeline.py              # rule-based only
    python src/classifiers/landmark_pipeline/pipeline.py --mlp        # rule-based + MLP side-by-side
    python src/classifiers/landmark_pipeline/pipeline.py --camera 1   # alternate camera index
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.classifiers.landmark_pipeline.face_mesh import FaceMesh
from src.classifiers.landmark_pipeline.feature_extractor import (
    FeatureExtractor, RIGHT_EYE, LEFT_EYE, MOUTH
)
from src.classifiers.landmark_pipeline.rule_based import RuleBasedClassifier
from src.classifiers.landmark_pipeline.mlp_classifier import MLPAttentionClassifier

MODELS_DIR = ROOT / 'models'

COLORS = {
    'ATTENTIVE':  (0, 200, 0),
    'SLEEPY':     (0, 165, 255),
    'DISTRACTED': (0, 0, 230),
}
NO_FACE_COLOR = (120, 120, 120)

_KEY_LANDMARKS = set(RIGHT_EYE + LEFT_EYE + MOUTH)


def _draw_landmarks(frame, landmarks):
    for i in _KEY_LANDMARKS:
        x, y = int(landmarks[i][0]), int(landmarks[i][1])
        cv2.circle(frame, (x, y), 2, (0, 230, 230), -1)


def _draw_hud(frame, features, rule_label, mlp_label=None):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    bar_h = 95 if mlp_label else 55
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    rule_col = COLORS.get(rule_label, NO_FACE_COLOR)
    cv2.putText(frame, f'Rule: {rule_label}',
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, rule_col, 2)

    if mlp_label:
        mlp_col = COLORS.get(mlp_label, NO_FACE_COLOR)
        cv2.putText(frame, f'MLP:  {mlp_label}',
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, mlp_col, 2)

    if features:
        info = (
            f"EAR {features['ear_avg']:.2f}  "
            f"MAR {features['mar']:.2f}  "
            f"Yaw {features['yaw']:+.1f}  "
            f"Pitch {features['pitch']:+.1f}  "
            f"PERCLOS {features['perclos']:.2f}  "
            f"Blinks/min {features['blink_rate']:.1f}"
        )
        cv2.putText(frame, info, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    else:
        cv2.putText(frame, 'No face detected', (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, NO_FACE_COLOR, 1)


def main():
    parser = argparse.ArgumentParser(description='Attention monitoring pipeline')
    parser.add_argument('--mlp', action='store_true', help='Enable MLP classifier')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    mlp = None
    if args.mlp:
        model_path  = MODELS_DIR / 'mlp_model.pkl'
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if not model_path.exists():
            print(f'MLP model not found at {model_path}.')
            print('Run `python src/training/train.py` first, then re-launch with --mlp.')
            sys.exit(1)
        mlp = MLPAttentionClassifier.load(str(model_path), str(scaler_path))
        print('MLP loaded successfully.')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera {args.camera}')
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera: {w}x{h}  |  Press Q to quit')

    mesh      = FaceMesh()
    extractor = FeatureExtractor(w, h)
    rule_clf  = RuleBasedClassifier()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = mesh.process(frame)
        features   = None
        rule_label = 'NO FACE'
        mlp_label  = None

        if landmarks:
            features   = extractor.extract(landmarks)
            rule_label, _ = rule_clf.predict(features)
            if mlp:
                mlp_label, _ = mlp.predict(features)
            _draw_landmarks(frame, landmarks)

        _draw_hud(frame, features, rule_label, mlp_label)
        cv2.imshow('Attention Monitor  [Q to quit]', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
