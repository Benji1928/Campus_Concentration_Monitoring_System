"""
Data collection tool for labeling attention states.

Usage:
    python src/classifiers/landmark_pipeline/collector.py              # default camera 0
    python src/classifiers/landmark_pipeline/collector.py --camera 1   # alternate camera

Controls (while window is focused):
    Hold  1  — record ATTENTIVE  samples
    Hold  2  — record SLEEPY     samples
    Hold  3  — record DISTRACTED samples
    Press Q  — quit and save CSV
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.classifiers.landmark_pipeline.face_mesh import FaceMesh
from src.classifiers.landmark_pipeline.feature_extractor import (
    FeatureExtractor, FEATURE_COLS, RIGHT_EYE, LEFT_EYE, MOUTH
)

DATA_PATH = ROOT / 'data' / 'labeled_features.csv'
LABEL_NAMES = {1: 'ATTENTIVE', 2: 'SLEEPY', 3: 'DISTRACTED'}
LABEL_COLORS = {
    'ATTENTIVE':  (0, 200, 0),
    'SLEEPY':     (0, 165, 255),
    'DISTRACTED': (0, 0, 230),
}
_KEY_LANDMARKS = set(RIGHT_EYE + LEFT_EYE + MOUTH)


def _draw_ui(frame, features, active_label, counts):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if active_label:
        name = LABEL_NAMES[active_label]
        col = LABEL_COLORS[name]
        cv2.putText(frame, f'Recording: {name}',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    else:
        cv2.putText(frame, 'Hold 1/2/3 to record  |  Q to quit',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1)

    count_str = '  '.join(
        f"{LABEL_NAMES[k]}: {counts[k]}" for k in sorted(counts)
    )
    cv2.putText(frame, count_str, (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera {args.camera}')
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera {args.camera}: {w}x{h}')
    print('Hold 1=ATTENTIVE  2=SLEEPY  3=DISTRACTED  |  Q to quit and save')

    mesh = FaceMesh()
    extractor = FeatureExtractor(w, h)
    rows = []
    counts = {1: 0, 2: 0, 3: 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        active_label = None
        for k, label_id in [(ord('1'), 1), (ord('2'), 2), (ord('3'), 3)]:
            if key == k:
                active_label = label_id
                break

        landmarks = mesh.process(frame)
        features = None

        if landmarks:
            features = extractor.extract(landmarks)
            for i in _KEY_LANDMARKS:
                x, y = int(landmarks[i][0]), int(landmarks[i][1])
                cv2.circle(frame, (x, y), 2, (0, 230, 230), -1)

            if active_label is not None:
                row = {col: features[col] for col in FEATURE_COLS}
                row['label'] = active_label - 1  # 0-indexed: 0=ATTENTIVE, 1=SLEEPY, 2=DISTRACTED
                rows.append(row)
                counts[active_label] += 1

        _draw_ui(frame, features, active_label, counts)
        cv2.imshow('Data Collector  [1=Attentive  2=Sleepy  3=Distracted  Q=Quit]', frame)

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()

    if not rows:
        print('No samples collected — nothing saved.')
        return

    DATA_PATH.parent.mkdir(exist_ok=True)
    write_header = not DATA_PATH.exists()
    with DATA_PATH.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS + ['label'])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    total = sum(counts.values())
    print(f'\nSaved {total} samples to {DATA_PATH}')
    for k, name in LABEL_NAMES.items():
        print(f'  {name}: {counts[k]}')


if __name__ == '__main__':
    main()
