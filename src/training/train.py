"""
Train the MLP classifier from collected CSV data.
Usage: python src/training/train.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.landmark.feature_extractor import FEATURE_COLS
from src.classifiers.mlp_classifier import MLPAttentionClassifier

DATA_PATH  = ROOT / 'data' / 'labeled_features.csv'
MODELS_DIR = ROOT / 'models'
LABEL_NAMES = ['ATTENTIVE', 'SLEEPY', 'DISTRACTED']


def main():
    if not DATA_PATH.exists():
        print(f'No data at {DATA_PATH}. Run src/data/collector.py first.')
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f'Loaded {len(df)} samples')
    print(df['label'].value_counts().to_string())
    print()

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPAttentionClassifier()
    clf.fit(X_train, y_train)

    X_test_s = clf._scaler.transform(X_test)
    y_pred   = clf._model.predict(X_test_s)

    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    MODELS_DIR.mkdir(exist_ok=True)
    clf.save(
        str(MODELS_DIR / 'mlp_model.pkl'),
        str(MODELS_DIR / 'scaler.pkl'),
    )
    print(f'Model saved to {MODELS_DIR}/')


if __name__ == '__main__':
    main()
