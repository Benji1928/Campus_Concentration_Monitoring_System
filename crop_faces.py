"""
Run YOLO face detection on test_new/ and save 90%+ confidence face crops.

Usage (from project root):
    python crop_faces.py

Output:
    test_new_crops/<class>/<original_stem>_crop<n>.jpg
    One crop per detected face per image. Images with no confident face are skipped.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import cv2

from src.detectors.yolo_detector import YOLOFaceDetector

INPUT_DIR  = ROOT / "test_new"
OUTPUT_DIR = ROOT / "test_new_crops"
MODEL_PATH = ROOT / "models" / "face_detection.pt"
CONF       = 0.70


def crop_faces():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    detector = YOLOFaceDetector(str(MODEL_PATH), conf=CONF)

    total_images = 0
    total_crops  = 0
    skipped      = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Collect all image files, preserving subfolder structure
    image_paths = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print(f"[WARN] No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images. Running detection at conf={CONF}...\n")

    for image_path in sorted(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [SKIP] Could not read: {image_path.name}")
            skipped += 1
            continue

        total_images += 1
        detections = detector.detect(frame)

        if not detections:
            print(f"  [SKIP] No face ≥{CONF:.0%}: {image_path.relative_to(INPUT_DIR)}")
            skipped += 1
            continue

        # Mirror subfolder structure under OUTPUT_DIR
        rel_parent = image_path.parent.relative_to(INPUT_DIR)
        out_dir = OUTPUT_DIR / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        for n, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            suffix = f"_crop{n}" if len(detections) > 1 else ""
            out_path = out_dir / f"{image_path.stem}{suffix}.jpg"
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

    print(f"\nDone.")
    print(f"  Images processed : {total_images}")
    print(f"  Face crops saved : {total_crops}  →  {OUTPUT_DIR}")
    print(f"  Skipped          : {skipped}")


if __name__ == "__main__":
    crop_faces()
