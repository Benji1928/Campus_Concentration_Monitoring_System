"""
Face counter dashboard — Tkinter display (avoids opencv-python-headless GUI issues).
Uses a YOLOv8 model to detect faces in the webcam feed and shows a live counter.
Run: python src/face_counter.py
     python src/face_counter.py --model models/face_detection.pt --conf 0.4
"""
import argparse
import threading
import time
from collections import deque
from pathlib import Path
from tkinter import Tk, Label, Canvas, StringVar, Frame
from PIL import Image, ImageTk, ImageDraw, ImageFont

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "face_detection.pt"

# Colours (RGB for PIL)
COL_BOX    = (0, 220, 80)
COL_PANEL  = (20, 20, 20)
COL_WHITE  = (240, 240, 240)
COL_YELLOW = (240, 220, 40)
COL_GREEN  = (80, 220, 80)


def draw_overlay(frame_bgr, boxes_scores: list, face_count: int,
                 fps: float, peak: int, session_s: float) -> Image.Image:
    """Draw bounding boxes + dashboard panel; returns a PIL Image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # --- Bounding boxes ---
    for (x1, y1, x2, y2), score in boxes_scores:
        draw.rectangle([x1, y1, x2, y2], outline=COL_BOX, width=2)
        draw.text((x1 + 4, y1 - 18), f"{score:.2f}", fill=COL_BOX)

    # --- Stats panel (top-right) ---
    pw, ph = 260, 180
    px, py = w - pw - 12, 12
    draw.rectangle([px, py, px + pw, py + ph], fill=(*COL_PANEL, 178))  # 70% opacity
    draw.rectangle([px, py, px + pw, py + ph], outline=COL_BOX, width=1)

    tx = px + 14
    draw.text((tx, py + 6),  "FACE COUNTER", fill=COL_YELLOW)
    draw.line([(px + 8, py + 30), (px + pw - 8, py + 30)], fill=COL_BOX, width=1)

    # Big count
    count_str = str(face_count)
    col = COL_GREEN if face_count > 0 else COL_WHITE
    try:
        big_font = ImageFont.truetype("arial.ttf", 72)
        sm_font  = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        big_font = ImageFont.load_default()
        sm_font  = big_font

    bb = draw.textbbox((0, 0), count_str, font=big_font)
    cw = bb[2] - bb[0]
    draw.text((px + (pw - cw) // 2, py + 35), count_str, fill=col, font=big_font)
    draw.text((tx + 22, py + 118), "faces in frame", fill=COL_WHITE, font=sm_font)

    mins, secs = divmod(int(session_s), 60)
    draw.text((tx,       py + 140), f"FPS:  {fps:5.1f}", fill=COL_WHITE, font=sm_font)
    draw.text((tx + 110, py + 140), f"Peak: {peak}",     fill=COL_WHITE, font=sm_font)
    draw.text((tx,       py + 158), f"Time: {mins:02d}:{secs:02d}", fill=COL_WHITE, font=sm_font)

    return img


class Dashboard:
    def __init__(self, root: Tk, model_path: str, conf: float, cam_index: int):
        self.root = root
        self.conf = conf
        self.running = True

        root.title("Face Counter Dashboard")
        root.configure(bg="#111")
        root.protocol("WM_DELETE_WINDOW", self.stop)

        self.canvas = Label(root, bg="#111")
        self.canvas.pack()

        self.status = StringVar(value="Loading model…")
        Label(root, textvariable=self.status, bg="#111", fg="#aaa",
              font=("Consolas", 10)).pack(pady=4)

        root.bind("<Key>", self._on_key)

        self.model = YOLO(model_path)
        self.cap   = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index}")

        self.fps_buf = deque(maxlen=30)
        self.peak    = 0
        self.t_start = time.time()
        self.t_prev  = time.time()

        self._update()

    def _on_key(self, event):
        if event.keysym.lower() in ("q", "escape"):
            self.stop()

    def _update(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._update)
            return

        results = self.model(frame, conf=self.conf, verbose=False)[0]

        boxes_scores = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            boxes_scores.append(((x1, y1, x2, y2), score))

        face_count = len(boxes_scores)
        self.peak  = max(self.peak, face_count)

        now = time.time()
        self.fps_buf.append(1.0 / max(now - self.t_prev, 1e-6))
        self.t_prev = now
        fps = sum(self.fps_buf) / len(self.fps_buf)

        img = draw_overlay(frame, boxes_scores, face_count, fps,
                           self.peak, now - self.t_start)

        # Scale to occupy ~1/4 of screen (half width × half height)
        target_w = self.root.winfo_screenwidth() // 2
        target_h = self.root.winfo_screenheight() // 2
        scale = min(target_w / img.width, target_h / img.height)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)),
                             Image.BILINEAR)

        photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=photo)
        self.canvas.image = photo  # keep reference

        mins, secs = divmod(int(now - self.t_start), 60)
        self.status.set(f"Faces: {face_count}  |  Peak: {self.peak}  |  "
                        f"FPS: {fps:.1f}  |  {mins:02d}:{secs:02d}  |  Q to quit")

        self.root.after(1, self._update)

    def stop(self):
        self.running = False
        self.cap.release()
        print(f"Session ended. Peak faces: {self.peak}")
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to YOLOv8 .pt weights")
    parser.add_argument("--conf",  type=float, default=0.80,   help="Confidence threshold")
    parser.add_argument("--cam",   type=int,   default=0,       help="Camera index")
    args = parser.parse_args()

    root = Tk()
    app  = Dashboard(root, args.model, args.conf, args.cam)
    root.mainloop()
