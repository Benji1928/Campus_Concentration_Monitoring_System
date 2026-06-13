"""
Bare-bones DearPyGui dashboard.
Accepts webcam / image / video → YOLOv8 face detection → user-selected classifier.

Run: python src/dashboard.py
"""
import queue
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, Tk

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.inference_pipeline import InferencePipeline, build_classifier_registry, PipelineResult

# ── Constants ─────────────────────────────────────────────────────────────────
FEED_W, FEED_H = 640, 480
WIN_W, WIN_H = 1280, 820
RESULT_PANEL_W = 380

LABEL_COLORS = {
    "ATTENTIVE":  (0.0, 0.78, 0.0, 1.0),
    "SLEEPY":     (1.0, 0.65, 0.0, 1.0),
    "DISTRACTED": (0.9, 0.1, 0.1, 1.0),
    "NO FACE":    (0.5, 0.5, 0.5, 1.0),
}
BOX_COLOR_BGR = (0, 220, 80)


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self):
        self._registry = build_classifier_registry()
        self._classifier_names = list(self._registry.keys())

        self._pipeline = InferencePipeline(FEED_W, FEED_H)
        self._pipeline.set_classifier(self._registry[self._classifier_names[0]]())

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._source_mode = "webcam"   # "webcam" | "file"
        self._file_path: str | None = None
        self._cam_index = 0

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(title="Concentration Monitor", width=WIN_W, height=WIN_H)
        dpg.setup_dearpygui()

        # Blank texture — RGBA float32 [0,1] flat list
        blank = [0.0] * (FEED_W * FEED_H * 4)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(FEED_W, FEED_H, blank, tag="feed_texture")

        with dpg.window(label="Dashboard", tag="main_win", no_title_bar=True,
                        no_resize=True, no_move=True,
                        width=WIN_W, height=WIN_H, pos=(0, 0)):

            # ── Source bar ──────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("Source:")
                dpg.add_radio_button(
                    items=["Webcam", "File"],
                    default_value="Webcam",
                    horizontal=True,
                    callback=self._on_source_change,
                    tag="source_radio",
                )
                dpg.add_input_int(
                    label="cam", default_value=0, width=60, min_value=0, max_value=9,
                    tag="cam_idx", callback=self._on_cam_idx_change,
                )
                dpg.add_button(label="Browse", tag="browse_btn",
                               callback=self._on_browse, enabled=False)
                dpg.add_button(label="Start", callback=self._on_start, tag="start_btn")
                dpg.add_button(label="Stop",  callback=self._on_stop,  tag="stop_btn",
                               enabled=False)

            dpg.add_separator()

            with dpg.group(horizontal=True):
                # ── Left: video feed ─────────────────────────────────────────
                with dpg.group():
                    dpg.add_image("feed_texture", width=FEED_W, height=FEED_H, tag="feed_img")
                    dpg.add_text("", tag="status_text")

                dpg.add_spacer(width=8)

                # ── Right: results panel ──────────────────────────────────────
                with dpg.child_window(width=RESULT_PANEL_W, height=FEED_H + 30,
                                      border=True, tag="result_panel"):
                    dpg.add_text("Classifier")
                    dpg.add_combo(
                        items=self._classifier_names,
                        default_value=self._classifier_names[0],
                        width=-1,
                        callback=self._on_classifier_change,
                        tag="clf_combo",
                    )
                    dpg.add_separator()

                    dpg.add_text("Result", tag="result_header")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Label:", color=(180, 180, 180, 255))
                        dpg.add_text("—", tag="result_label", color=(200, 200, 200, 255))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Conf: ", color=(180, 180, 180, 255))
                        dpg.add_text("—", tag="result_conf", color=(200, 200, 200, 255))

                    dpg.add_separator()
                    dpg.add_text("Confidence", color=(180, 180, 180, 255))

                    for i, name in enumerate(["ATTENTIVE", "SLEEPY", "DISTRACTED"]):
                        col = [int(c * 255) for c in LABEL_COLORS[name][:3]] + [255]
                        dpg.add_text(name, color=col, tag=f"prob_label_{i}")
                        dpg.add_progress_bar(default_value=0.0, width=-1, tag=f"prob_bar_{i}")
                        dpg.add_text("0%", tag=f"prob_pct_{i}", color=(200, 200, 200, 255))

                    dpg.add_separator()
                    dpg.add_text("Features", color=(180, 180, 180, 255), tag="feat_header")
                    for feat in ["EAR", "MAR", "Yaw", "Pitch", "PERCLOS", "Blink/min"]:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{feat}:", color=(150, 150, 150, 255), indent=8)
                            dpg.add_text("—", tag=f"feat_{feat}", color=(200, 200, 200, 255))

        dpg.show_viewport()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_source_change(self, sender, value):
        self._source_mode = "webcam" if value == "Webcam" else "file"
        dpg.configure_item("browse_btn", enabled=(self._source_mode == "file"))
        dpg.configure_item("cam_idx", enabled=(self._source_mode == "webcam"))

    def _on_cam_idx_change(self, sender, value):
        self._cam_index = value

    def _on_browse(self, *_):
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path:
            self._file_path = path
            dpg.set_value("status_text", f"File: {Path(path).name}")

    def _on_classifier_change(self, sender, value):
        self._pipeline.set_classifier(self._registry[value]())
        self._pipeline.reset()
        self._clear_results()

    def _on_start(self, *_):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        dpg.configure_item("start_btn", enabled=False)
        dpg.configure_item("stop_btn", enabled=True)

        if self._source_mode == "file" and self._file_path:
            path = self._file_path
            ext = Path(path).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self._worker_thread = threading.Thread(
                    target=self._image_worker, args=(path,), daemon=True
                )
            else:
                self._worker_thread = threading.Thread(
                    target=self._video_worker, args=(path,), daemon=True
                )
        else:
            self._worker_thread = threading.Thread(
                target=self._video_worker, args=(self._cam_index,), daemon=True
            )
        self._worker_thread.start()

    def _on_stop(self, *_):
        self._stop_event.set()
        dpg.configure_item("start_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)

    # ── Worker threads ────────────────────────────────────────────────────────

    def _image_worker(self, path: str):
        frame = cv2.imread(path)
        if frame is None:
            dpg.set_value("status_text", f"Cannot read: {path}")
            self._reset_buttons()
            return
        frame = cv2.resize(frame, (FEED_W, FEED_H))
        result = self._pipeline.process_frame(frame)
        annotated = _annotate(frame, result)
        rgba = _to_rgba(annotated)
        try:
            self._frame_queue.put_nowait((rgba, result))
        except queue.Full:
            pass
        self._reset_buttons()

    def _video_worker(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            dpg.set_value("status_text", f"Cannot open: {source}")
            self._reset_buttons()
            return

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            # Skip inference when the display hasn't caught up — drain the camera
            # buffer but don't burn CPU on detection/classification for dropped frames.
            if self._frame_queue.full():
                continue
            frame = cv2.resize(frame, (FEED_W, FEED_H))
            result = self._pipeline.process_frame(frame)
            annotated = _annotate(frame, result)
            rgba = _to_rgba(annotated)
            try:
                self._frame_queue.put_nowait((rgba, result))
            except queue.Full:
                pass

        cap.release()
        self._reset_buttons()

    def _reset_buttons(self):
        dpg.configure_item("start_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)

    # ── UI update (called from main thread each render frame) ─────────────────

    def _update_ui(self):
        try:
            rgba, result = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        dpg.set_value("feed_texture", rgba)

        n_faces = len(result.faces)
        status = f"Faces: {n_faces}  |  Classifier: {result.active_classifier_name}"
        dpg.set_value("status_text", status)

        clf_result = result.classifier_result
        if clf_result is not None:
            col = [int(c * 255) for c in LABEL_COLORS.get(clf_result.label, LABEL_COLORS["NO FACE"])[:3]] + [255]
            dpg.configure_item("result_label", color=col)
            dpg.set_value("result_label", clf_result.label)
            dpg.set_value("result_conf", f"{clf_result.confidence:.1%}")

            for i, p in enumerate(clf_result.probabilities):
                dpg.set_value(f"prob_bar_{i}", float(p))
                dpg.set_value(f"prob_pct_{i}", f"{p:.1%}")
        else:
            self._clear_results()

        features = result.features
        if features:
            dpg.set_value("feat_EAR",       f"{features['ear_avg']:.3f}")
            dpg.set_value("feat_MAR",       f"{features['mar']:.3f}")
            dpg.set_value("feat_Yaw",       f"{features['yaw']:+.1f}°")
            dpg.set_value("feat_Pitch",     f"{features['pitch']:+.1f}°")
            dpg.set_value("feat_PERCLOS",   f"{features['perclos']:.2f}")
            dpg.set_value("feat_Blink/min", f"{features['blink_rate']:.1f}")
        else:
            for feat in ["EAR", "MAR", "Yaw", "Pitch", "PERCLOS", "Blink/min"]:
                dpg.set_value(f"feat_{feat}", "—")

    def _clear_results(self):
        dpg.set_value("result_label", "—")
        dpg.set_value("result_conf", "—")
        for i in range(3):
            dpg.set_value(f"prob_bar_{i}", 0.0)
            dpg.set_value(f"prob_pct_{i}", "0%")

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        while dpg.is_dearpygui_running():
            self._update_ui()
            dpg.render_dearpygui_frame()
            time.sleep(1 / 60)

        self._stop_event.set()
        self._pipeline.close()
        dpg.destroy_context()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, result: PipelineResult) -> np.ndarray:
    out = frame.copy()
    for face in result.faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)
        cv2.putText(out, f"{face.confidence:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR_BGR, 1)
    if result.classifier_result:
        label = result.classifier_result.label
        col_rgb = LABEL_COLORS.get(label, LABEL_COLORS["NO FACE"])
        col_bgr = tuple(int(c * 255) for c in (col_rgb[2], col_rgb[1], col_rgb[0]))
        cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_bgr, 2)
    return out


def _to_rgba(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if (w, h) != (FEED_W, FEED_H):
        frame_bgr = cv2.resize(frame_bgr, (FEED_W, FEED_H))
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) * (1.0 / 255.0)).ravel()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dash = Dashboard()
    dash.run()
