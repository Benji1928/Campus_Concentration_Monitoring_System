"""
DearPyGui dashboard — two tabs:
  Live Stream  : webcam / video file → YOLOv8 face detection → per-face classification
  Image Upload : batch image inference with per-class probability breakdown

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
WIN_W, WIN_H = 1280, 840
RESULT_PANEL_W = 380

LABEL_COLORS = {
    "ATTENTIVE":  (0.0, 0.78, 0.0, 1.0),
    "DROWSY":     (1.0, 0.65, 0.0, 1.0),
    "DISTRACTED": (0.9, 0.1, 0.1, 1.0),
    "NO FACE":    (0.5, 0.5, 0.5, 1.0),
}
BOX_COLOR_BGR = (0, 220, 80)
CLASS_NAMES = ["ATTENTIVE", "DROWSY", "DISTRACTED"]


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self):
        self._registry = build_classifier_registry()
        self._classifier_names = list(self._registry.keys())

        self._pipeline = InferencePipeline(FEED_W, FEED_H)
        self._pipeline.set_classifier(self._registry[self._classifier_names[0]]())

        # Live stream state
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._source_mode = "webcam"
        self._file_path: str | None = None
        self._cam_index = 0

        # Upload / batch state
        self._upload_paths: list[str] = []
        self._batch_queue: queue.Queue = queue.Queue()
        self._batch_worker_thread: threading.Thread | None = None
        self._batch_stop_event = threading.Event()
        self._batch_label_counts: dict[str, int] = {}

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(title="Concentration Monitor", width=WIN_W, height=WIN_H)
        dpg.setup_dearpygui()

        blank = [0.0] * (FEED_W * FEED_H * 4)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(FEED_W, FEED_H, blank, tag="feed_texture")

        with dpg.window(label="Dashboard", tag="main_win", no_title_bar=True,
                        no_resize=True, no_move=True,
                        width=WIN_W, height=WIN_H, pos=(0, 0)):
            with dpg.tab_bar():
                with dpg.tab(label="Live Stream"):
                    self._build_live_tab()
                with dpg.tab(label="Image Upload"):
                    self._build_upload_tab()

        dpg.show_viewport()

    def _build_live_tab(self):
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
            dpg.add_button(label="Browse...", tag="browse_btn",
                           callback=self._on_browse, enabled=False)
            dpg.add_button(label="Start Stream", callback=self._on_start, tag="start_btn")
            dpg.add_button(label="Stop Stream", callback=self._on_stop, tag="stop_btn",
                           enabled=False)

        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_image("feed_texture", width=FEED_W, height=FEED_H, tag="feed_img")
                dpg.add_text("", tag="status_text")

            dpg.add_spacer(width=8)

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

                dpg.add_text("Classification", tag="result_header")
                with dpg.group(horizontal=True):
                    dpg.add_text("Label:", color=(180, 180, 180, 255))
                    dpg.add_text("—", tag="result_label", color=(200, 200, 200, 255))
                with dpg.group(horizontal=True):
                    dpg.add_text("Conf: ", color=(180, 180, 180, 255))
                    dpg.add_text("—", tag="result_conf", color=(200, 200, 200, 255))

                dpg.add_separator()
                dpg.add_text("Probabilities", color=(180, 180, 180, 255))

                for i, name in enumerate(CLASS_NAMES):
                    col = [int(c * 255) for c in LABEL_COLORS[name][:3]] + [255]
                    dpg.add_text(name, color=col, tag=f"prob_label_{i}")
                    dpg.add_progress_bar(default_value=0.0, width=-1, tag=f"prob_bar_{i}")
                    dpg.add_text("0%", tag=f"prob_pct_{i}", color=(200, 200, 200, 255))

                dpg.add_separator()
                dpg.add_text("Face Features", color=(180, 180, 180, 255), tag="feat_header")
                for feat in ["EAR", "MAR", "Yaw", "Pitch", "PERCLOS", "Blink/min"]:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{feat}:", color=(150, 150, 150, 255), indent=8)
                        dpg.add_text("—", tag=f"feat_{feat}", color=(200, 200, 200, 255))

    def _build_upload_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add Images", callback=self._on_add_images)
            dpg.add_button(label="Clear All", callback=self._on_clear_images)
            dpg.add_text("0 images selected", tag="upload_count")

        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_text("Classifier:")
            dpg.add_combo(
                items=self._classifier_names,
                default_value=self._classifier_names[0],
                width=220,
                tag="upload_clf_combo",
            )

        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Analyze Images", callback=self._on_run_batch,
                           tag="upload_run_btn")
            dpg.add_text("", tag="upload_status", color=(160, 160, 160, 255))

        dpg.add_separator()

        with dpg.child_window(tag="upload_results_panel", height=620, border=False):
            pass  # populated dynamically by _add_result_row

        dpg.add_separator()
        dpg.add_text("", tag="upload_summary", color=(200, 200, 200, 255))

    # ── Live stream callbacks ─────────────────────────────────────────────────

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
        if not path:
            return
        self._file_path = path
        ext = Path(path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            self._stop_event.set()
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._clear_feed()
            self._clear_results()
            self._stop_event.clear()
            dpg.set_value("status_text", f"Loading {Path(path).name}...")
            dpg.configure_item("start_btn", enabled=False)
            dpg.configure_item("stop_btn", enabled=False)
            self._worker_thread = threading.Thread(
                target=self._image_worker, args=(path,), daemon=True
            )
            self._worker_thread.start()
        else:
            dpg.set_value("status_text", f"Ready: {Path(path).name} — press Start Stream")

    def _on_classifier_change(self, sender, value):
        self._pipeline.set_classifier(self._registry[value]())
        self._pipeline.reset()
        self._clear_results()

    def _on_start(self, *_):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self._batch_worker_thread and self._batch_worker_thread.is_alive():
            dpg.set_value("status_text", "Batch inference running — wait for it to finish first")
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
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._clear_feed()
        self._clear_results()
        dpg.set_value("status_text", "")

    # ── Upload tab callbacks ──────────────────────────────────────────────────

    def _on_add_images(self, *_):
        root = Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not paths:
            return
        self._upload_paths.extend(paths)
        dpg.set_value("upload_count", f"{len(self._upload_paths)} images selected")

    def _on_clear_images(self, *_):
        self._upload_paths.clear()
        dpg.set_value("upload_count", "0 images selected")
        dpg.delete_item("upload_results_panel", children_only=True)
        dpg.set_value("upload_summary", "")
        dpg.set_value("upload_status", "")

    def _on_run_batch(self, *_):
        if not self._upload_paths:
            dpg.set_value("upload_status", "No images selected.")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            dpg.set_value("upload_status", "Stop live stream first.")
            return
        if self._batch_worker_thread and self._batch_worker_thread.is_alive():
            dpg.set_value("upload_status", "Already running.")
            return

        dpg.delete_item("upload_results_panel", children_only=True)
        dpg.set_value("upload_summary", "")
        self._batch_label_counts = {k: 0 for k in CLASS_NAMES}

        n = len(self._upload_paths)
        dpg.set_value("upload_status", f"Running... 0/{n}")
        dpg.configure_item("upload_run_btn", enabled=False)

        self._batch_stop_event.clear()
        clf_name = dpg.get_value("upload_clf_combo")
        paths = list(self._upload_paths)
        self._batch_worker_thread = threading.Thread(
            target=self._batch_worker, args=(paths, clf_name), daemon=True
        )
        self._batch_worker_thread.start()

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
        dpg.set_value("status_text", f"Done: {Path(path).name}")
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

    def _batch_worker(self, paths: list[str], clf_name: str):
        pipeline = InferencePipeline(FEED_W, FEED_H)
        pipeline.set_classifier(self._registry[clf_name]())

        for i, path in enumerate(paths):
            if self._batch_stop_event.is_set():
                break
            frame = cv2.imread(path)
            if frame is None:
                self._batch_queue.put((i, path, None))
                continue
            frame = cv2.resize(frame, (FEED_W, FEED_H))
            pipeline.reset()
            result = pipeline.process_frame(frame)
            self._batch_queue.put((i, path, result))

        pipeline.close()
        self._batch_queue.put(None)  # sentinel

    def _reset_buttons(self):
        dpg.configure_item("start_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)

    # ── UI update (main thread, each render frame) ────────────────────────────

    def _update_ui(self):
        # Live feed
        try:
            rgba, result = self._frame_queue.get_nowait()
            dpg.set_value("feed_texture", rgba)

            n_faces = len(result.faces)
            dpg.set_value(
                "status_text",
                f"Faces detected: {n_faces}",
            )

            clf_result = result.classifier_result
            if clf_result is not None:
                col = [int(c * 255) for c in LABEL_COLORS.get(
                    clf_result.label, LABEL_COLORS["NO FACE"])[:3]] + [255]
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
        except queue.Empty:
            pass

        # Batch queue
        while True:
            try:
                item = self._batch_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:  # sentinel — batch done
                n = len(self._upload_paths)
                dpg.set_value("upload_status", f"Done — {n} images processed")
                dpg.configure_item("upload_run_btn", enabled=True)
                counts = self._batch_label_counts
                summary = "   ".join(f"{k}: {counts.get(k, 0)}" for k in CLASS_NAMES)
                dpg.set_value("upload_summary", f"Summary:  {summary}")
                break

            idx, path, result = item
            processed = idx + 1
            total = len(self._upload_paths)
            dpg.set_value("upload_status", f"Running... {processed}/{total}")
            self._add_result_row(idx, path, result)
            if result and result.classifier_result:
                label = result.classifier_result.label
                self._batch_label_counts[label] = self._batch_label_counts.get(label, 0) + 1

    def _add_result_row(self, idx: int, path: str, result):
        filename = Path(path).name
        parent = "upload_results_panel"

        with dpg.group(tag=f"upload_row_{idx}", parent=parent):
            if result is None:
                dpg.add_text(f"[ERROR]  {filename}  — could not read file",
                             color=(180, 80, 80, 255))
                dpg.add_separator()
                return

            n_faces = len(result.faces)

            # Header: filename + YOLO face count
            with dpg.group(horizontal=True):
                dpg.add_text(filename, color=(220, 220, 220, 255))
                face_txt = f"  {n_faces} face{'s' if n_faces != 1 else ''} detected"
                dpg.add_text(face_txt, color=(120, 120, 120, 255))

            if n_faces == 0:
                dpg.add_text("  No face — skipping classification",
                             color=(160, 80, 80, 255), indent=8)
            else:
                # Per-face classification results (all faces, not just primary)
                for fi, clf_r in enumerate(result.classifier_results):
                    face_tag = f"Face {fi + 1}" if n_faces > 1 else "Result"
                    if clf_r is None:
                        with dpg.group(horizontal=True, indent=8):
                            dpg.add_text(f"[{face_tag}]", color=(130, 130, 130, 255))
                            dpg.add_text("classifier error", color=(180, 80, 80, 255))
                        continue

                    label_col = [int(c * 255) for c in LABEL_COLORS.get(
                        clf_r.label, LABEL_COLORS["NO FACE"])[:3]] + [255]

                    with dpg.group(indent=8):
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"[{face_tag}]", color=(130, 130, 130, 255))
                            dpg.add_text(f"  {clf_r.label}", color=label_col)
                            dpg.add_text(f"  conf {clf_r.confidence:.1%}",
                                         color=(150, 150, 150, 255))

                        for ci, name in enumerate(CLASS_NAMES):
                            p = float(clf_r.probabilities[ci]) if clf_r.probabilities is not None else 0.0
                            col = [int(c * 255) for c in LABEL_COLORS[name][:3]] + [255]
                            with dpg.group(horizontal=True, indent=8):
                                dpg.add_text(name, color=col)
                                dpg.add_progress_bar(default_value=p, width=180,
                                                     tag=f"upload_bar_{idx}_{fi}_{ci}")
                                dpg.add_text(f"{p:.1%}", color=(200, 200, 200, 255))

            dpg.add_separator()

    def _clear_feed(self):
        dpg.set_value("feed_texture", [0.0] * (FEED_W * FEED_H * 4))

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
        self._batch_stop_event.set()
        self._pipeline.close()
        dpg.destroy_context()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, result: PipelineResult) -> np.ndarray:
    out = frame.copy()
    for i, face in enumerate(result.faces):
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)

        clf_r = result.classifier_results[i] if i < len(result.classifier_results) else None
        if clf_r is not None:
            col_rgb = LABEL_COLORS.get(clf_r.label, LABEL_COLORS["NO FACE"])
            col_bgr = tuple(int(c * 255) for c in (col_rgb[2], col_rgb[1], col_rgb[0]))
            text = f"{clf_r.label} {clf_r.confidence:.0%}"
            cv2.putText(out, text, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_bgr, 1)
        else:
            cv2.putText(out, f"{face.confidence:.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR_BGR, 1)

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
