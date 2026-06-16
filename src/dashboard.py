"""
DearPyGui dashboard — live webcam feed → YOLOv8 face detection → per-face classification.

Run: python src/dashboard.py
"""
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.inference_pipeline import InferencePipeline, build_classifier_registry, PipelineResult

# ── Constants ─────────────────────────────────────────────────────────────────
INFER_W, INFER_H = 640, 480
WIN_W, WIN_H = 1280, 800
RESULT_PANEL_W = 380
DISPLAY_W = WIN_W - RESULT_PANEL_W - 24
DISPLAY_H = int(DISPLAY_W * 9 / 16)
PLOT_H = 180
PLOT_BUFFER = 120

LABEL_COLORS = {
    "ATTENTIVE":  (0.0, 0.78, 0.0, 1.0),
    "DROWSY":     (1.0, 0.65, 0.0, 1.0),
    "DISTRACTED": (0.9, 0.1, 0.1, 1.0),
    "NO FACE":    (0.5, 0.5, 0.5, 1.0),
}
LABEL_COLORS_BGR = {
    name: tuple(int(c * 255) for c in (rgba[2], rgba[1], rgba[0]))
    for name, rgba in LABEL_COLORS.items()
}
CLASS_NAMES = ["ATTENTIVE", "DROWSY", "DISTRACTED"]
SERIES_COLORS = {
    "ATTENTIVE":  (0, 200, 70, 255),
    "DROWSY":     (255, 165, 0, 255),
    "DISTRACTED": (230, 25, 25, 255),
}


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self):
        self._registry = build_classifier_registry()
        self._classifier_names = list(self._registry.keys())

        self._pipeline = InferencePipeline(INFER_W, INFER_H)
        self._pipeline.set_classifier(self._registry[self._classifier_names[0]]())

        # Live stream state
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cam_index = 0

        # UI state
        self._show_boxes: bool = True
        self._det_conf: float = 0.3
        self._total_faces: int = 0

        # Time series data
        self._plot_x: list = list(range(PLOT_BUFFER))
        self._plot_counts: dict = {name: [0.0] * PLOT_BUFFER for name in CLASS_NAMES}
        self._plot_frame_idx: int = 0

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(title="Concentration Monitor", width=WIN_W, height=WIN_H)
        dpg.setup_dearpygui()

        blank = np.zeros(DISPLAY_W * DISPLAY_H * 4, dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(DISPLAY_W, DISPLAY_H, blank, tag="feed_texture")

        with dpg.window(label="Dashboard", tag="main_win", no_title_bar=True,
                        no_resize=True, no_move=True,
                        width=WIN_W, height=WIN_H, pos=(0, 0)):
            self._build_live_tab()

        dpg.show_viewport()
        self._build_series_themes()

    def _build_live_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="cam", default_value=0, width=60, min_value=0, max_value=9,
                tag="cam_idx", callback=self._on_cam_idx_change,
            )
            dpg.add_button(label="Start Stream", callback=self._on_start, tag="start_btn")
            dpg.add_button(label="Stop Stream", callback=self._on_stop, tag="stop_btn",
                           enabled=False)
            dpg.add_spacer(width=12)
            dpg.add_checkbox(label="Show Boxes", default_value=True,
                             tag="show_boxes_cb", callback=self._on_show_boxes_change)

        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_image("feed_texture", width=DISPLAY_W, height=DISPLAY_H, tag="feed_img")
                dpg.add_text("", tag="status_text")

            dpg.add_spacer(width=8)

            with dpg.child_window(width=RESULT_PANEL_W, height=DISPLAY_H + 30,
                                  border=True, tag="result_panel"):
                dpg.add_text("Classifier")
                dpg.add_combo(
                    items=self._classifier_names,
                    default_value=self._classifier_names[0],
                    width=-1,
                    callback=self._on_classifier_change,
                    tag="clf_combo",
                )
                dpg.add_text("Detection Threshold", color=(180, 180, 180, 255))
                dpg.add_drag_float(
                    tag="conf_slider", min_value=0.1, max_value=0.99,
                    default_value=0.3, speed=0.01, format="%.2f", width=-1,
                    callback=self._on_conf_change,
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
                dpg.add_text("Session Counts", color=(180, 180, 180, 255))
                with dpg.group(horizontal=True):
                    dpg.add_text("Total Faces:", color=(180, 180, 180, 255), indent=8)
                    dpg.add_text("0", tag="live_count_total", color=(200, 200, 200, 255))
                for name in CLASS_NAMES:
                    col = [int(c * 255) for c in LABEL_COLORS[name][:3]] + [255]
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{name}:", color=col, indent=8)
                        dpg.add_text("0", tag=f"live_count_{name}",
                                     color=(200, 200, 200, 255))

        dpg.add_separator()
        self._build_plot_section()

    def _build_plot_section(self):
        with dpg.plot(tag="time_plot", label="Attentiveness Over Time",
                      width=-1, height=PLOT_H, no_menus=True, no_box_select=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, tag="plot_x_axis",
                              no_gridlines=True, no_tick_labels=True)
            dpg.set_axis_limits("plot_x_axis", 0, PLOT_BUFFER - 1)
            with dpg.plot_axis(dpg.mvYAxis, tag="plot_y_axis", label="Faces"):
                dpg.add_line_series(
                    self._plot_x, [0.0] * PLOT_BUFFER,
                    label="ATTENTIVE", tag="series_ATTENTIVE",
                )
                dpg.add_line_series(
                    self._plot_x, [0.0] * PLOT_BUFFER,
                    label="DROWSY", tag="series_DROWSY",
                )
                dpg.add_line_series(
                    self._plot_x, [0.0] * PLOT_BUFFER,
                    label="DISTRACTED", tag="series_DISTRACTED",
                )

    def _build_series_themes(self):
        for name, color in SERIES_COLORS.items():
            with dpg.theme() as theme_id:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                        category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(f"series_{name}", theme_id)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_cam_idx_change(self, sender, value):
        self._cam_index = value

    def _on_conf_change(self, sender, value):
        self._det_conf = value
        self._pipeline.set_detection_conf(value)

    def _on_classifier_change(self, sender, value):
        self._pipeline.set_classifier(self._registry[value]())
        self._pipeline.reset()
        self._clear_results()
        self._reset_live_counts()

    def _on_show_boxes_change(self, sender, value):
        self._show_boxes = value

    def _on_start(self, *_):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._reset_live_counts()
        dpg.configure_item("start_btn", enabled=False)
        dpg.configure_item("stop_btn", enabled=True)
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
        self._reset_live_counts()
        dpg.set_value("status_text", "")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _video_worker(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            dpg.set_value("status_text", f"Cannot open camera {source}")
            self._reset_buttons()
            return

        scale_x = DISPLAY_W / INFER_W
        scale_y = DISPLAY_H / INFER_H

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            if self._frame_queue.full():
                continue
            infer_frame = cv2.resize(frame, (INFER_W, INFER_H))
            display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            result = self._pipeline.process_frame(infer_frame)
            annotated = self._annotate(display_frame, result, scale_x, scale_y)
            rgba = self._to_rgba(annotated)
            try:
                self._frame_queue.put_nowait((rgba, result))
            except queue.Full:
                pass

        cap.release()
        self._reset_buttons()

    def _reset_buttons(self):
        dpg.configure_item("start_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)

    # ── UI update (main thread, each render frame) ────────────────────────────

    def _update_ui(self):
        try:
            rgba, result = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        if self._stop_event.is_set():
            return

        dpg.set_value("feed_texture", rgba)

        n_faces = len(result.faces)
        dpg.set_value("status_text", f"Faces detected: {n_faces}")

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

        frame_counts = {k: 0 for k in CLASS_NAMES}
        for clf_r in result.classifier_results:
            if clf_r is not None and clf_r.label in frame_counts:
                frame_counts[clf_r.label] += 1
        for name in CLASS_NAMES:
            dpg.set_value(f"live_count_{name}", str(frame_counts[name]))

        self._total_faces += n_faces
        dpg.set_value("live_count_total", str(self._total_faces))

        self._update_plot(frame_counts)

    def _update_plot(self, frame_counts: dict):
        slot = self._plot_frame_idx % PLOT_BUFFER
        for name in CLASS_NAMES:
            self._plot_counts[name][slot] = float(frame_counts[name])
        self._plot_frame_idx += 1
        slot_now = self._plot_frame_idx % PLOT_BUFFER
        for name in CLASS_NAMES:
            rotated = self._plot_counts[name][slot_now:] + self._plot_counts[name][:slot_now]
            dpg.set_value(f"series_{name}", [self._plot_x, rotated])
        dpg.fit_axis_data("plot_y_axis")

    # ── Instance helpers ──────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, result: PipelineResult,
                  scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
        out = frame.copy()
        for i, face in enumerate(result.faces):
            x1, y1, x2, y2 = face.bbox
            x1d = int(x1 * scale_x)
            y1d = int(y1 * scale_y)
            x2d = int(x2 * scale_x)
            y2d = int(y2 * scale_y)

            if self._show_boxes:
                cv2.rectangle(out, (x1d, y1d), (x2d, y2d), (0, 220, 80), 2)

            clf_r = result.classifier_results[i] if i < len(result.classifier_results) else None
            if clf_r is not None:
                # Detection confidence on topmost line
                det_y = max(y1d - 6 - 3 * 16, 10)
                cv2.putText(out, f"det:{face.confidence:.2f}", (x1d, det_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                # Stack all 3 class probabilities above box (ATTENTIVE top, DISTRACTED closest)
                for li, name in enumerate(CLASS_NAMES):
                    p = float(clf_r.probabilities[li])
                    col_bgr = LABEL_COLORS_BGR[name]
                    y_pos = y1d - 6 - (2 - li) * 16
                    y_pos = max(y_pos, det_y + 12 + li * 14)
                    cv2.putText(out, f"{name} {p:.0%}", (x1d, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_bgr, 1)
            else:
                cv2.putText(out, f"{face.confidence:.2f}", (x1d, max(y1d - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
        return out

    def _to_rgba(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgba_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        out = np.empty(rgba_u8.size, dtype=np.float32)
        np.multiply(rgba_u8.ravel(), np.float32(1.0 / 255.0), out=out)
        return out

    def _reset_live_counts(self):
        self._total_faces = 0
        dpg.set_value("live_count_total", "0")
        for name in CLASS_NAMES:
            dpg.set_value(f"live_count_{name}", "0")

    def _clear_feed(self):
        dpg.set_value("feed_texture",
                      np.zeros(DISPLAY_W * DISPLAY_H * 4, dtype=np.float32))

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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dash = Dashboard()
    dash.run()
