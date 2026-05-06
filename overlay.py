"""
Project:Charm — Debug/Status Overlay Renderer
"""

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_GREEN = (0, 255, 0)
_BLUE = (255, 180, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_WHITE = (255, 255, 255)
_CYAN = (255, 255, 0)
_DARK_BG = (30, 30, 30)
_ORANGE = (0, 140, 255)
_MAGENTA = (200, 0, 200)


class DebugOverlay:
    """Renders diagnostic overlays onto video frames."""

    def __init__(self) -> None:
        self._prev_time: float = time.perf_counter()
        self._fps: float = 0.0
        self._fps_alpha: float = 0.1
        self._gesture_flash: str = ""
        self._gesture_flash_until: float = 0.0

    def _update_fps(self) -> None:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        instant_fps = 1.0 / (dt + 1e-9)
        self._fps = self._fps_alpha * instant_fps + (1.0 - self._fps_alpha) * self._fps

    def draw_landmarks(self, frame: np.ndarray, landmarks: Any, frame_w: int, frame_h: int) -> None:
        for i, lm in enumerate(landmarks):
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 1, _GREEN, -1)

        for idx_list in [config.LEFT_IRIS, config.RIGHT_IRIS]:
            for i in idx_list:
                x = int(landmarks[i].x * frame_w)
                y = int(landmarks[i].y * frame_h)
                cv2.circle(frame, (x, y), 3, _BLUE, -1)

    def draw_iris_centers(self, frame: np.ndarray, left_center: np.ndarray, right_center: np.ndarray) -> None:
        for center in [left_center, right_center]:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(frame, (cx, cy), _CYAN, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    def draw_fps(self, frame: np.ndarray) -> None:
        self._update_fps()
        cv2.putText(frame, f"FPS: {self._fps:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 1)

    def draw_ear_values(self, frame: np.ndarray, left_ear: float, right_ear: float, threshold: float = 0.0) -> None:
        text = f"EAR  L:{left_ear:.2f}  R:{right_ear:.2f}  thr:{threshold:.2f}"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)

    def draw_gaze_info(self, frame: np.ndarray, screen_x: int, screen_y: int) -> None:
        cv2.putText(frame, f"Gaze: ({screen_x}, {screen_y})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)

    def draw_frame_time(self, frame: np.ndarray, frame_ms: float) -> None:
        colour = _GREEN if frame_ms < 33.0 else _YELLOW
        cv2.putText(frame, f"Frame: {frame_ms:.1f}ms", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    def draw_sensitivity(self, frame: np.ndarray, smooth_level: int, smooth_max: int, ear_level: int, ear_max: int) -> None:
        h = frame.shape[0]
        text = f"Smooth: {smooth_level + 1}/{smooth_max + 1} [/]  Blink: {ear_level + 1}/{ear_max + 1} [-/=]"
        cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ORANGE, 1)

    def draw_scroll_mode(self, frame: np.ndarray, active: bool, direction: str = "") -> None:
        if active:
            h, w = frame.shape[:2]
            cv2.putText(frame, f"SCROLL MODE: {direction}", (w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CYAN, 2)

    def flash_gesture(self, gesture_name: str, duration: float = 0.5) -> None:
        self._gesture_flash = gesture_name
        self._gesture_flash_until = time.perf_counter() + duration

    def draw_gesture_flash(self, frame: np.ndarray) -> None:
        if time.perf_counter() < self._gesture_flash_until:
            h, w = frame.shape[:2]
            text_size = cv2.getTextSize(self._gesture_flash, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tx = (w - text_size[0]) // 2
            cv2.putText(frame, self._gesture_flash, (tx, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _MAGENTA, 2)

    def show_warning(self, frame: np.ndarray, message: str) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), _DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, message, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _YELLOW, 2)

    def show_paused_banner(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), _DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        text = "PAUSED  --  Press ESC to resume"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, text, (tx, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 2)
