"""
Project:Charm — Interactive Calibration Routine
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

import config
from landmark_processor import LandmarkProcessor

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    gaze_min_x: float
    gaze_max_x: float
    gaze_min_y: float
    gaze_max_y: float

    def to_dict(self) -> dict:
        return {
            "gaze_min_x": self.gaze_min_x, "gaze_max_x": self.gaze_max_x,
            "gaze_min_y": self.gaze_min_y, "gaze_max_y": self.gaze_max_y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationData":
        return cls(
            gaze_min_x=float(d["gaze_min_x"]), gaze_max_x=float(d["gaze_max_x"]),
            gaze_min_y=float(d["gaze_min_y"]), gaze_max_y=float(d["gaze_max_y"]),
        )


def _get_calibration_path() -> Path:
    return Path(os.path.expanduser(config.CALIBRATION_FILE))


def load_calibration() -> Optional[CalibrationData]:
    """Load calibration data from disk, or None if unavailable."""
    path = _get_calibration_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cal = CalibrationData.from_dict(data)
        logger.info("Calibration loaded from %s", path)
        return cal
    except FileNotFoundError:
        logger.info("No calibration file found at %s", path)
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Invalid calibration file: %s", exc)
        return None
    except PermissionError as exc:
        logger.warning("Cannot read calibration file: %s", exc)
        return None


def save_calibration(cal: CalibrationData) -> None:
    path = _get_calibration_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cal.to_dict(), f, indent=2)
        logger.info("Calibration saved to %s", path)
    except PermissionError as exc:
        logger.error("Cannot write calibration file: %s", exc)


def run_calibration(
    cap: Any, processor: LandmarkProcessor, screen_w: int, screen_h: int,
) -> CalibrationData:
    """Run 5-point interactive calibration and return calibrated gaze boundaries."""
    margin = 60
    targets: List[Tuple[int, int]] = [
        (margin, margin),
        (screen_w - margin, margin),
        (screen_w - margin, screen_h - margin),
        (margin, screen_h - margin),
        (screen_w // 2, screen_h // 2),
    ]
    target_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left", "Center"]

    window_name = "Charm Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    all_x: List[float] = []
    all_y: List[float] = []
    frames_per_point = 30
    discard_frames = 10

    for idx, (tx, ty) in enumerate(targets):
        logger.info("Calibration point %d/%d: %s (%d, %d)", idx + 1, len(targets), target_names[idx], tx, ty)

        waiting = True
        while waiting:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 30, (0, 255, 0), 2)
            cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)
            cv2.line(canvas, (tx - 40, ty), (tx + 40, ty), (0, 255, 0), 1)
            cv2.line(canvas, (tx, ty - 40), (tx, ty + 40), (0, 255, 0), 1)

            cv2.putText(canvas, f"Look at the green target ({target_names[idx]})",
                        (screen_w // 2 - 250, screen_h // 2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "Press SPACE when ready",
                        (screen_w // 2 - 160, screen_h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.putText(canvas, f"Point {idx + 1} of {len(targets)}",
                        (screen_w // 2 - 80, screen_h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                waiting = False
            elif key == ord("q") or key == 27:
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("Calibration cancelled by user")

        point_x: List[float] = []
        point_y: List[float] = []
        collected = 0

        while collected < frames_per_point:
            frame = cap.read()
            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]
            landmarks = processor.process(frame)
            if landmarks is None:
                continue

            left_iris = LandmarkProcessor.get_iris_center(landmarks, config.LEFT_IRIS, frame_w, frame_h)
            right_iris = LandmarkProcessor.get_iris_center(landmarks, config.RIGHT_IRIS, frame_w, frame_h)
            iris_center = (left_iris + right_iris) / 2.0

            collected += 1

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 30, (0, 255, 0), 2)
            cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)
            cv2.putText(canvas, f"Collecting... {collected}/{frames_per_point}",
                        (screen_w // 2 - 120, screen_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            cv2.imshow(window_name, canvas)
            cv2.waitKey(1)

            if collected > discard_frames:
                point_x.append(float(iris_center[0]))
                point_y.append(float(iris_center[1]))

        avg_x = float(np.mean(point_x))
        avg_y = float(np.mean(point_y))
        all_x.append(avg_x)
        all_y.append(avg_y)
        logger.info("  Iris center avg: (%.1f, %.1f)", avg_x, avg_y)

    cv2.destroyWindow(window_name)

    range_x = max(all_x) - min(all_x)
    range_y = max(all_y) - min(all_y)
    pad_x = range_x * config.CALIBRATION_PADDING
    pad_y = range_y * config.CALIBRATION_PADDING

    cal = CalibrationData(
        gaze_min_x=min(all_x) - pad_x, gaze_max_x=max(all_x) + pad_x,
        gaze_min_y=min(all_y) - pad_y, gaze_max_y=max(all_y) + pad_y,
    )

    logger.info("Calibration complete: X=[%.1f, %.1f], Y=[%.1f, %.1f]",
                cal.gaze_min_x, cal.gaze_max_x, cal.gaze_min_y, cal.gaze_max_y)

    save_calibration(cal)
    return cal
