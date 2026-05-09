"""
Project:Charm — Main Entry Point
"""

import argparse
import logging
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from pynput import keyboard
from screeninfo import get_monitors

import config
from calibration import CalibrationData, load_calibration, run_calibration
from capture import WebcamCapture
from gaze_mapper import OneEuroFilter, DeadZoneFilter, map_gaze_to_screen
from gesture_detector import GestureDetector, GestureEvent, compute_ear
from landmark_processor import LandmarkProcessor
from mouse_controller import MouseController
from overlay import DebugOverlay

logger = logging.getLogger("charm")

PAUSED = threading.Event()


def _on_key_press(key: keyboard.Key) -> None:
    try:
        if key == keyboard.Key.esc:
            if PAUSED.is_set():
                PAUSED.clear()
                logger.info("Resumed (ESC)")
            else:
                PAUSED.set()
                logger.info("Paused (ESC)")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Project:Charm",
        description="Eye-tracking mouse control via standard webcam",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration on startup")
    parser.add_argument("--no-mouse", action="store_true", help="Disable mouse control (overlay-only mode)")
    return parser.parse_args()


def detect_screen() -> tuple[int, int]:
    """Detect target monitor resolution."""
    monitors = get_monitors()
    for i, m in enumerate(monitors):
        logger.info("Monitor %d: %s (%dx%d at %d,%d)%s",
                     i, m.name or "Unknown", m.width, m.height, m.x, m.y,
                     " [PRIMARY]" if m.is_primary else "")

    idx = config.TARGET_MONITOR_INDEX
    if idx >= len(monitors):
        logger.warning("TARGET_MONITOR_INDEX=%d but only %d monitors found. Using 0.", idx, len(monitors))
        idx = 0

    target = monitors[idx]
    logger.info("Target monitor: %dx%d", target.width, target.height)
    return target.width, target.height


class SensitivityController:
    """Runtime-adjustable sensitivity for smoothing and blink threshold."""

    def __init__(self, one_euro: OneEuroFilter, gesture: GestureDetector) -> None:
        self._one_euro = one_euro
        self._gesture = gesture
        self._smooth_idx: int = config.SENSITIVITY_SMOOTH_DEFAULT_IDX
        self._ear_idx: int = config.SENSITIVITY_EAR_DEFAULT_IDX
        self._smooth_levels = config.SENSITIVITY_SMOOTH_LEVELS
        self._ear_levels = config.SENSITIVITY_EAR_LEVELS
        self._apply_smooth()
        self._apply_ear()

    def _apply_smooth(self) -> None:
        val = self._smooth_levels[self._smooth_idx]
        self._one_euro.min_cutoff = val
        logger.info("Smoothing: level %d/%d (min_cutoff=%.2f)", self._smooth_idx + 1, len(self._smooth_levels), val)

    def _apply_ear(self) -> None:
        val = self._ear_levels[self._ear_idx]
        self._gesture.set_ear_threshold(val)
        logger.info("Blink sensitivity: level %d/%d (EAR threshold=%.3f)", self._ear_idx + 1, len(self._ear_levels), val)

    def decrease_smooth(self) -> None:
        if self._smooth_idx > 0:
            self._smooth_idx -= 1
            self._apply_smooth()

    def increase_smooth(self) -> None:
        if self._smooth_idx < len(self._smooth_levels) - 1:
            self._smooth_idx += 1
            self._apply_smooth()

    def decrease_ear(self) -> None:
        if self._ear_idx > 0:
            self._ear_idx -= 1
            self._apply_ear()

    def increase_ear(self) -> None:
        if self._ear_idx < len(self._ear_levels) - 1:
            self._ear_idx += 1
            self._apply_ear()

    @property
    def smooth_idx(self) -> int:
        return self._smooth_idx

    @property
    def smooth_max(self) -> int:
        return len(self._smooth_levels) - 1

    @property
    def ear_idx(self) -> int:
        return self._ear_idx

    @property
    def ear_max(self) -> int:
        return len(self._ear_levels) - 1


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger.info("=== Project:Charm starting ===")

    ks_listener = keyboard.Listener(on_press=_on_key_press)
    ks_listener.daemon = True
    ks_listener.start()
    logger.info("Kill switch active (press ESC to pause/resume)")

    screen_w, screen_h = detect_screen()

    cap: Optional[WebcamCapture] = None
    processor: Optional[LandmarkProcessor] = None

    try:
        cap = WebcamCapture()
        processor = LandmarkProcessor()
        mouse = MouseController()
        overlay = DebugOverlay()
        gesture = GestureDetector()

        one_euro = OneEuroFilter()
        dead_zone = DeadZoneFilter()
        sensitivity = SensitivityController(one_euro, gesture)

        calibration: Optional[CalibrationData] = None
        if not args.calibrate:
            calibration = load_calibration()
        if calibration is None:
            logger.info("Starting interactive calibration...")
            calibration = run_calibration(cap, processor, screen_w, screen_h)

        consecutive_lost_frames: int = 0
        window_name = "Project:Charm"
        last_screen_x: int = screen_w // 2
        last_screen_y: int = screen_h // 2

        logger.info("Entering main loop")
        logger.info("Controls: ESC=pause  q=quit  c=recalibrate  [/]=smoothing  -/==blink")

        while True:
            frame_start = time.perf_counter()

            if PAUSED.is_set():
                frame = cap.read()
                if frame is not None:
                    overlay.show_paused_banner(frame)
                    cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested via 'q' key")
                    break
                continue

            frame = cap.read()
            if frame is None:
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.01)
                continue

            frame_h_px, frame_w_px = frame.shape[:2]
            landmarks = processor.process(frame)

            if landmarks is None:
                consecutive_lost_frames += 1
                if consecutive_lost_frames >= config.FACE_LOST_THRESHOLD:
                    overlay.show_warning(frame, "TRACKING LOST -- Move into frame")
                    one_euro.reset()
                    dead_zone.reset()
                    gesture.reset()
                overlay.draw_fps(frame)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    logger.info("Recalibration requested...")
                    calibration = run_calibration(cap, processor, screen_w, screen_h)
                continue
            else:
                consecutive_lost_frames = 0

            if not processor.check_stability(landmarks, frame_w_px, frame_h_px):
                overlay.draw_fps(frame)
                overlay.show_warning(frame, "Landmark instability -- frame skipped")
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)
                continue

            left_iris = LandmarkProcessor.get_iris_center(landmarks, config.LEFT_IRIS, frame_w_px, frame_h_px)
            right_iris = LandmarkProcessor.get_iris_center(landmarks, config.RIGHT_IRIS, frame_w_px, frame_h_px)

            # Blink-freeze: prevent cursor jumping down during blinks
            # Uses both raw EAR check AND tracker state to cover the full blink lifecycle
            # (including transition frames where EAR is above threshold but iris hasn't settled)
            left_ear = compute_ear(landmarks, config.LEFT_EYE_EAR, frame_w_px, frame_h_px)
            right_ear = compute_ear(landmarks, config.RIGHT_EYE_EAR, frame_w_px, frame_h_px)
            blink_threshold = gesture.current_ear_threshold
            ear_open = (left_ear > blink_threshold) and (right_ear > blink_threshold)
            eyes_open = ear_open and not gesture.is_blinking

            if eyes_open:
                raw_iris = (left_iris + right_iris) / 2.0
                smoothed_iris = one_euro.update(raw_iris)

                screen_x, screen_y = map_gaze_to_screen(
                    iris_x=float(smoothed_iris[0]), iris_y=float(smoothed_iris[1]),
                    gaze_min_x=calibration.gaze_min_x, gaze_max_x=calibration.gaze_max_x,
                    gaze_min_y=calibration.gaze_min_y, gaze_max_y=calibration.gaze_max_y,
                    screen_w=screen_w, screen_h=screen_h,
                )

                filtered = dead_zone.apply(np.array([screen_x, screen_y], dtype=np.float32))
                screen_x, screen_y = int(filtered[0]), int(filtered[1])
                last_screen_x, last_screen_y = screen_x, screen_y

                if not args.no_mouse:
                    mouse.move(screen_x, screen_y)
            else:
                screen_x, screen_y = last_screen_x, last_screen_y

            events = gesture.update(
                landmarks, frame_w_px, frame_h_px,
                left_iris_center_y=float(left_iris[1]),
                right_iris_center_y=float(right_iris[1]),
            )

            if not args.no_mouse:
                for event in events:
                    if event == GestureEvent.LEFT_CLICK:
                        mouse.left_click()
                        overlay.flash_gesture("LEFT CLICK")
                    elif event == GestureEvent.RIGHT_CLICK:
                        mouse.right_click()
                        overlay.flash_gesture("RIGHT CLICK")
                    elif event == GestureEvent.DOUBLE_LEFT_CLICK:
                        mouse.double_click()
                        overlay.flash_gesture("DOUBLE CLICK")
                    elif event == GestureEvent.DOUBLE_RIGHT_CLICK:
                        mouse.copy()
                        overlay.flash_gesture("COPY (Ctrl+C)")
                    elif event == GestureEvent.SCROLL_UP:
                        mouse.scroll(config.SCROLL_SPEED_PIXELS)
                    elif event == GestureEvent.SCROLL_DOWN:
                        mouse.scroll(-config.SCROLL_SPEED_PIXELS)

            overlay.draw_landmarks(frame, landmarks, frame_w_px, frame_h_px)
            overlay.draw_iris_centers(frame, left_iris, right_iris)
            overlay.draw_fps(frame)
            overlay.draw_ear_values(frame, left_ear, right_ear, gesture.current_ear_threshold)
            overlay.draw_gaze_info(frame, screen_x, screen_y)

            overlay.draw_sensitivity(frame, sensitivity.smooth_idx, sensitivity.smooth_max,
                                     sensitivity.ear_idx, sensitivity.ear_max)
            overlay.draw_gesture_flash(frame)

            if gesture.scroll_mode:
                direction_label = "UP" if config.SCROLL_SPEED_PIXELS > 0 else "DOWN"
                overlay.draw_scroll_mode(frame, True, direction_label)

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            overlay.draw_frame_time(frame, frame_ms)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit requested via 'q' key")
                break
            elif key == ord("c"):
                logger.info("Recalibration requested...")
                calibration = run_calibration(cap, processor, screen_w, screen_h)
                one_euro.reset()
                dead_zone.reset()
            elif key == ord("["):
                sensitivity.decrease_smooth()
            elif key == ord("]"):
                sensitivity.increase_smooth()
            elif key == ord("-"):
                sensitivity.decrease_ear()
            elif key == ord("="):
                sensitivity.increase_ear()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    except RuntimeError as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)
    finally:
        logger.info("Cleaning up...")
        if cap is not None:
            cap.release()
        if processor is not None:
            processor.close()
        cv2.destroyAllWindows()
        logger.info("=== Project:Charm stopped ===")


if __name__ == "__main__":
    main()
