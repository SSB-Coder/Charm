"""
Project:Charm — Gesture Detector
"""

import logging
import time
from enum import Enum, auto
from typing import Any, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


class GestureEvent(Enum):
    NONE = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    BOTH_CLICK = auto()
    DOUBLE_LEFT_CLICK = auto()
    DOUBLE_RIGHT_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()


class _EARBaseline:
    """Auto-calibrates resting EAR per eye to derive blink threshold."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: List[float] = []
        self._baseline: Optional[float] = None
        self._threshold: float = config.EAR_BLINK_THRESHOLD
        self._calibrated: bool = False

    def feed(self, ear: float) -> None:
        if self._calibrated:
            return
        if ear > 0.15:
            self._samples.append(ear)
        if len(self._samples) >= config.EAR_BASELINE_FRAMES:
            self._baseline = float(np.mean(self._samples))
            self._threshold = self._baseline * config.EAR_BASELINE_RATIO
            self._calibrated = True
            logger.info(
                "%s eye EAR baseline=%.3f, auto-threshold=%.3f",
                self.name, self._baseline, self._threshold,
            )

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def reset(self) -> None:
        self._samples.clear()
        self._baseline = None
        self._threshold = config.EAR_BLINK_THRESHOLD
        self._calibrated = False


class _EyeState(Enum):
    OPEN = auto()
    CLOSING = auto()
    CLOSED = auto()
    BLINK_FIRED = auto()
    COOLDOWN = auto()


class _EyeBlinkTracker:
    """Per-eye blink state machine with depth tracking."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.state: _EyeState = _EyeState.OPEN
        self._closed_frame_count: int = 0
        self._close_start_time: float = 0.0
        self._cooldown_frame_count: int = 0
        self._min_ear: float = 1.0

    def update(self, ear: float, threshold: float) -> Tuple[bool, float]:
        """Returns (blink_fired, blink_depth). Fires on rising edge."""
        is_closed = ear < threshold
        blink_fired = False
        blink_depth = 1.0

        if self.state == _EyeState.OPEN:
            if is_closed:
                self.state = _EyeState.CLOSING
                self._closed_frame_count = 1
                self._close_start_time = time.perf_counter()
                self._min_ear = ear

        elif self.state == _EyeState.CLOSING:
            if is_closed:
                self._closed_frame_count += 1
                self._min_ear = min(self._min_ear, ear)
                if self._closed_frame_count >= config.EAR_BLINK_CONSEC_FRAMES:
                    self.state = _EyeState.CLOSED
            else:
                self.state = _EyeState.OPEN
                self._closed_frame_count = 0
                self._min_ear = 1.0

        elif self.state == _EyeState.CLOSED:
            self._min_ear = min(self._min_ear, ear)
            elapsed_ms = (time.perf_counter() - self._close_start_time) * 1000.0
            if not is_closed:
                if elapsed_ms <= config.BLINK_MAX_DURATION_MS:
                    blink_fired = True
                    blink_depth = self._min_ear
                    self.state = _EyeState.BLINK_FIRED
                    logger.debug("%s blink (%.0fms, depth=%.3f)", self.name, elapsed_ms, blink_depth)
                else:
                    self.state = _EyeState.COOLDOWN
                    self._cooldown_frame_count = 0
            elif elapsed_ms > config.BLINK_MAX_DURATION_MS:
                self.state = _EyeState.COOLDOWN
                self._cooldown_frame_count = 0

        elif self.state == _EyeState.BLINK_FIRED:
            self.state = _EyeState.COOLDOWN
            self._cooldown_frame_count = 0
            self._min_ear = 1.0

        elif self.state == _EyeState.COOLDOWN:
            if not is_closed:
                self._cooldown_frame_count += 1
                if self._cooldown_frame_count >= config.EAR_REOPEN_FRAMES:
                    self.state = _EyeState.OPEN
                    self._min_ear = 1.0
            else:
                self._cooldown_frame_count = 0

        return blink_fired, blink_depth

    def reset(self) -> None:
        self.state = _EyeState.OPEN
        self._closed_frame_count = 0
        self._close_start_time = 0.0
        self._cooldown_frame_count = 0
        self._min_ear = 1.0


def compute_ear(
    landmarks: Any, eye_indices: List[int], frame_w: int, frame_h: int,
) -> float:
    """Compute Eye Aspect Ratio (Soukupova & Cech, 2016) from 6 landmarks."""
    pts = np.array(
        [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in eye_indices],
        dtype=np.float32,
    )
    A = float(np.linalg.norm(pts[1] - pts[5]))
    B = float(np.linalg.norm(pts[2] - pts[4]))
    C = float(np.linalg.norm(pts[0] - pts[3]))
    return (A + B) / (2.0 * C + 1e-6)


def compute_vertical_gaze_ratio(
    landmarks: Any, eye_top_idx: int, eye_bot_idx: int,
    iris_center_y: float, frame_h: int,
) -> float:
    """Vertical iris position in eye socket. 0=up, 0.5=center, 1=down."""
    top_y = landmarks[eye_top_idx].y * frame_h
    bot_y = landmarks[eye_bot_idx].y * frame_h
    ratio = (iris_center_y - top_y) / (bot_y - top_y + 1e-6)
    return float(np.clip(ratio, 0.0, 1.0))


class _DoubleClickTracker:
    """Upgrades rapid same-eye blinks to double-click events."""

    def __init__(self) -> None:
        self._last_left_click_time: float = 0.0
        self._last_right_click_time: float = 0.0
        self._left_cooldown_until: float = 0.0
        self._right_cooldown_until: float = 0.0

    def check_left(self, now: float) -> bool:
        if now < self._left_cooldown_until:
            return False
        dt = (now - self._last_left_click_time) * 1000.0
        if 0 < dt < config.DOUBLE_CLICK_WINDOW_MS:
            self._left_cooldown_until = now + config.DOUBLE_CLICK_COOLDOWN_MS / 1000.0
            self._last_left_click_time = 0.0
            return True
        self._last_left_click_time = now
        return False

    def check_right(self, now: float) -> bool:
        if now < self._right_cooldown_until:
            return False
        dt = (now - self._last_right_click_time) * 1000.0
        if 0 < dt < config.DOUBLE_CLICK_WINDOW_MS:
            self._right_cooldown_until = now + config.DOUBLE_CLICK_COOLDOWN_MS / 1000.0
            self._last_right_click_time = 0.0
            return True
        self._last_right_click_time = now
        return False

    def reset(self) -> None:
        self._last_left_click_time = 0.0
        self._last_right_click_time = 0.0
        self._left_cooldown_until = 0.0
        self._right_cooldown_until = 0.0


class GestureDetector:
    """Combines blink detection, double-click, cross-eye suppression, and scroll."""

    def __init__(self) -> None:
        self._left_eye = _EyeBlinkTracker("Left")
        self._right_eye = _EyeBlinkTracker("Right")
        self._left_baseline = _EARBaseline("Left")
        self._right_baseline = _EARBaseline("Right")
        self._double_click = _DoubleClickTracker()

        self._last_left_blink_time: float = 0.0
        self._last_right_blink_time: float = 0.0
        self._last_left_blink_depth: float = 1.0
        self._last_right_blink_depth: float = 1.0

        self._right_suppress_until: float = 0.0
        self._left_suppress_until: float = 0.0

        self._ear_threshold_override: Optional[float] = None

        self._scroll_mode: bool = False
        self._scroll_direction: int = 0
        self._scroll_active_frames: int = 0
        self._scroll_neutral_frames: int = 0

    @property
    def scroll_mode(self) -> bool:
        return self._scroll_mode

    @property
    def current_ear_threshold(self) -> float:
        if self._ear_threshold_override is not None:
            return self._ear_threshold_override
        if self._left_baseline.calibrated:
            return (self._left_baseline.threshold + self._right_baseline.threshold) / 2.0
        return config.EAR_BLINK_THRESHOLD

    def set_ear_threshold(self, value: float) -> None:
        self._ear_threshold_override = value
        logger.info("EAR blink threshold set to %.3f", value)

    def clear_ear_override(self) -> None:
        self._ear_threshold_override = None

    def _is_deep_blink(self, depth: float, threshold: float) -> bool:
        return depth < threshold * config.DUAL_BLINK_MIN_DEPTH_RATIO

    def update(
        self, landmarks: Any, frame_w: int, frame_h: int,
        left_iris_center_y: float, right_iris_center_y: float,
    ) -> List[GestureEvent]:
        """Process one frame, return list of gesture events."""
        events: List[GestureEvent] = []
        now = time.perf_counter()

        left_ear = compute_ear(landmarks, config.LEFT_EYE_EAR, frame_w, frame_h)
        right_ear = compute_ear(landmarks, config.RIGHT_EYE_EAR, frame_w, frame_h)

        self._left_baseline.feed(left_ear)
        self._right_baseline.feed(right_ear)

        if self._ear_threshold_override is not None:
            left_thresh = self._ear_threshold_override
            right_thresh = self._ear_threshold_override
        else:
            left_thresh = self._left_baseline.threshold
            right_thresh = self._right_baseline.threshold

        left_blinked, left_depth = self._left_eye.update(left_ear, left_thresh)
        right_blinked, right_depth = self._right_eye.update(right_ear, right_thresh)

        # Cross-eye sympathetic suppression
        if left_blinked and now < self._left_suppress_until:
            if not self._is_deep_blink(left_depth, left_thresh):
                left_blinked = False
                logger.debug("Left blink suppressed (sympathetic, depth=%.3f)", left_depth)

        if right_blinked and now < self._right_suppress_until:
            if not self._is_deep_blink(right_depth, right_thresh):
                right_blinked = False
                logger.debug("Right blink suppressed (sympathetic, depth=%.3f)", right_depth)

        if left_blinked:
            self._last_left_blink_time = now
            self._last_left_blink_depth = left_depth
            self._right_suppress_until = now + config.CROSS_EYE_SUPPRESSION_MS / 1000.0

        if right_blinked:
            self._last_right_blink_time = now
            self._last_right_blink_depth = right_depth
            self._left_suppress_until = now + config.CROSS_EYE_SUPPRESSION_MS / 1000.0

        # Dual-blink detection with depth validation
        both_blink = False

        if left_blinked and right_blinked:
            left_deep = self._is_deep_blink(left_depth, left_thresh)
            right_deep = self._is_deep_blink(right_depth, right_thresh)

            if left_deep and right_deep:
                both_blink = True
                events.append(GestureEvent.BOTH_CLICK)
            elif left_deep and not right_deep:
                right_blinked = False
            elif right_deep and not left_deep:
                left_blinked = False
            else:
                left_blinked = False
                right_blinked = False

        elif left_blinked or right_blinked:
            dt = abs(self._last_left_blink_time - self._last_right_blink_time) * 1000.0
            if 0 < dt < config.BLINK_BOTH_TOLERANCE_MS:
                left_deep = self._is_deep_blink(self._last_left_blink_depth, left_thresh)
                right_deep = self._is_deep_blink(self._last_right_blink_depth, right_thresh)
                if left_deep and right_deep:
                    both_blink = True
                    events.append(GestureEvent.BOTH_CLICK)
                    left_blinked = False
                    right_blinked = False

        if not both_blink:
            if left_blinked:
                if self._double_click.check_left(now):
                    events.append(GestureEvent.DOUBLE_LEFT_CLICK)
                else:
                    events.append(GestureEvent.LEFT_CLICK)

            if right_blinked:
                if self._double_click.check_right(now):
                    events.append(GestureEvent.DOUBLE_RIGHT_CLICK)
                else:
                    events.append(GestureEvent.RIGHT_CLICK)

        # Scroll via vertical gaze
        left_vg = compute_vertical_gaze_ratio(
            landmarks, config.LEFT_EYE_TOP_BOTTOM[0], config.LEFT_EYE_TOP_BOTTOM[1],
            left_iris_center_y, frame_h,
        )
        right_vg = compute_vertical_gaze_ratio(
            landmarks, config.RIGHT_EYE_TOP_BOTTOM[0], config.RIGHT_EYE_TOP_BOTTOM[1],
            right_iris_center_y, frame_h,
        )
        avg_vg = (left_vg + right_vg) / 2.0

        looking_up = avg_vg < config.VERTICAL_GAZE_UP_THRESHOLD
        looking_down = avg_vg > config.VERTICAL_GAZE_DOWN_THRESHOLD
        extreme_gaze = looking_up or looking_down

        if not self._scroll_mode:
            if extreme_gaze:
                self._scroll_active_frames += 1
                self._scroll_direction = -1 if looking_up else 1
                if self._scroll_active_frames >= config.SCROLL_ACTIVATION_FRAMES:
                    self._scroll_mode = True
                    self._scroll_neutral_frames = 0
            else:
                self._scroll_active_frames = 0
        else:
            if extreme_gaze:
                self._scroll_neutral_frames = 0
                direction = -1 if looking_up else 1
                if direction == self._scroll_direction:
                    events.append(GestureEvent.SCROLL_UP if direction < 0 else GestureEvent.SCROLL_DOWN)
            else:
                self._scroll_neutral_frames += 1
                if self._scroll_neutral_frames >= config.SCROLL_DEACTIVATION_FRAMES:
                    self._scroll_mode = False
                    self._scroll_active_frames = 0

        return events

    def reset(self) -> None:
        self._left_eye.reset()
        self._right_eye.reset()
        self._double_click.reset()
        self._last_left_blink_time = 0.0
        self._last_right_blink_time = 0.0
        self._last_left_blink_depth = 1.0
        self._last_right_blink_depth = 1.0
        self._left_suppress_until = 0.0
        self._right_suppress_until = 0.0
        self._scroll_mode = False
        self._scroll_direction = 0
        self._scroll_active_frames = 0
        self._scroll_neutral_frames = 0
