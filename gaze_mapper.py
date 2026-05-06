"""
Project:Charm — Gaze-to-Screen Coordinate Mapper
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


class _LowPassFilter:
    """First-order low-pass (EMA) filter for scalar values."""

    def __init__(self) -> None:
        self._prev: Optional[float] = None

    def apply(self, value: float, alpha: float) -> float:
        if self._prev is None:
            self._prev = value
        else:
            self._prev = alpha * value + (1.0 - alpha) * self._prev
        return self._prev

    def reset(self) -> None:
        self._prev = None

    @property
    def has_prev(self) -> bool:
        return self._prev is not None

    @property
    def prev(self) -> float:
        return self._prev if self._prev is not None else 0.0


class OneEuroFilter:
    """
    One Euro Filter (Casiez et al., CHI 2012) for adaptive 2D smoothing.
    Smooth when still, responsive when moving fast.
    """

    def __init__(
        self,
        freq: float = config.ONE_EURO_FREQ,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filt = _LowPassFilter()
        self._dx_filt = _LowPassFilter()
        self._y_filt = _LowPassFilter()
        self._dy_filt = _LowPassFilter()

    @staticmethod
    def _smoothing_factor(freq: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def _filter_axis(
        self, value: float, x_filt: _LowPassFilter, dx_filt: _LowPassFilter,
    ) -> float:
        if x_filt.has_prev:
            d_value = (value - x_filt.prev) * self.freq
        else:
            d_value = 0.0

        alpha_d = self._smoothing_factor(self.freq, self.d_cutoff)
        d_hat = dx_filt.apply(d_value, alpha_d)

        cutoff = self.min_cutoff + self.beta * abs(d_hat)
        alpha = self._smoothing_factor(self.freq, cutoff)

        return x_filt.apply(value, alpha)

    def update(self, new_position: np.ndarray) -> np.ndarray:
        """Filter a 2D position. Returns smoothed [x, y]."""
        sx = self._filter_axis(float(new_position[0]), self._x_filt, self._dx_filt)
        sy = self._filter_axis(float(new_position[1]), self._y_filt, self._dy_filt)
        return np.array([sx, sy], dtype=np.float32)

    def reset(self) -> None:
        self._x_filt.reset()
        self._dx_filt.reset()
        self._y_filt.reset()
        self._dy_filt.reset()


class DeadZoneFilter:
    """Suppresses cursor movements below a pixel threshold."""

    def __init__(self, threshold: float = config.DEAD_ZONE_PIXELS) -> None:
        self.threshold = threshold
        self._last_position: Optional[np.ndarray] = None

    def apply(self, position: np.ndarray) -> np.ndarray:
        if self._last_position is None:
            self._last_position = position.copy()
            return position.copy()

        delta = float(np.linalg.norm(position - self._last_position))
        if delta < self.threshold:
            return self._last_position.copy()

        self._last_position = position.copy()
        return position.copy()

    def reset(self) -> None:
        self._last_position = None


def map_gaze_to_screen(
    iris_x: float, iris_y: float,
    gaze_min_x: float, gaze_max_x: float,
    gaze_min_y: float, gaze_max_y: float,
    screen_w: int, screen_h: int,
) -> Tuple[int, int]:
    """Map iris pixel coordinates to screen coordinates via linear interpolation."""
    ix = float(np.clip(iris_x, gaze_min_x, gaze_max_x))
    iy = float(np.clip(iris_y, gaze_min_y, gaze_max_y))

    nx = (ix - gaze_min_x) / (gaze_max_x - gaze_min_x + 1e-6)
    ny = (iy - gaze_min_y) / (gaze_max_y - gaze_min_y + 1e-6)

    # Mirror X: webcam is horizontally flipped relative to user
    nx = 1.0 - nx

    screen_x = int(np.clip(nx * screen_w, 0, screen_w - 1))
    screen_y = int(np.clip(ny * screen_h, 0, screen_h - 1))

    return screen_x, screen_y
