"""
Project:Charm — Webcam Capture Abstraction Layer
"""

import logging
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class WebcamCapture:
    """Abstraction over cv2.VideoCapture for consistent webcam access."""

    def __init__(self, camera_index: int = config.CAMERA_INDEX) -> None:
        logger.info("Opening webcam (index=%d)...", camera_index)
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Ensure no other application is using it "
                "and that camera permissions are granted."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.FRAME_RATE)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Webcam opened: %dx%d @ %.1f fps (requested %dx%d @ %d fps)",
            actual_w, actual_h, actual_fps,
            config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FRAME_RATE,
        )

    def read(self) -> Optional[np.ndarray]:
        """Read a single BGR frame, or None on failure."""
        success, frame = self._cap.read()
        if not success or frame is None:
            logger.warning("Failed to read frame from webcam")
            return None
        return frame

    def release(self) -> None:
        """Release the webcam device."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Webcam released")

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
