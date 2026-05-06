"""
Project:Charm — MediaPipe Landmark Processor
"""

import logging
import os
from typing import Any, List, Optional

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


class _LandmarkListAdapter:
    """Wraps the tasks API landmark list for indexed access compatibility."""

    def __init__(self, landmarks: list) -> None:
        self._landmarks = landmarks

    def __getitem__(self, index: int) -> Any:
        return self._landmarks[index]

    def __len__(self) -> int:
        return len(self._landmarks)

    def __iter__(self):
        return iter(self._landmarks)


class LandmarkProcessor:
    """Extracts 478-point facial landmarks via MediaPipe FaceLandmarker."""

    def __init__(self) -> None:
        model_path = config.FACE_LANDMARKER_MODEL
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at '{model_path}'. "
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        logger.info("Initializing MediaPipe FaceLandmarker (model=%s)...", model_path)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=config.MAX_NUM_FACES,
            min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_timestamp_ms: int = 0
        self._prev_nose_tip: Optional[np.ndarray] = None
        logger.info("FaceLandmarker initialized successfully")

    def process(self, frame: np.ndarray) -> Optional[_LandmarkListAdapter]:
        """Run FaceLandmarker on a BGR frame. Returns landmarks or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33

        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            self._prev_nose_tip = None
            return None

        return _LandmarkListAdapter(result.face_landmarks[0])

    def check_stability(
        self, landmarks: _LandmarkListAdapter, frame_w: int, frame_h: int
    ) -> bool:
        """Reject frames with implausible nose-tip jumps (detection artifacts)."""
        nose = np.array(
            [landmarks[1].x * frame_w, landmarks[1].y * frame_h],
            dtype=np.float32,
        )

        if self._prev_nose_tip is None:
            self._prev_nose_tip = nose
            return True

        delta = float(np.linalg.norm(nose - self._prev_nose_tip))
        self._prev_nose_tip = nose

        if delta > config.LANDMARK_STABILITY_MAX_DELTA:
            logger.debug("Landmark instability: nose delta=%.1f px", delta)
            return False

        return True

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            logger.info("MediaPipe FaceLandmarker closed")

    @staticmethod
    def get_iris_center(
        landmarks: _LandmarkListAdapter,
        iris_indices: List[int],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """Compute pixel-space (x, y) center of the iris from 4 landmarks."""
        points = np.array(
            [
                [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
                for i in iris_indices
            ],
            dtype=np.float32,
        )
        return points.mean(axis=0)
