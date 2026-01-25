"""Fuse L2CS gaze and iris gaze with a Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

from src.face_detector import FaceDetectionResult
from src.gaze_estimator import GazeFrameResult
from src.iris_gaze import IrisGazeFrameResult, EyeGazeVector


@dataclass
class GazeFusionConfig:
    """Configuration for fusion Kalman filter and visualization."""

    line_length: Optional[int] = None  # If None, uses bbox width * line_scale
    line_scale: float = 1.0
    line_thickness: int = 2
    line_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    iris_scale: float = 6.0  # Used when L2CS length is unavailable
    process_noise: float = 1.0
    l2cs_meas_noise: float = 25.0
    iris_meas_noise: float = 9.0


@dataclass
class GazeFusionFrameResult:
    """Per-frame fused 2D gaze vector."""

    frame_index: int
    dx: float
    dy: float
    angle: float
    amplitude: float


class GazeFusionEstimator:
    """Kalman filter fusion for L2CS and iris gaze."""

    def __init__(self, config: GazeFusionConfig) -> None:
        """Initialize fusion parameters and filter state."""
        self.config = config
        self.state: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None

    def _vector_from_l2cs(
        self,
        face_bbox: Tuple[int, int, int, int],
        pitch: float,
        yaw: float,
    ) -> Tuple[float, float, float]:
        """Convert L2CS pitch/yaw into a 2D vector for fusion."""
        x1, _, x2, _ = face_bbox
        bbox_width = max(1, x2 - x1)
        length = self.config.line_length
        if length is None:
            length = int(bbox_width * self.config.line_scale)

        dx = -length * math.sin(pitch) * math.cos(yaw)
        dy = -length * math.sin(yaw)
        return float(dx), float(dy), float(length)

    def _normalize(self, dx: float, dy: float) -> Tuple[float, float, float]:
        """Normalize a 2D vector and return unit direction + magnitude."""
        amp = math.hypot(dx, dy)
        if amp <= 1e-6:
            return 0.0, 0.0, 0.0
        return dx / amp, dy / amp, amp

    def _init_state(self, dx: float, dy: float) -> None:
        """Initialize the Kalman state from a measurement."""
        self.state = np.array([[dx], [dy], [0.0], [0.0]], dtype=np.float32)
        self.cov = np.eye(4, dtype=np.float32) * 100.0

    def _predict(self, dt: float) -> None:
        """Predict step for constant-velocity Kalman model."""
        if self.state is None or self.cov is None:
            return
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        q = np.eye(4, dtype=np.float32) * self.config.process_noise
        self.state = f @ self.state
        self.cov = f @ self.cov @ f.T + q

    def _update(self, dx: float, dy: float, meas_noise: float) -> None:
        """Update step for a 2D position measurement."""
        if self.state is None or self.cov is None:
            self._init_state(dx, dy)
            return
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        r = np.eye(2, dtype=np.float32) * meas_noise
        z = np.array([[dx], [dy]], dtype=np.float32)

        y = z - (h @ self.state)
        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + (k @ y)
        i = np.eye(4, dtype=np.float32)
        self.cov = (i - k @ h) @ self.cov

    def _get_anchor(
        self,
        face_bbox: Tuple[int, int, int, int],
        iris_result: Optional[IrisGazeFrameResult],
    ) -> Tuple[float, float]:
        """Choose the vector origin: iris midpoint or face bbox center."""
        if iris_result is not None and iris_result.combined is not None:
            return iris_result.combined.eye_center
        x1, y1, x2, y2 = face_bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def run(
        self,
        video_path: Path,
        face_result: FaceDetectionResult,
        gaze_results: List[GazeFrameResult],
        iris_results: List[IrisGazeFrameResult],
        output_path: Path,
    ) -> List[GazeFusionFrameResult]:
        """Fuse two gaze sources, draw fused vector, and return per-frame results."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        dt = 1.0 / fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        results: List[GazeFusionFrameResult] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self._predict(dt)

            l2cs_dx = l2cs_dy = None
            l2cs_length = None
            if frame_index < len(gaze_results):
                g = gaze_results[frame_index]
                if not (math.isnan(g.pitch) or math.isnan(g.yaw)):
                    l2cs_dx, l2cs_dy, l2cs_length = self._vector_from_l2cs(face_result.bbox, g.pitch, g.yaw)

            iris_dx = iris_dy = None
            iris_frame = iris_results[frame_index] if frame_index < len(iris_results) else None
            if iris_frame is not None and iris_frame.combined is not None:
                c = iris_frame.combined
                unit_x, unit_y, amp = self._normalize(c.dx, c.dy)
                if l2cs_length is not None:
                    iris_dx = unit_x * l2cs_length
                    iris_dy = unit_y * l2cs_length
                elif amp > 0:
                    iris_dx = unit_x * amp * self.config.iris_scale
                    iris_dy = unit_y * amp * self.config.iris_scale

            if l2cs_dx is not None and l2cs_dy is not None:
                self._update(l2cs_dx, l2cs_dy, self.config.l2cs_meas_noise)
            if iris_dx is not None and iris_dy is not None:
                self._update(iris_dx, iris_dy, self.config.iris_meas_noise)

            if self.state is None:
                writer.write(frame)
                results.append(
                    GazeFusionFrameResult(
                        frame_index=frame_index,
                        dx=float("nan"),
                        dy=float("nan"),
                        angle=float("nan"),
                        amplitude=float("nan"),
                    )
                )
                frame_index += 1
                continue

            dx = float(self.state[0, 0])
            dy = float(self.state[1, 0])
            angle = float(math.atan2(dy, dx))
            amplitude = float(math.hypot(dx, dy))

            anchor = self._get_anchor(face_result.bbox, iris_frame)
            end = (int(round(anchor[0] + dx)), int(round(anchor[1] + dy)))
            start = (int(round(anchor[0])), int(round(anchor[1])))
            cv2.arrowedLine(frame, start, end, self.config.line_color, self.config.line_thickness, cv2.LINE_AA)

            writer.write(frame)
            results.append(
                GazeFusionFrameResult(
                    frame_index=frame_index,
                    dx=dx,
                    dy=dy,
                    angle=angle,
                    amplitude=amplitude,
                )
            )

            frame_index += 1

        cap.release()
        writer.release()

        return results
