"""Head pose estimation using solvePnP and landmark correspondences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.face_landmarks import FaceLandmarksFrameResult


@dataclass
class HeadPoseConfig:
    """Configuration for head axis rendering."""

    axis_length: float = 60.0
    line_thickness: int = 2
    x_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    y_color: Tuple[int, int, int] = (0, 255, 0)  # BGR green
    z_color: Tuple[int, int, int] = (255, 0, 0)  # BGR blue


@dataclass
class HeadPoseFrameResult:
    """Per-frame head pose from solvePnP (rvec/tvec and Euler)."""

    frame_index: int
    rvec: Tuple[float, float, float]
    tvec: Tuple[float, float, float]
    yaw: float
    pitch: float
    roll: float


class HeadPoseEstimator:
    """Estimates head rotation with solvePnP and draws 3D axes."""

    # MediaPipe landmark indices (FaceMesh): https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
    LANDMARK_IDXS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 263,
        "right_eye_outer": 33,
        "left_mouth": 287,
        "right_mouth": 57,
    }

    def __init__(self, config: HeadPoseConfig) -> None:
        """Store rendering configuration for axes."""
        self.config = config

    def _get_image_points(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> np.ndarray | None:
        """Select the minimal 2D landmark set needed for solvePnP."""
        if not landmarks:
            return None

        try:
            points = np.array(
                [
                    landmarks[self.LANDMARK_IDXS["nose_tip"]][:2],
                    landmarks[self.LANDMARK_IDXS["chin"]][:2],
                    landmarks[self.LANDMARK_IDXS["left_eye_outer"]][:2],
                    landmarks[self.LANDMARK_IDXS["right_eye_outer"]][:2],
                    landmarks[self.LANDMARK_IDXS["left_mouth"]][:2],
                    landmarks[self.LANDMARK_IDXS["right_mouth"]][:2],
                ],
                dtype=np.float32,
            )
        except IndexError:
            return None

        return points

    def _get_model_points(self) -> np.ndarray:
        """Return a standard 3D face model (mm) for solvePnP."""
        # Standard 3D model points (approximate, in mm)
        return np.array(
            [
                (0.0, 0.0, 0.0),        # nose tip
                (0.0, -63.6, -12.5),    # chin
                (-43.3, 32.7, -26.0),   # left eye outer
                (43.3, 32.7, -26.0),    # right eye outer
                (-28.9, -28.9, -24.1),  # left mouth corner
                (28.9, -28.9, -24.1),   # right mouth corner
            ],
            dtype=np.float32,
        )

    def _rvec_to_euler(self, rvec: np.ndarray) -> Tuple[float, float, float]:
        """Convert Rodrigues rotation to Euler angles (yaw, pitch, roll)."""
        rot_mat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)

        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = 0.0

        return float(yaw), float(pitch), float(roll)

    def _draw_axis(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        """Project and draw 3D XYZ axes on the frame."""
        axis = np.float32(
            [
                [self.config.axis_length, 0, 0],
                [0, self.config.axis_length, 0],
                [0, 0, self.config.axis_length],
            ]
        )
        origin = np.float32([[0, 0, 0]])
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
        axis_2d, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

        o = tuple(origin_2d[0].ravel().astype(int))
        x = tuple(axis_2d[0].ravel().astype(int))
        y = tuple(axis_2d[1].ravel().astype(int))
        z = tuple(axis_2d[2].ravel().astype(int))

        cv2.line(frame, o, x, self.config.x_color, self.config.line_thickness)
        cv2.line(frame, o, y, self.config.y_color, self.config.line_thickness)
        cv2.line(frame, o, z, self.config.z_color, self.config.line_thickness)

    def run(
        self,
        video_path: Path,
        landmarks_results: List[FaceLandmarksFrameResult],
        camera_matrix: np.ndarray,
        output_path: Path,
    ) -> List[HeadPoseFrameResult]:
        """Estimate head pose per frame and save an annotated video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        model_points = self._get_model_points()
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        results: List[HeadPoseFrameResult] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index >= len(landmarks_results):
                writer.write(frame)
                frame_index += 1
                continue

            landmark_frame = landmarks_results[frame_index]
            image_points = self._get_image_points(landmark_frame.landmarks)

            if image_points is None or len(image_points) != 6:
                results.append(
                    HeadPoseFrameResult(
                        frame_index=frame_index,
                        rvec=(float("nan"),) * 3,
                        tvec=(float("nan"),) * 3,
                        yaw=float("nan"),
                        pitch=float("nan"),
                        roll=float("nan"),
                    )
                )
                writer.write(frame)
                frame_index += 1
                continue

            success, rvec, tvec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                results.append(
                    HeadPoseFrameResult(
                        frame_index=frame_index,
                        rvec=(float("nan"),) * 3,
                        tvec=(float("nan"),) * 3,
                        yaw=float("nan"),
                        pitch=float("nan"),
                        roll=float("nan"),
                    )
                )
                writer.write(frame)
                frame_index += 1
                continue

            yaw, pitch, roll = self._rvec_to_euler(rvec)
            results.append(
                HeadPoseFrameResult(
                    frame_index=frame_index,
                    rvec=tuple(float(v) for v in rvec.flatten()),
                    tvec=tuple(float(v) for v in tvec.flatten()),
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )

            self._draw_axis(frame, camera_matrix, rvec, tvec)
            writer.write(frame)

            frame_index += 1

        cap.release()
        writer.release()

        return results
