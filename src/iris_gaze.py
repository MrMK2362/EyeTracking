"""Iris-based gaze estimation using eye/iris landmarks and head de-rotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

import cv2
import mediapipe as mp
import numpy as np

from src.face_landmarks import FaceLandmarksFrameResult
from src.head_pose import HeadPoseFrameResult


@dataclass
class IrisGazeConfig:
    """Configuration for iris gaze extraction and rendering."""

    vector_scale: float = 10.0
    line_thickness: int = 2
    line_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    eye_margin: int = 6
    iris_threshold: int = 25


@dataclass
class EyeGazeVector:
    """2D gaze vector with direction and amplitude."""

    dx: float
    dy: float
    angle: float
    amplitude: float
    eye_center: Tuple[float, float]
    iris_center: Tuple[float, float]


@dataclass
class IrisGazeFrameResult:
    """Per-frame iris gaze vectors and vergence."""

    frame_index: int
    left: Optional[EyeGazeVector]
    right: Optional[EyeGazeVector]
    combined: Optional[EyeGazeVector]
    vergence: Optional[float]


class IrisGazeEstimator:
    """Compute iris gaze vectors from FaceMesh landmarks."""

    def __init__(self, config: IrisGazeConfig) -> None:
        """Initialize landmark index sets and settings."""
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        (
            self.left_eye_indices,
            self.right_eye_indices,
            self.left_iris_indices,
            self.right_iris_indices,
        ) = self._build_eye_indices()

    def _build_eye_indices(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Collect eye and iris landmark indices used for gaze."""
        def collect(conn_set):
            idxs = set()
            for a, b in conn_set:
                idxs.add(a)
                idxs.add(b)
            return sorted(idxs)

        left_eye = collect(self.mp_face_mesh.FACEMESH_LEFT_EYE)
        right_eye = collect(self.mp_face_mesh.FACEMESH_RIGHT_EYE)
        left_iris = collect(self.mp_face_mesh.FACEMESH_LEFT_IRIS)
        right_iris = collect(self.mp_face_mesh.FACEMESH_RIGHT_IRIS)

        return left_eye, right_eye, left_iris, right_iris

    def _transform_points(self, pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """Apply a 2D affine transform to point coordinates."""
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        hom = np.hstack([pts, ones])
        return (mat @ hom.T).T

    def _bbox_from_points(self, pts: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        """Create a padded bounding box around a point set."""
        if pts.size == 0:
            return 0, 0, 0, 0
        x_min = int(max(0, np.min(pts[:, 0]) - self.config.eye_margin))
        y_min = int(max(0, np.min(pts[:, 1]) - self.config.eye_margin))
        x_max = int(min(width, np.max(pts[:, 0]) + self.config.eye_margin))
        y_max = int(min(height, np.max(pts[:, 1]) + self.config.eye_margin))
        return x_min, y_min, x_max, y_max

    def _detect_iris_center(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """Estimate iris center from low-intensity pixels in the eye ROI."""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)
        min_val, _, min_loc, _ = cv2.minMaxLoc(roi_blur)

        mask = roi_blur <= (min_val + self.config.iris_threshold)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return float(x1 + min_loc[0]), float(y1 + min_loc[1])

        cx = float(x1 + xs.mean())
        cy = float(y1 + ys.mean())
        return cx, cy

    def _to_vector(
        self,
        eye_center: Tuple[float, float],
        iris_center: Tuple[float, float],
    ) -> EyeGazeVector:
        """Convert eye/iris centers into a 2D gaze vector."""
        dx = iris_center[0] - eye_center[0]
        dy = iris_center[1] - eye_center[1]
        amplitude = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        return EyeGazeVector(
            dx=float(dx),
            dy=float(dy),
            angle=float(angle),
            amplitude=float(amplitude),
            eye_center=eye_center,
            iris_center=iris_center,
        )

    def _combine_vectors(
        self,
        left: Optional[EyeGazeVector],
        right: Optional[EyeGazeVector],
    ) -> Tuple[Optional[EyeGazeVector], Optional[float]]:
        """Combine left/right vectors into a binocular vector and vergence."""
        if left is None or right is None:
            return None, None

        eye_center = (
            (left.eye_center[0] + right.eye_center[0]) / 2.0,
            (left.eye_center[1] + right.eye_center[1]) / 2.0,
        )
        iris_center = (
            (left.iris_center[0] + right.iris_center[0]) / 2.0,
            (left.iris_center[1] + right.iris_center[1]) / 2.0,
        )

        combined = self._to_vector(eye_center, iris_center)
        vergence = right.angle - left.angle
        return combined, float(vergence)

    def run(
        self,
        video_path: Path,
        landmarks_results: List[FaceLandmarksFrameResult],
        head_pose_results: List[HeadPoseFrameResult],
        output_path: Path,
    ) -> List[IrisGazeFrameResult]:
        """Process a video and return iris gaze vectors per frame."""
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

        results: List[IrisGazeFrameResult] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index >= len(landmarks_results) or frame_index >= len(head_pose_results):
                writer.write(frame)
                frame_index += 1
                continue

            lm_frame = landmarks_results[frame_index]
            hp_frame = head_pose_results[frame_index]

            if not lm_frame.landmarks:
                results.append(IrisGazeFrameResult(frame_index=frame_index, left=None, right=None, combined=None, vergence=None))
                writer.write(frame)
                frame_index += 1
                continue

            roll = hp_frame.roll
            if roll is None or math.isnan(roll):
                roll = 0.0

            # De-rotate by roll (2D)
            pts = np.array([[x, y] for x, y, _ in lm_frame.landmarks], dtype=np.float32)
            if pts.size == 0:
                results.append(IrisGazeFrameResult(frame_index=frame_index, left=None, right=None, combined=None, vergence=None))
                writer.write(frame)
                frame_index += 1
                continue

            center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            rot_deg = -math.degrees(roll)
            rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
            rot_inv = cv2.invertAffineTransform(rot_mat)

            rotated = cv2.warpAffine(frame, rot_mat, (width, height))
            rotated_pts = self._transform_points(pts, rot_mat)

            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

            left_eye_pts = np.array(
                [rotated_pts[i] for i in self.left_eye_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            right_eye_pts = np.array(
                [rotated_pts[i] for i in self.right_eye_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            left_iris_pts = np.array(
                [rotated_pts[i] for i in self.left_iris_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            right_iris_pts = np.array(
                [rotated_pts[i] for i in self.right_iris_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )

            left_bbox = self._bbox_from_points(left_eye_pts, width, height)
            right_bbox = self._bbox_from_points(right_eye_pts, width, height)

            left_center = None
            right_center = None

            if left_eye_pts.size != 0:
                left_center = (float(left_eye_pts[:, 0].mean()), float(left_eye_pts[:, 1].mean()))
            if right_eye_pts.size != 0:
                right_center = (float(right_eye_pts[:, 0].mean()), float(right_eye_pts[:, 1].mean()))

            left_iris = None
            right_iris = None

            # Iris center from landmarks (preferred)
            if left_iris_pts.size != 0:
                left_iris = (float(left_iris_pts[:, 0].mean()), float(left_iris_pts[:, 1].mean()))
            if right_iris_pts.size != 0:
                right_iris = (float(right_iris_pts[:, 0].mean()), float(right_iris_pts[:, 1].mean()))

            if left_iris is None and left_center is not None:
                left_iris = self._detect_iris_center(gray, left_bbox)
            if right_iris is None and right_center is not None:
                right_iris = self._detect_iris_center(gray, right_bbox)

            left_vec = None
            right_vec = None

            if left_center is not None and left_iris is not None:
                eye_orig = self._transform_points(np.array([left_center], dtype=np.float32), rot_inv)[0]
                iris_orig = self._transform_points(np.array([left_iris], dtype=np.float32), rot_inv)[0]
                left_vec = self._to_vector(tuple(eye_orig), tuple(iris_orig))
            if right_center is not None and right_iris is not None:
                eye_orig = self._transform_points(np.array([right_center], dtype=np.float32), rot_inv)[0]
                iris_orig = self._transform_points(np.array([right_iris], dtype=np.float32), rot_inv)[0]
                right_vec = self._to_vector(tuple(eye_orig), tuple(iris_orig))

            combined_vec, vergence = self._combine_vectors(left_vec, right_vec)

            # Draw on original frame
            if left_vec is not None:
                dx = left_vec.dx * self.config.vector_scale
                dy = left_vec.dy * self.config.vector_scale
                end = (int(round(left_vec.eye_center[0] + dx)), int(round(left_vec.eye_center[1] + dy)))
                start = (int(round(left_vec.eye_center[0])), int(round(left_vec.eye_center[1])))
                cv2.arrowedLine(frame, start, end, self.config.line_color, self.config.line_thickness, cv2.LINE_AA)

            if right_vec is not None:
                dx = right_vec.dx * self.config.vector_scale
                dy = right_vec.dy * self.config.vector_scale
                end = (int(round(right_vec.eye_center[0] + dx)), int(round(right_vec.eye_center[1] + dy)))
                start = (int(round(right_vec.eye_center[0])), int(round(right_vec.eye_center[1])))
                cv2.arrowedLine(frame, start, end, self.config.line_color, self.config.line_thickness, cv2.LINE_AA)

            if combined_vec is not None:
                dx = combined_vec.dx * self.config.vector_scale
                dy = combined_vec.dy * self.config.vector_scale
                end = (int(round(combined_vec.eye_center[0] + dx)), int(round(combined_vec.eye_center[1] + dy)))
                start = (int(round(combined_vec.eye_center[0])), int(round(combined_vec.eye_center[1])))
                cv2.arrowedLine(frame, start, end, self.config.line_color, self.config.line_thickness, cv2.LINE_AA)

            writer.write(frame)

            results.append(
                IrisGazeFrameResult(
                    frame_index=frame_index,
                    left=left_vec,
                    right=right_vec,
                    combined=combined_vec,
                    vergence=vergence,
                )
            )

            frame_index += 1

        cap.release()
        writer.release()

        return results
