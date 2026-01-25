"""Render a 3D scene from landmarks, head pose, and fused gaze."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

from src.face_landmarks import FaceLandmarksFrameResult
from src.gaze_fusion import GazeFusionFrameResult
from src.head_pose import HeadPoseFrameResult


@dataclass
class Scene3DConfig:
    """Configuration for 3D scene projection and rendering."""
    landmark_scale: float = 1.0
    z_scale: float = 1.0
    depth_offset: float = 800.0
    point_radius: int = 1
    landmark_color: Tuple[int, int, int] = (0, 255, 0)  # BGR green
    gaze_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    gaze_length: float = 200.0
    gaze_angle_scale: float = 500.0  # pixels -> radians via atan2
    line_thickness: int = 2
    camera_yaw_deg: float = 0.0
    camera_pitch_deg: float = 0.0
    camera_roll_deg: float = 0.0
    axis_length: float = 120.0
    axis_thickness: int = 2
    axis_x_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    axis_y_color: Tuple[int, int, int] = (0, 255, 0)  # BGR green
    axis_z_color: Tuple[int, int, int] = (255, 0, 0)  # BGR blue
    head_dist_process_noise: float = 25.0
    head_dist_meas_noise: float = 100.0


@dataclass
class Scene3DFrameResult:
    """Per-frame gaze angles used for the 3D gaze vector."""
    frame_index: int
    gaze_yaw: float
    gaze_pitch: float
    gaze_length: float


class Scene3DEstimator:
    """Projects landmarks and gaze into a simple 3D visualization."""

    def __init__(self, config: Scene3DConfig) -> None:
        """Store scene configuration."""
        self.config = config

    def _rotation_from_camera(self) -> np.ndarray:
        """Create a camera rotation matrix from yaw/pitch/roll."""
        yaw = math.radians(self.config.camera_yaw_deg)
        pitch = math.radians(self.config.camera_pitch_deg)
        roll = math.radians(self.config.camera_roll_deg)

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)

        return rz @ ry @ rx

    def _rotation_from_head_pose(self, rvec: Optional[Tuple[float, float, float]]) -> Optional[np.ndarray]:
        """Convert an rvec into a rotation matrix (if valid)."""
        if rvec is None:
            return None
        if any(math.isnan(v) for v in rvec):
            return None
        rvec_np = np.array(rvec, dtype=np.float32).reshape(3, 1)
        rot, _ = cv2.Rodrigues(rvec_np)
        return rot.astype(np.float32)

    def _project_points(
        self,
        points: np.ndarray,
        width: int,
        height: int,
        camera_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        """Project 3D points into 2D using a pinhole camera model."""
        if camera_matrix is None:
            fx = fy = float(max(width, height))
            cx = width / 2.0
            cy = height / 2.0
        else:
            fx = float(camera_matrix[0, 0])
            fy = float(camera_matrix[1, 1])
            cx = float(camera_matrix[0, 2])
            cy = float(camera_matrix[1, 2])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2] + self.config.depth_offset
        z = np.clip(z, 1.0, None)

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        return np.stack([u, v], axis=1)

    def run(
        self,
        video_path: Path,
        landmarks_results: List[FaceLandmarksFrameResult],
        head_pose_results: List[HeadPoseFrameResult],
        fusion_results: List[GazeFusionFrameResult],
        camera_matrix: Optional[np.ndarray],
        output_path: Path,
    ) -> List[Scene3DFrameResult]:
        """Render a 3D landmark scene with head axes and gaze vector."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = min(len(landmarks_results), len(fusion_results), len(head_pose_results))
        results: List[Scene3DFrameResult] = []
        head_dist = None
        head_cov = None

        for idx in range(frame_count):
            lm_frame = landmarks_results[idx]
            hp_frame = head_pose_results[idx]
            fuse_frame = fusion_results[idx]

            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if hp_frame.tvec is not None and not any(math.isnan(v) for v in hp_frame.tvec):
                tx, ty, tz = hp_frame.tvec
                dist = math.sqrt(tx * tx + ty * ty + tz * tz)
                if head_dist is None:
                    head_dist = dist
                    head_cov = 1.0
                else:
                    head_cov = head_cov + self.config.head_dist_process_noise
                    k = head_cov / (head_cov + self.config.head_dist_meas_noise)
                    head_dist = head_dist + k * (dist - head_dist)
                    head_cov = (1.0 - k) * head_cov
                print(f"[Scene3D] Frame {idx}: head distance = {head_dist:.2f} mm")

            if not lm_frame.landmarks:
                writer.write(canvas)
                results.append(
                    Scene3DFrameResult(
                        frame_index=idx,
                        gaze_yaw=float("nan"),
                        gaze_pitch=float("nan"),
                        gaze_length=float("nan"),
                    )
                )
                continue

            points = np.array([[x, y, z] for x, y, z in lm_frame.landmarks], dtype=np.float32)
            center = points.mean(axis=0)
            points = (points - center) * self.config.landmark_scale
            points[:, 2] *= self.config.z_scale

            rot_cam = self._rotation_from_camera()
            points = (rot_cam @ points.T).T

            projected = self._project_points(points, width, height, camera_matrix)
            for px, py in projected:
                cv2.circle(
                    canvas,
                    (int(round(px)), int(round(py))),
                    self.config.point_radius,
                    self.config.landmark_color,
                    -1,
                    cv2.LINE_AA,
                )

            # Gaze vector from Kalman output -> direction + amplitude angle (degrees -> radians)
            if math.isnan(fuse_frame.dx) or math.isnan(fuse_frame.dy):
                gaze_yaw = float("nan")
                gaze_pitch = float("nan")
                gaze_length = float("nan")
            else:
                direction = fuse_frame.angle
                amp_deg = max(0.0, min(90.0, float(fuse_frame.amplitude)))
                gaze_yaw = direction
                gaze_pitch = math.radians(amp_deg)
                if head_dist is None:
                    gaze_length = self.config.gaze_length
                else:
                    gaze_length = float(head_dist)

                dir_vec = np.array(
                    [
                        math.sin(gaze_pitch) * math.cos(gaze_yaw),
                        math.sin(gaze_pitch) * math.sin(gaze_yaw),
                        -math.cos(gaze_pitch),
                    ],
                    dtype=np.float32,
                )
                dir_vec *= gaze_length
                dir_vec = rot_cam @ dir_vec
                
                start_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                end_3d = dir_vec
                proj = self._project_points(np.stack([start_3d, end_3d], axis=0), width, height, camera_matrix)
                start = tuple(np.round(proj[0]).astype(int))
                end = tuple(np.round(proj[1]).astype(int))
                cv2.arrowedLine(canvas, start, end, self.config.gaze_color, self.config.line_thickness, cv2.LINE_AA)

            # Draw head-pose axis in scene (from solvePnP)
            rot_head = self._rotation_from_head_pose(hp_frame.rvec)
            if rot_head is not None:
                axis = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [self.config.axis_length, 0.0, 0.0],
                        [0.0, self.config.axis_length, 0.0],
                        [0.0, 0.0, self.config.axis_length],
                    ],
                    dtype=np.float32,
                )
                axis = (rot_cam @ (rot_head @ axis.T)).T
                proj = self._project_points(axis, width, height, camera_matrix)
                origin = tuple(np.round(proj[0]).astype(int))
                x_end = tuple(np.round(proj[1]).astype(int))
                y_end = tuple(np.round(proj[2]).astype(int))
                z_end = tuple(np.round(proj[3]).astype(int))
                cv2.line(canvas, origin, x_end, self.config.axis_x_color, self.config.axis_thickness)
                cv2.line(canvas, origin, y_end, self.config.axis_y_color, self.config.axis_thickness)
                cv2.line(canvas, origin, z_end, self.config.axis_z_color, self.config.axis_thickness)

            writer.write(canvas)
            results.append(
                Scene3DFrameResult(
                    frame_index=idx,
                    gaze_yaw=gaze_yaw,
                    gaze_pitch=gaze_pitch,
                    gaze_length=gaze_length,
                )
            )

        writer.release()

        return results
