"""MediaPipe FaceMesh landmarks and colored landmark rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import inspect

import cv2
import mediapipe as mp
import numpy as np

from src.face_detector import FaceDetectionResult


@dataclass
class FaceLandmarksConfig:
    """Configuration for MediaPipe face landmarks."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    refine_landmarks: bool = True  # Enables iris landmarks
    point_radius: int = 1
    eye_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red
    face_color: Tuple[int, int, int] = (0, 255, 0)  # BGR green


@dataclass
class FaceLandmarksFrameResult:
    """Per-frame 3D landmarks mapped to full-frame pixel space."""

    frame_index: int
    landmarks: List[Tuple[float, float, float]]  # (x, y, z) in full-frame pixel coords


class FaceLandmarksEstimator:
    """Extracts FaceMesh landmarks inside a face crop and draws them."""

    def __init__(self, config: FaceLandmarksConfig) -> None:
        """Initialize MediaPipe FaceMesh and cache landmark index groups."""
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        mesh_kwargs = {
            "static_image_mode": False,
            "max_num_faces": 1,
            "refine_landmarks": config.refine_landmarks,
            "min_detection_confidence": config.min_detection_confidence,
            "min_tracking_confidence": config.min_tracking_confidence,
            "model_complexity": config.model_complexity,
        }
        supported = set(inspect.signature(self.mp_face_mesh.FaceMesh).parameters.keys())
        mesh_kwargs = {k: v for k, v in mesh_kwargs.items() if k in supported}
        self.face_mesh = self.mp_face_mesh.FaceMesh(**mesh_kwargs)

        self.eye_indices = self._build_eye_indices()

    def _build_eye_indices(self) -> set[int]:
        """Collect indices for eyes, irises, and eyebrows for coloring."""
        # Collect indices for eyes, irises, and eyebrows
        eye_sets = [
            self.mp_face_mesh.FACEMESH_LEFT_EYE,
            self.mp_face_mesh.FACEMESH_RIGHT_EYE,
            self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            self.mp_face_mesh.FACEMESH_LEFT_IRIS,
            self.mp_face_mesh.FACEMESH_RIGHT_IRIS,
        ]
        indices: set[int] = set()
        for conn_set in eye_sets:
            for a, b in conn_set:
                indices.add(a)
                indices.add(b)
        return indices

    def run(
        self,
        video_path: Path,
        face_result: FaceDetectionResult,
        output_path: Path,
    ) -> List[FaceLandmarksFrameResult]:
        """Process a video, draw landmarks, and return per-frame points."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1, y1, x2, y2 = face_result.bbox
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(1, min(x2, frame_width))
        y2 = max(1, min(y2, frame_height))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        results: List[FaceLandmarksFrameResult] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                results.append(FaceLandmarksFrameResult(frame_index=frame_index, landmarks=[]))
                writer.write(frame)
                frame_index += 1
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(face_rgb)

            frame_landmarks: List[Tuple[float, float, float]] = []
            if output.multi_face_landmarks:
                landmarks = output.multi_face_landmarks[0].landmark
                for idx, lm in enumerate(landmarks):
                    px = x1 + lm.x * (x2 - x1)
                    py = y1 + lm.y * (y2 - y1)
                    pz = lm.z * (x2 - x1)

                    frame_landmarks.append((px, py, pz))

                    color = self.config.eye_color if idx in self.eye_indices else self.config.face_color
                    cv2.circle(
                        frame,
                        (int(round(px)), int(round(py))),
                        self.config.point_radius,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

            results.append(FaceLandmarksFrameResult(frame_index=frame_index, landmarks=frame_landmarks))
            writer.write(frame)
            frame_index += 1

        cap.release()
        writer.release()

        return results
