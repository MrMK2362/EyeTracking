"""L2CS-Net based gaze estimation from a face crop."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.face_detector import FaceDetectionResult

L2CS_ROOT = Path(__file__).resolve().parents[1] / "models"
if str(L2CS_ROOT) not in sys.path:
    sys.path.insert(0, str(L2CS_ROOT))

from l2cs.utils import getArch  # noqa: E402


@dataclass
class GazeEstimatorConfig:
    """Configuration for L2CS gaze model and visualization."""

    weights_path: Path
    arch: str = "ResNet50"
    device: str = "cpu"  # Example: "cuda:0" or "cpu"
    line_length: Optional[int] = None  # If None, uses bbox width * line_scale
    line_scale: float = 1.0
    line_thickness: int = 2
    line_color: Tuple[int, int, int] = (0, 0, 255)  # BGR red


@dataclass
class GazeFrameResult:
    """Per-frame gaze angles in radians."""

    frame_index: int
    pitch: float
    yaw: float


class GazeEstimator:
    """Runs L2CS-Net over a face crop and draws a gaze arrow."""

    def __init__(self, config: GazeEstimatorConfig) -> None:
        """Load the L2CS model and prepare normalization."""
        self.config = config
        self.device = torch.device(config.device)

        self.model = getArch(config.arch, 90)
        self.model.load_state_dict(torch.load(str(config.weights_path), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _predict_gaze(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        """Predict pitch/yaw (radians) for a single face crop."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = self.transform(face_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.model(img)
            pitch_pred = self.softmax(gaze_pitch)
            yaw_pred = self.softmax(gaze_yaw)

            pitch_pred = torch.sum(pitch_pred * self.idx_tensor, dim=1) * 4 - 180
            yaw_pred = torch.sum(yaw_pred * self.idx_tensor, dim=1) * 4 - 180

            pitch = (pitch_pred.item() * math.pi) / 180.0
            yaw = (yaw_pred.item() * math.pi) / 180.0

        return pitch, yaw

    def _draw_gaze_line(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        """Render a gaze arrow on the frame using pitch/yaw."""
        x1, y1, x2, y2 = bbox
        bbox_width = max(1, x2 - x1)
        bbox_height = max(1, y2 - y1)

        length = self.config.line_length
        if length is None:
            length = int(bbox_width * self.config.line_scale)

        center = (int(x1 + bbox_width / 2.0), int(y1 + bbox_height / 2.0))
        dx = -length * math.sin(pitch) * math.cos(yaw)
        dy = -length * math.sin(yaw)

        end = (int(center[0] + dx), int(center[1] + dy))
        cv2.arrowedLine(
            frame,
            center,
            end,
            self.config.line_color,
            self.config.line_thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        return frame

    def run(
        self,
        video_path: Path,
        face_result: FaceDetectionResult,
        output_path: Path,
    ) -> List[GazeFrameResult]:
        """Process a video, draw gaze, and return per-frame angles."""
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

        results: List[GazeFrameResult] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                results.append(GazeFrameResult(frame_index=frame_index, pitch=float("nan"), yaw=float("nan")))
                writer.write(frame)
                frame_index += 1
                continue

            pitch, yaw = self._predict_gaze(face)
            results.append(GazeFrameResult(frame_index=frame_index, pitch=pitch, yaw=yaw))

            annotated = self._draw_gaze_line(frame, (x1, y1, x2, y2), pitch, yaw)
            writer.write(annotated)

            frame_index += 1

        cap.release()
        writer.release()

        return results
