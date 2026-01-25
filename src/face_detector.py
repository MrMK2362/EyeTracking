"""Face detection and fixed crop generation using YOLOv8 face model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


@dataclass
class FaceDetectorConfig:
    """Configuration for face detection inference."""

    model_path: Path
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None


@dataclass
class FaceDetectionResult:
    """Result of a face detection pass with a stable crop box."""

    bbox: Tuple[int, int, int, int]
    crop_width: int
    crop_height: int


class FaceDetector:
    """Detects the dominant face and returns a union crop box across frames."""

    def __init__(self, config: FaceDetectorConfig) -> None:
        """Load the YOLO model once and store detection settings."""
        self.config = config
        self.model = YOLO(str(config.model_path))

    def _find_union_bbox(self, video_path: Path) -> Tuple[Tuple[int, int, int, int], int, int]:
        """Scan the video and build a union bounding box over all frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        left = None
        top = None
        right = None
        bottom = None
        saw_detection = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_width <= 0 or frame_height <= 0:
                frame_height, frame_width = frame.shape[:2]

            results = self.model.predict(
                source=frame,
                conf=self.config.conf,
                iou=self.config.iou,
                imgsz=self.config.imgsz,
                device=self.config.device,
                verbose=False,
            )
            if not results:
                continue

            boxes = results[0].boxes
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue

            # Pick the largest face per frame, then union across frames.
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = int(areas.argmax().item())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

            if not saw_detection:
                left, top, right, bottom = x1, y1, x2, y2
                saw_detection = True
            else:
                left = min(left, x1)
                top = min(top, y1)
                right = max(right, x2)
                bottom = max(bottom, y2)

        cap.release()

        if not saw_detection:
            raise RuntimeError("No faces detected in the video.")

        return (int(left), int(top), int(right), int(bottom)), frame_width, frame_height

    def detect(self, video_path: Path) -> FaceDetectionResult:
        """Return the stabilized face crop bbox and its output dimensions."""
        bbox, frame_width, frame_height = self._find_union_bbox(video_path)
        x1, y1, x2, y2 = bbox

        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError("Could not determine video frame size.")

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(1, min(x2, frame_width))
        y2 = max(1, min(y2, frame_height))

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)

        return FaceDetectionResult(bbox=(x1, y1, x2, y2), crop_width=crop_w, crop_height=crop_h)
