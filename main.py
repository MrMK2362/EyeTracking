"""Main pipeline runner for offline videos and live gaze tracking."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from src.face_detector import FaceDetector, FaceDetectorConfig, FaceDetectionResult
from src.gaze_estimator import GazeEstimator, GazeEstimatorConfig
from src.face_landmarks import FaceLandmarksEstimator, FaceLandmarksConfig
from src.head_pose import HeadPoseEstimator, HeadPoseConfig
from src.iris_gaze import IrisGazeEstimator, IrisGazeConfig, IrisGazeFrameResult
from src.gaze_fusion import GazeFusionEstimator, GazeFusionConfig
from src.scene_3d import Scene3DEstimator, Scene3DConfig


def find_input_video(input_dir: Path, flag: str) -> Path:
    """Resolve the input video path by flag and common extensions."""
    exts = [".mp4", ".avi", ".mov", ".mkv"]
    for ext in exts:
        candidate = input_dir / f"{flag}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No input video found for flag '{flag}'. Tried: {', '.join(exts)}"
    )


def compute_intrinsics(frame_width: int, frame_height: int) -> np.ndarray:
    """Compute an approximate camera intrinsics matrix for the given frame size."""
    # Base calibration for 1920x1080 with ~75° FOV
    base_width = 1920.0
    base_height = 1080.0
    base_fx = 1251.0
    base_fy = 1251.0

    scale_x = frame_width / base_width
    scale_y = frame_height / base_height

    fx = base_fx * scale_x
    fy = base_fy * scale_y
    cx = frame_width / 2.0
    cy = frame_height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def calibrate_video(
    video_path: Path,
    output_path: Path,
    camera_matrix: np.ndarray,
) -> Path:
    """Undistort the video using a simple camera matrix and zero distortion."""
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

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
        writer.write(undistorted)

    cap.release()
    writer.release()

    return output_path


def write_cropped_video(
    video_path: Path,
    output_path: Path,
    result: FaceDetectionResult,
) -> None:
    """Write a cropped face video from a stabilized bounding box."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (result.crop_width, result.crop_height),
    )

    x1, y1, x2, y2 = result.bbox
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        if cropped.shape[1] != result.crop_width or cropped.shape[0] != result.crop_height:
            cropped = cv2.resize(cropped, (result.crop_width, result.crop_height))

        writer.write(cropped)

    cap.release()
    writer.release()


def main() -> None:
    """Run the full offline pipeline or start live mode if flag=0."""
    parser = argparse.ArgumentParser(description="Run the face detector pipeline.")
    parser.add_argument("--flag", required=True, help="Input video name without extension (e.g., 1)")
    parser.add_argument("--input-dir", default="input", help="Input folder path")
    parser.add_argument("--output-dir", default="output", help="Output folder path")
    args = parser.parse_args()

    # ===== EDIT THESE SETTINGS IF NEEDED =====
    model_path = Path("models") / "yolov8n-face.pt"
    conf = 0.25
    iou = 0.45
    imgsz = 640
    device = None  # Example: "cuda:0" or "cpu"

    gaze_weights = Path("models") / "models" / "Gaze360" / "L2CSNet_gaze360.pkl"
    gaze_arch = "ResNet50"
    gaze_device = "cpu"  # Example: "cuda:0" or "cpu"
    gaze_line_length = None  # None = bbox width * gaze_line_scale
    gaze_line_scale = 1.0
    gaze_line_thickness = 2

    landmarks_detection_confidence = 0.5
    landmarks_tracking_confidence = 0.5
    landmarks_model_complexity = 1
    landmarks_refine = True
    landmarks_point_radius = 1

    head_axis_length = 60.0
    head_axis_thickness = 2

    iris_vector_scale = 10.0
    iris_line_thickness = 2

    fusion_line_scale = 1.0
    fusion_line_thickness = 2
    fusion_process_noise = 1.0
    fusion_l2cs_noise = 10.0
    fusion_iris_noise = 36.0
    fusion_iris_scale = 1.0

    scene_landmark_scale = 1.0
    scene_z_scale = 1.0
    scene_depth_offset = 800.0
    scene_point_radius = 1
    scene_gaze_length = 200.0
    scene_gaze_angle_scale = 180.0
    scene_camera_yaw_deg = 0.0
    scene_camera_pitch_deg = 0.0
    scene_camera_roll_deg = 0.0
    scene_axis_length = 120.0
    scene_axis_thickness = 2
    scene_head_dist_process_noise = 25.0
    scene_head_dist_meas_noise = 100.0
    # ========================================

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.flag == "0":
        run_live(
            output_dir=output_dir,
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            gaze_weights=gaze_weights,
            gaze_arch=gaze_arch,
            gaze_device=gaze_device,
            gaze_line_length=gaze_line_length,
            gaze_line_scale=gaze_line_scale,
            gaze_line_thickness=gaze_line_thickness,
            landmarks_detection_confidence=landmarks_detection_confidence,
            landmarks_tracking_confidence=landmarks_tracking_confidence,
            landmarks_model_complexity=landmarks_model_complexity,
            landmarks_refine=landmarks_refine,
            landmarks_point_radius=landmarks_point_radius,
            head_axis_length=head_axis_length,
            head_axis_thickness=head_axis_thickness,
            iris_vector_scale=iris_vector_scale,
            iris_line_thickness=iris_line_thickness,
            fusion_line_scale=fusion_line_scale,
            fusion_line_thickness=fusion_line_thickness,
            fusion_process_noise=fusion_process_noise,
            fusion_l2cs_noise=fusion_l2cs_noise,
            fusion_iris_noise=fusion_iris_noise,
            fusion_iris_scale=fusion_iris_scale,
            scene_landmark_scale=scene_landmark_scale,
            scene_z_scale=scene_z_scale,
            scene_depth_offset=scene_depth_offset,
            scene_point_radius=scene_point_radius,
            scene_gaze_length=scene_gaze_length,
            scene_gaze_angle_scale=scene_gaze_angle_scale,
            scene_camera_yaw_deg=scene_camera_yaw_deg,
            scene_camera_pitch_deg=scene_camera_pitch_deg,
            scene_camera_roll_deg=scene_camera_roll_deg,
            scene_axis_length=scene_axis_length,
            scene_axis_thickness=scene_axis_thickness,
            scene_head_dist_process_noise=scene_head_dist_process_noise,
            scene_head_dist_meas_noise=scene_head_dist_meas_noise,
        )
        return

    input_video = find_input_video(input_dir, args.flag)
    calibrated_video = output_dir / f"{args.flag}_calibrated.mp4"
    output_video = output_dir / f"{args.flag}_face.mp4"
    gaze_video = output_dir / f"{args.flag}_gaze.mp4"
    landmarks_video = output_dir / f"{args.flag}_landmarks.mp4"
    head_pose_video = output_dir / f"{args.flag}_headpose.mp4"
    iris_gaze_video = output_dir / f"{args.flag}_iris_gaze.mp4"
    fusion_video = output_dir / f"{args.flag}_fusion.mp4"
    scene_video = output_dir / f"{args.flag}_scene3d.mp4"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    camera_matrix = compute_intrinsics(frame_width, frame_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "camera_K.npy", camera_matrix)
    np.savetxt(output_dir / "camera_K.txt", camera_matrix, fmt="%.6f")

    pipeline_video = calibrate_video(input_video, calibrated_video, camera_matrix)

    # Face detection: compute a stable crop box for downstream steps.
    detector = FaceDetector(
        FaceDetectorConfig(
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
    )

    result = detector.detect(pipeline_video)
    write_cropped_video(pipeline_video, output_video, result)

    # L2CS gaze estimation on the full frame using the face crop.
    gaze_estimator = GazeEstimator(
        GazeEstimatorConfig(
            weights_path=gaze_weights,
            arch=gaze_arch,
            device=gaze_device,
            line_length=gaze_line_length,
            line_scale=gaze_line_scale,
            line_thickness=gaze_line_thickness,
        )
    )

    gaze_results = gaze_estimator.run(pipeline_video, result, gaze_video)

    # MediaPipe landmarks for eyes/iris and 3D face points.
    landmarks_estimator = FaceLandmarksEstimator(
        FaceLandmarksConfig(
            min_detection_confidence=landmarks_detection_confidence,
            min_tracking_confidence=landmarks_tracking_confidence,
            model_complexity=landmarks_model_complexity,
            refine_landmarks=landmarks_refine,
            point_radius=landmarks_point_radius,
        )
    )

    landmarks_results = landmarks_estimator.run(pipeline_video, result, landmarks_video)

    # Head pose via solvePnP using sparse landmarks.
    head_pose_estimator = HeadPoseEstimator(
        HeadPoseConfig(
            axis_length=head_axis_length,
            line_thickness=head_axis_thickness,
        )
    )
    head_pose_results = head_pose_estimator.run(
        pipeline_video,
        landmarks_results,
        camera_matrix,
        head_pose_video,
    )

    # Iris-based gaze from binocular landmarks after de-rotation.
    iris_gaze_estimator = IrisGazeEstimator(
        IrisGazeConfig(
            vector_scale=iris_vector_scale,
            line_thickness=iris_line_thickness,
        )
    )
    iris_gaze_results = iris_gaze_estimator.run(
        pipeline_video,
        landmarks_results,
        head_pose_results,
        iris_gaze_video,
    )

    # Kalman fusion of L2CS and iris gaze vectors.
    fusion_estimator = GazeFusionEstimator(
        GazeFusionConfig(
            line_scale=fusion_line_scale,
            line_thickness=fusion_line_thickness,
            process_noise=fusion_process_noise,
            l2cs_meas_noise=fusion_l2cs_noise,
            iris_meas_noise=fusion_iris_noise,
            iris_scale=fusion_iris_scale,
        )
    )
    fusion_results = fusion_estimator.run(
        pipeline_video,
        result,
        gaze_results,
        iris_gaze_results,
        fusion_video,
    )

    # 3D scene rendering from landmarks, head pose, and fused gaze.
    scene_estimator = Scene3DEstimator(
        Scene3DConfig(
            landmark_scale=scene_landmark_scale,
            z_scale=scene_z_scale,
            depth_offset=scene_depth_offset,
            point_radius=scene_point_radius,
            gaze_length=scene_gaze_length,
            gaze_angle_scale=scene_gaze_angle_scale,
            camera_yaw_deg=scene_camera_yaw_deg,
            camera_pitch_deg=scene_camera_pitch_deg,
            camera_roll_deg=scene_camera_roll_deg,
            axis_length=scene_axis_length,
            axis_thickness=scene_axis_thickness,
            head_dist_process_noise=scene_head_dist_process_noise,
            head_dist_meas_noise=scene_head_dist_meas_noise,
        )
    )
    scene_results = scene_estimator.run(
        pipeline_video,
        landmarks_results,
        head_pose_results,
        fusion_results,
        camera_matrix,
        scene_video,
    )

    print(f"Input: {input_video}")
    print(f"Calibrated: {calibrated_video}")
    print(f"Output: {output_video}")
    print(f"Gaze output: {gaze_video}")
    print(f"Landmarks output: {landmarks_video}")
    print(f"Head pose output: {head_pose_video}")
    print(f"Iris gaze output: {iris_gaze_video}")
    print(f"Fusion output: {fusion_video}")
    print(f"3D scene output: {scene_video}")
    print(f"K matrix saved: {output_dir / 'camera_K.npy'}")
    print(f"Cropped dimensions: {result.crop_width} x {result.crop_height}")
    print(f"Gaze frames: {len(gaze_results)}")
    print(f"Landmark frames: {len(landmarks_results)}")
    print(f"Head pose frames: {len(head_pose_results)}")
    print(f"Iris gaze frames: {len(iris_gaze_results)}")
    print(f"Fusion frames: {len(fusion_results)}")
    print(f"Scene frames: {len(scene_results)}")


def run_live(
    output_dir: Path,
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    gaze_weights: Path,
    gaze_arch: str,
    gaze_device: str,
    gaze_line_length: int | None,
    gaze_line_scale: float,
    gaze_line_thickness: int,
    landmarks_detection_confidence: float,
    landmarks_tracking_confidence: float,
    landmarks_model_complexity: int,
    landmarks_refine: bool,
    landmarks_point_radius: int,
    head_axis_length: float,
    head_axis_thickness: int,
    iris_vector_scale: float,
    iris_line_thickness: int,
    fusion_line_scale: float,
    fusion_line_thickness: int,
    fusion_process_noise: float,
    fusion_l2cs_noise: float,
    fusion_iris_noise: float,
    fusion_iris_scale: float,
    scene_landmark_scale: float,
    scene_z_scale: float,
    scene_depth_offset: float,
    scene_point_radius: int,
    scene_gaze_length: float,
    scene_gaze_angle_scale: float,
    scene_camera_yaw_deg: float,
    scene_camera_pitch_deg: float,
    scene_camera_roll_deg: float,
    scene_axis_length: float,
    scene_axis_thickness: int,
    scene_head_dist_process_noise: float,
    scene_head_dist_meas_noise: float,
) -> None:
    """Run the full pipeline on a live camera stream and show windows."""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read from camera.")

    frame_height, frame_width = frame.shape[:2]
    camera_matrix = compute_intrinsics(frame_width, frame_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "camera_K.npy", camera_matrix)
    np.savetxt(output_dir / "camera_K.txt", camera_matrix, fmt="%.6f")

    detector = FaceDetector(
        FaceDetectorConfig(
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
    )

    gaze_estimator = GazeEstimator(
        GazeEstimatorConfig(
            weights_path=gaze_weights,
            arch=gaze_arch,
            device=gaze_device,
            line_length=gaze_line_length,
            line_scale=gaze_line_scale,
            line_thickness=gaze_line_thickness,
        )
    )

    landmarks_estimator = FaceLandmarksEstimator(
        FaceLandmarksConfig(
            min_detection_confidence=landmarks_detection_confidence,
            min_tracking_confidence=landmarks_tracking_confidence,
            model_complexity=landmarks_model_complexity,
            refine_landmarks=landmarks_refine,
            point_radius=landmarks_point_radius,
        )
    )

    head_pose_estimator = HeadPoseEstimator(
        HeadPoseConfig(
            axis_length=head_axis_length,
            line_thickness=head_axis_thickness,
        )
    )

    iris_gaze_estimator = IrisGazeEstimator(
        IrisGazeConfig(
            vector_scale=iris_vector_scale,
            line_thickness=iris_line_thickness,
        )
    )

    fusion_estimator = GazeFusionEstimator(
        GazeFusionConfig(
            line_scale=fusion_line_scale,
            line_thickness=fusion_line_thickness,
            process_noise=fusion_process_noise,
            l2cs_meas_noise=fusion_l2cs_noise,
            iris_meas_noise=fusion_iris_noise,
            iris_scale=fusion_iris_scale,
        )
    )

    scene_estimator = Scene3DEstimator(
        Scene3DConfig(
            landmark_scale=scene_landmark_scale,
            z_scale=scene_z_scale,
            depth_offset=scene_depth_offset,
            point_radius=scene_point_radius,
            gaze_length=scene_gaze_length,
            gaze_angle_scale=scene_gaze_angle_scale,
            camera_yaw_deg=scene_camera_yaw_deg,
            camera_pitch_deg=scene_camera_pitch_deg,
            camera_roll_deg=scene_camera_roll_deg,
            axis_length=scene_axis_length,
            axis_thickness=scene_axis_thickness,
            head_dist_process_noise=scene_head_dist_process_noise,
            head_dist_meas_noise=scene_head_dist_meas_noise,
        )
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    dt = 1.0 / fps
    head_dist = None
    head_cov = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_height, frame_width = frame.shape[:2]

        # Face detection (largest face)
        bbox = None
        results = detector.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy
            if len(boxes) > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(areas.argmax().item())
                x1, y1, x2, y2 = boxes[best_idx].tolist()
                x1 = max(0, min(int(x1), frame_width - 1))
                y1 = max(0, min(int(y1), frame_height - 1))
                x2 = max(1, min(int(x2), frame_width))
                y2 = max(1, min(int(y2), frame_height))
                bbox = (x1, y1, x2, y2)

        if bbox is None:
            cv2.imshow("Live Gaze Fusion", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # L2CS gaze
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        l2cs_pitch = l2cs_yaw = None
        if face.size != 0:
            l2cs_pitch, l2cs_yaw = gaze_estimator._predict_gaze(face)

        # Face landmarks on cropped face, mapped to full frame
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        lm_output = landmarks_estimator.face_mesh.process(face_rgb)
        landmarks = []
        if lm_output.multi_face_landmarks:
            for lm in lm_output.multi_face_landmarks[0].landmark:
                px = x1 + lm.x * (x2 - x1)
                py = y1 + lm.y * (y2 - y1)
                pz = lm.z * (x2 - x1)
                landmarks.append((px, py, pz))

        # Head pose (solvePnP)
        roll = 0.0
        rvec_tuple = None
        if landmarks:
            image_points = head_pose_estimator._get_image_points(landmarks)
            if image_points is not None and len(image_points) == 6:
                model_points = head_pose_estimator._get_model_points()
                dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    rvec_tuple = tuple(float(v) for v in rvec.flatten())
                    tx, ty, tz = (float(tvec[0]), float(tvec[1]), float(tvec[2]))
                    dist = math.sqrt(tx * tx + ty * ty + tz * tz)
                    if head_dist is None:
                        head_dist = dist
                        head_cov = 1.0
                    else:
                        head_cov = head_cov + scene_head_dist_process_noise
                        k = head_cov / (head_cov + scene_head_dist_meas_noise)
                        head_dist = head_dist + k * (dist - head_dist)
                        head_cov = (1.0 - k) * head_cov
                    print(f"[Live Scene3D] head distance = {head_dist:.2f} mm")
                    _, _, roll = head_pose_estimator._rvec_to_euler(rvec)

        # Iris gaze (binocular) after de-rotation
        left_vec = right_vec = combined_vec = None
        vergence = None
        if landmarks:
            pts = np.array([[x, y] for x, y, _ in landmarks], dtype=np.float32)
            center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            rot_deg = -math.degrees(roll)
            rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
            rot_inv = cv2.invertAffineTransform(rot_mat)
            rotated = cv2.warpAffine(frame, rot_mat, (frame_width, frame_height))
            rotated_pts = iris_gaze_estimator._transform_points(pts, rot_mat)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

            left_eye_pts = np.array(
                [rotated_pts[i] for i in iris_gaze_estimator.left_eye_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            right_eye_pts = np.array(
                [rotated_pts[i] for i in iris_gaze_estimator.right_eye_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            left_iris_pts = np.array(
                [rotated_pts[i] for i in iris_gaze_estimator.left_iris_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )
            right_iris_pts = np.array(
                [rotated_pts[i] for i in iris_gaze_estimator.right_iris_indices if i < len(rotated_pts)],
                dtype=np.float32,
            )

            left_bbox = iris_gaze_estimator._bbox_from_points(left_eye_pts, frame_width, frame_height)
            right_bbox = iris_gaze_estimator._bbox_from_points(right_eye_pts, frame_width, frame_height)

            left_center = right_center = None
            if left_eye_pts.size != 0:
                left_center = (float(left_eye_pts[:, 0].mean()), float(left_eye_pts[:, 1].mean()))
            if right_eye_pts.size != 0:
                right_center = (float(right_eye_pts[:, 0].mean()), float(right_eye_pts[:, 1].mean()))

            left_iris = right_iris = None
            if left_iris_pts.size != 0:
                left_iris = (float(left_iris_pts[:, 0].mean()), float(left_iris_pts[:, 1].mean()))
            if right_iris_pts.size != 0:
                right_iris = (float(right_iris_pts[:, 0].mean()), float(right_iris_pts[:, 1].mean()))

            if left_iris is None and left_center is not None:
                left_iris = iris_gaze_estimator._detect_iris_center(gray, left_bbox)
            if right_iris is None and right_center is not None:
                right_iris = iris_gaze_estimator._detect_iris_center(gray, right_bbox)

            if left_center is not None and left_iris is not None:
                eye_orig = iris_gaze_estimator._transform_points(np.array([left_center], dtype=np.float32), rot_inv)[0]
                iris_orig = iris_gaze_estimator._transform_points(np.array([left_iris], dtype=np.float32), rot_inv)[0]
                left_vec = iris_gaze_estimator._to_vector(tuple(eye_orig), tuple(iris_orig))
            if right_center is not None and right_iris is not None:
                eye_orig = iris_gaze_estimator._transform_points(np.array([right_center], dtype=np.float32), rot_inv)[0]
                iris_orig = iris_gaze_estimator._transform_points(np.array([right_iris], dtype=np.float32), rot_inv)[0]
                right_vec = iris_gaze_estimator._to_vector(tuple(eye_orig), tuple(iris_orig))

            combined_vec, vergence = iris_gaze_estimator._combine_vectors(left_vec, right_vec)

        iris_frame = IrisGazeFrameResult(
            frame_index=0,
            left=left_vec,
            right=right_vec,
            combined=combined_vec,
            vergence=vergence,
        )

        # Kalman fusion
        fusion_estimator._predict(dt)

        if l2cs_pitch is not None and l2cs_yaw is not None:
            l2cs_dx, l2cs_dy, _ = fusion_estimator._vector_from_l2cs(bbox, l2cs_pitch, l2cs_yaw)
            fusion_estimator._update(l2cs_dx, l2cs_dy, fusion_l2cs_noise)

        if combined_vec is not None:
            unit_x, unit_y, amp = fusion_estimator._normalize(combined_vec.dx, combined_vec.dy)
            if amp > 0:
                iris_dx = unit_x * amp * fusion_iris_scale
                iris_dy = unit_y * amp * fusion_iris_scale
                fusion_estimator._update(iris_dx, iris_dy, fusion_iris_noise)

        if fusion_estimator.state is not None:
            dx = float(fusion_estimator.state[0, 0])
            dy = float(fusion_estimator.state[1, 0])
            anchor = fusion_estimator._get_anchor(bbox, iris_frame)
            start = (int(round(anchor[0])), int(round(anchor[1])))
            end = (int(round(anchor[0] + dx)), int(round(anchor[1] + dy)))
            cv2.arrowedLine(frame, start, end, (0, 0, 255), fusion_line_thickness, cv2.LINE_AA)

        # 3D scene window
        scene_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if landmarks:
            points = np.array([[x, y, z] for x, y, z in landmarks], dtype=np.float32)
            center = points.mean(axis=0)
            points = (points - center) * scene_landmark_scale
            points[:, 2] *= scene_z_scale

            rot_cam = scene_estimator._rotation_from_camera()
            points = (rot_cam @ points.T).T

            projected = scene_estimator._project_points(points, frame_width, frame_height, camera_matrix)
            for px, py in projected:
                cv2.circle(
                    scene_canvas,
                    (int(round(px)), int(round(py))),
                    scene_point_radius,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA,
                )

            if fusion_estimator.state is not None:
                dx = float(fusion_estimator.state[0, 0])
                dy = float(fusion_estimator.state[1, 0])
                direction = math.atan2(dy, dx)
                amp = math.hypot(dx, dy)
                amp_deg = max(0.0, min(90.0, amp))
                gaze_yaw = direction
                gaze_pitch = math.radians(amp_deg)

                dir_vec = np.array(
                    [
                        math.sin(gaze_pitch) * math.cos(gaze_yaw),
                        math.sin(gaze_pitch) * math.sin(gaze_yaw),
                        -math.cos(gaze_pitch),
                    ],
                    dtype=np.float32,
                )
                gaze_len = head_dist if head_dist is not None else scene_gaze_length
                dir_vec *= gaze_len
                dir_vec = rot_cam @ dir_vec

                start_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                end_3d = dir_vec
                proj = scene_estimator._project_points(
                    np.stack([start_3d, end_3d], axis=0),
                    frame_width,
                    frame_height,
                    camera_matrix,
                )
                start = tuple(np.round(proj[0]).astype(int))
                end = tuple(np.round(proj[1]).astype(int))
                cv2.arrowedLine(scene_canvas, start, end, (0, 0, 255), fusion_line_thickness, cv2.LINE_AA)

            # Draw head-pose axis in live scene
            if rvec_tuple is not None:
                rot_head = scene_estimator._rotation_from_head_pose(rvec_tuple)
                if rot_head is not None:
                    axis = np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [scene_axis_length, 0.0, 0.0],
                            [0.0, scene_axis_length, 0.0],
                            [0.0, 0.0, scene_axis_length],
                        ],
                        dtype=np.float32,
                    )
                    axis = (rot_cam @ (rot_head @ axis.T)).T
                    proj = scene_estimator._project_points(axis, frame_width, frame_height, camera_matrix)
                    origin = tuple(np.round(proj[0]).astype(int))
                    x_end = tuple(np.round(proj[1]).astype(int))
                    y_end = tuple(np.round(proj[2]).astype(int))
                    z_end = tuple(np.round(proj[3]).astype(int))
                    cv2.line(scene_canvas, origin, x_end, (0, 0, 255), scene_axis_thickness)
                    cv2.line(scene_canvas, origin, y_end, (0, 255, 0), scene_axis_thickness)
                    cv2.line(scene_canvas, origin, z_end, (255, 0, 0), scene_axis_thickness)

        cv2.imshow("Live Gaze Fusion", frame)
        cv2.imshow("Live 3D Scene", scene_canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
