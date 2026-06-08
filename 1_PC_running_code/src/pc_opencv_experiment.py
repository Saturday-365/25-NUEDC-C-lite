from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── 默认相机参数（无标定时使用近似 pinhole 模型，需用户标定后替换） ──────────
# iPhone 主摄（实测 26mm 等效焦距）在 4032×3024 下约为 fx=fy≈3000
# 使用时请通过 --camera-matrix 或 --calib-file 提供标定结果
DEFAULT_CAMERA_MATRIX = np.array([
    [3000.0, 0, 2016.0],
    [0, 3000.0, 1512.0],
    [0, 0, 1]
], dtype=np.float64)

DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)


@dataclass
class ExperimentConfig:
    board_width_mm: float = 168.1
    board_height_mm: float = 255.1
    warp_width_px: int = 800
    min_board_area_ratio: float = 0.04
    inner_margin_ratio: float = 0.08
    min_inner_area_px: float = 400.0
    # ── 相机参数 ──
    camera_matrix: np.ndarray = field(default_factory=lambda: DEFAULT_CAMERA_MATRIX.copy())
    dist_coeffs: np.ndarray = field(default_factory=lambda: DEFAULT_DIST_COEFFS.copy())
    # ── 距离修正（分段线性），格式 [(真实距离mm, 原始Zmm), ...] ──
    calib_points: list[tuple[float, float]] = field(default_factory=list)
    # ── PnP 方法 ──
    pnp_method: int = cv2.SOLVEPNP_IPPE

    @property
    def warp_height_px(self) -> int:
        return round(self.warp_width_px * self.board_height_mm / self.board_width_mm)

    @property
    def board_diagonal_mm(self) -> float:
        return np.sqrt(self.board_width_mm**2 + self.board_height_mm**2)

    @property
    def object_points(self) -> np.ndarray:
        """A4 外框四角点（左上→右上→右下→左下），原点在中心"""
        w2 = self.board_width_mm / 2.0
        h2 = self.board_height_mm / 2.0
        return np.array([
            [-w2, -h2, 0],
            [ w2, -h2, 0],
            [ w2,  h2, 0],
            [-w2,  h2, 0],
        ], dtype=np.float64)


@dataclass
class ShapeMeasurement:
    source: str
    shape: str
    pixel_size: float
    measured_size_mm: float
    area_mm2: float
    contour_area_px: float
    center_x: int
    center_y: int


@dataclass
class PnPResult:
    """PnP 位姿解算结果"""
    source: str
    tvec_x_mm: float
    tvec_y_mm: float
    tvec_z_mm: float      # 即原始距离 D_raw
    distance_mm: float     # 修正后的距离 D_cal
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rvec: np.ndarray       # 保留原始旋转向量
    tvec: np.ndarray       # 保留原始平移向量


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


# ═══════════════════════════════════════════════════════════════
# PnP 位姿解算
# ═══════════════════════════════════════════════════════════════

def solve_pnp_pose(
    board_quad: np.ndarray, config: ExperimentConfig
) -> PnPResult | None:
    """利用 A4 外框四角点进行 PnP 位姿解算。

    返回 PnPResult，包含目标板中心在相机坐标系下的 (X, Y, Z) 坐标、
    欧拉角（偏航/俯仰/滚转）以及修正后的距离。
    """
    image_points = board_quad.reshape(4, 2).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        config.object_points,
        image_points,
        config.camera_matrix,
        config.dist_coeffs,
        flags=config.pnp_method,
    )
    if not success:
        return None

    # ── 平移向量 → 坐标 ──
    tx = float(tvec[0, 0])
    ty = float(tvec[1, 0])
    tz = float(tvec[2, 0])

    # ── 旋转向量 → 欧拉角 ──
    R, _ = cv2.Rodrigues(rvec)
    pitch = float(np.arctan2(R[2, 1], R[2, 2]) * 180.0 / np.pi)
    yaw   = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180.0 / np.pi)
    roll  = float(np.arctan2(R[1, 0], R[0, 0]) * 180.0 / np.pi)

    # ── 距离修正 ──
    distance = correct_distance(tz, config.calib_points)

    return PnPResult(
        source="",
        tvec_x_mm=tx,
        tvec_y_mm=ty,
        tvec_z_mm=tz,
        distance_mm=distance,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        rvec=rvec,
        tvec=tvec,
    )


def correct_distance(raw_z_mm: float, calib_points: list[tuple[float, float]]) -> float:
    """分段线性插值距离修正。

    calib_points: [(真实距离mm, 原始Zmm), ...]
    至少需要 2 个标定点。
    若标定点不足或超出范围，则返回原始 Z。
    """
    if len(calib_points) < 2:
        return raw_z_mm

    # 按原始 Z 排序
    sorted_pts = sorted(calib_points, key=lambda p: p[1])
    real_vals, raw_vals = zip(*sorted_pts)

    # 边界外推
    if raw_z_mm <= raw_vals[0]:
        offset = real_vals[0] - raw_vals[0]
        return raw_z_mm + offset
    if raw_z_mm >= raw_vals[-1]:
        offset = real_vals[-1] - raw_vals[-1]
        return raw_z_mm + offset

    # 内插
    for i in range(len(raw_vals) - 1):
        if raw_vals[i] <= raw_z_mm <= raw_vals[i + 1]:
            alpha = (raw_z_mm - raw_vals[i]) / (raw_vals[i + 1] - raw_vals[i])
            return real_vals[i] + alpha * (real_vals[i + 1] - real_vals[i])

    return raw_z_mm


def draw_pose_info(
    image: np.ndarray,
    pose: PnPResult,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float = 80.0,
) -> None:
    """在图像上绘制三维坐标轴和位姿文本信息。"""
    # ── 3D 坐标轴 ──
    if pose.rvec is not None and pose.tvec is not None:
        axes_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ]).reshape(-1, 3)
        axes_2d, _ = cv2.projectPoints(axes_3d, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2).astype(np.int32)
        origin = tuple(axes_2d[0])
        cv2.line(image, origin, tuple(axes_2d[1]), (0, 0, 255), 3)  # X: 红
        cv2.line(image, origin, tuple(axes_2d[2]), (0, 255, 0), 3)  # Y: 绿
        cv2.line(image, origin, tuple(axes_2d[3]), (255, 0, 0), 3)  # Z: 蓝

    # ── 文本信息 ──
    lines = [
        f"Position: X={pose.tvec_x_mm:+.0f}  Y={pose.tvec_y_mm:+.0f}  Z={pose.tvec_z_mm:.0f} mm",
        f"Distance: D_raw={pose.tvec_z_mm:.0f}  D_cal={pose.distance_mm:.0f} mm",
        f"Euler: Yaw={pose.yaw_deg:+.1f}  Pitch={pose.pitch_deg:+.1f}  Roll={pose.roll_deg:+.1f} deg",
    ]
    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (10, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )


def preprocess_for_black_contours(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


def find_board_quad(binary: np.ndarray, config: ExperimentConfig) -> np.ndarray | None:
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    image_area = binary.shape[0] * binary.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * config.min_board_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return order_quad_points(candidates[0][1])


def warp_board(
    image: np.ndarray, board_quad: np.ndarray, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    width = config.warp_width_px
    height = config.warp_height_px
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(board_quad, target)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def classify_inner_shape(contour: np.ndarray) -> tuple[str, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return "unknown", 0.0

    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)

    if len(approx) == 3:
        return "triangle", perimeter / 3.0

    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    if rect_w <= 0 or rect_h <= 0:
        return "unknown", 0.0

    aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
    fill_ratio = area / (rect_w * rect_h)

    if len(approx) == 4 and aspect < 1.25 and fill_ratio > 0.72:
        return "square", (rect_w + rect_h) / 2.0

    if circularity > 0.72:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return "circle", radius * 2.0

    return "unknown", max(rect_w, rect_h)


def shape_area_mm2(shape: str, size_mm: float) -> float:
    if shape == "triangle":
        return size_mm * size_mm * np.sqrt(3.0) / 4.0
    if shape == "square":
        return size_mm * size_mm
    if shape == "circle":
        radius = size_mm / 2.0
        return np.pi * radius * radius
    return 0.0


def detect_inner_shapes(
    warped: np.ndarray, source_name: str, config: ExperimentConfig
) -> tuple[list[ShapeMeasurement], np.ndarray]:
    annotated = warped.copy()
    binary = preprocess_for_black_contours(warped)

    margin_x = round(warped.shape[1] * config.inner_margin_ratio)
    margin_y = round(warped.shape[0] * config.inner_margin_ratio)
    inner = binary[margin_y : warped.shape[0] - margin_y, margin_x : warped.shape[1] - margin_x]

    contours = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    px_to_mm = (config.board_width_mm / warped.shape[1] + config.board_height_mm / warped.shape[0]) / 2.0

    measurements: list[ShapeMeasurement] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_inner_area_px:
            continue

        contour = contour + np.array([[[margin_x, margin_y]]], dtype=contour.dtype)
        shape, pixel_size = classify_inner_shape(contour)
        if shape == "unknown" or pixel_size <= 0:
            continue

        measured_size_mm = pixel_size * px_to_mm
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        measurements.append(
            ShapeMeasurement(
                source=source_name,
                shape=shape,
                pixel_size=pixel_size,
                measured_size_mm=measured_size_mm,
                area_mm2=shape_area_mm2(shape, measured_size_mm),
                contour_area_px=area,
                center_x=center_x,
                center_y=center_y,
            )
        )

        cv2.drawContours(annotated, [contour], -1, (0, 180, 0), 2)
        cv2.circle(annotated, (center_x, center_y), 4, (0, 0, 255), -1)
        label = f"{shape} {measured_size_mm:.1f}mm"
        cv2.putText(
            annotated,
            label,
            (center_x + 8, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return measurements, annotated


def process_image(
    image_path: Path, output_dir: Path, config: ExperimentConfig
) -> tuple[list[ShapeMeasurement], PnPResult | None]:
    """处理单张图像：外框检测 → PnP 位姿解算 → 透视校正 → 内部图形测量。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # ── 1. 外框检测 ──
    binary = preprocess_for_black_contours(image)
    board_quad = find_board_quad(binary, config)
    if board_quad is None:
        raise ValueError(f"No rectangular target board found: {image_path}")

    # ── 2. PnP 位姿解算 ──
    pose = solve_pnp_pose(board_quad, config)
    if pose is not None:
        pose.source = image_path.name

    # ── 3. 标注原始图像 ──
    raw_annotated = image.copy()
    cv2.polylines(raw_annotated, [board_quad.astype(np.int32)], True, (0, 0, 255), 3)
    for idx, point in enumerate(board_quad.astype(np.int32), start=1):
        cv2.putText(
            raw_annotated,
            str(idx),
            tuple(point),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    # 绘制位姿信息
    if pose is not None:
        draw_pose_info(raw_annotated, pose, config.camera_matrix, config.dist_coeffs)

    # ── 4. 透视校正 & 内部图形检测 ──
    warped, _ = warp_board(image, board_quad, config)
    measurements, warped_annotated = detect_inner_shapes(warped, image_path.name, config)
    # 在 warped 图像上也叠加位姿信息
    if pose is not None:
        draw_pose_info(warped_annotated, pose, config.camera_matrix, config.dist_coeffs)

    # ── 5. 保存结果 ──
    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_binary.png"), binary)
    cv2.imwrite(str(output_dir / f"{stem}_raw.png"), raw_annotated)
    cv2.imwrite(str(output_dir / f"{stem}_warped.png"), warped_annotated)
    return measurements, pose


def iter_input_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_measurements_csv(output_path: Path, measurements: list[ShapeMeasurement]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "shape",
                "pixel_size",
                "measured_size_mm",
                "area_mm2",
                "contour_area_px",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        for item in measurements:
            writer.writerow(
                {
                    "source": item.source,
                    "shape": item.shape,
                    "pixel_size": f"{item.pixel_size:.3f}",
                    "measured_size_mm": f"{item.measured_size_mm:.3f}",
                    "area_mm2": f"{item.area_mm2:.3f}",
                    "contour_area_px": f"{item.contour_area_px:.3f}",
                    "center_x": item.center_x,
                    "center_y": item.center_y,
                }
            )


def write_pose_csv(output_path: Path, poses: list[PnPResult]) -> None:
    """将位姿解算结果写入 CSV。"""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source", "tvec_x_mm", "tvec_y_mm", "tvec_z_mm",
            "distance_mm", "yaw_deg", "pitch_deg", "roll_deg",
        ])
        writer.writeheader()
        for p in poses:
            writer.writerow({
                "source": p.source,
                "tvec_x_mm": f"{p.tvec_x_mm:.1f}",
                "tvec_y_mm": f"{p.tvec_y_mm:.1f}",
                "tvec_z_mm": f"{p.tvec_z_mm:.1f}",
                "distance_mm": f"{p.distance_mm:.1f}",
                "yaw_deg": f"{p.yaw_deg:.2f}",
                "pitch_deg": f"{p.pitch_deg:.2f}",
                "roll_deg": f"{p.roll_deg:.2f}",
            })


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 OpenCV 相机标定输出的 npz / json 文件加载内参和畸变系数。

    支持的格式:
      - OpenCV 的 npz (含 camera_matrix, dist_coeffs 键)
      - 自定义 JSON (含 camera_matrix, dist_coeffs 键)
    """
    ext = calib_path.suffix.lower()
    if ext == ".npz":
        我data = np.load(str(calib_path))
        cmat = data["camera_matrix"]
        dcoeff = data["dist_coeffs"]
    elif ext == ".json":
        with open(calib_path) as f:
            data = json.load(f)
        cmat = np.array(data["camera_matrix"], dtype=np.float64)
        dcoeff = np.array(data["dist_coeffs"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported calibration file format: {ext}")
    return cmat, dcoeff


def capture_from_camera(camera_index: int, capture_path: Path) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera opened but did not return a frame")
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(capture_path), frame)
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PC OpenCV experiment — PnP pose estimation, perspective correction, "
                    "and inner shape measurement for the NUEDC 2025 C target board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 处理单张图片\n"
            "  python pc_opencv_experiment.py --input img.jpg --output out/\n\n"
            "  # 批量处理文件夹\n"
            "  python pc_opencv_experiment.py --input data/input/ --output data/output/\n\n"
            "  # 使用相机标定文件\n"
            "  python pc_opencv_experiment.py --input data/input/ --output data/output/ \\\n"
            "      --calib-file data/calib/iphone_calib.npz\n\n"
            "  # 指定相机内参 (fx fy cx cy)\n"
            "  python pc_opencv_experiment.py --input data/input/ --output data/output/ \\\n"
            "      --camera-matrix 1158.86 1183.95 357.72 200.27\n\n"
            "  # 从 USB 相机采集\n"
            "  python pc_opencv_experiment.py --camera 0 --capture data/input/shot.jpg\n"
        ),
    )
    parser.add_argument("--input", type=Path, help="Input image file or image directory.")
    parser.add_argument("--output", type=Path, default=Path("data/output"),
                        help="Output directory for results (default: data/output).")
    parser.add_argument("--camera", type=int, help="USB camera index, usually 0.")
    parser.add_argument("--capture", type=Path, help="Path to save one camera frame.")

    # ── 目标板尺寸 ──
    parser.add_argument("--board-width-mm", type=float, default=168.1,
                        help="A4 outer border width in mm (default: 168.1).")
    parser.add_argument("--board-height-mm", type=float, default=255.1,
                        help="A4 outer border height in mm (default: 255.1).")

    # ── 相机参数 ──
    parser.add_argument("--calib-file", type=Path,
                        help="Path to camera calibration .npz or .json file.")
    parser.add_argument("--camera-matrix", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                        help="Camera intrinsics: fx fy cx cy (overrides --calib-file).")

    # ── 距离修正标定 ──
    parser.add_argument("--calib-csv", type=Path,
                        help="CSV for distance calibration: columns 'real_mm,raw_z_mm'.")

    # ── iPhone 图片压缩辅助 ──
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="Resize input images to W×H before processing (e.g. for iPhone photos).")
    return parser.parse_args()


def load_calib_csv(path: Path) -> list[tuple[float, float]]:
    """加载距离标定 CSV: 每行 (真实距离mm, 原始Zmm)。"""
    points: list[tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0].startswith("real"):
                continue
            points.append((float(row[0]), float(row[1])))
    return points


def main() -> int:
    args = parse_args()

    # ── 相机参数 ──
    camera_matrix = DEFAULT_CAMERA_MATRIX.copy()
    dist_coeffs = DEFAULT_DIST_COEFFS.copy()

    if args.camera_matrix is not None:
        fx, fy, cx, cy = args.camera_matrix
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    elif args.calib_file is not None:
        camera_matrix, dist_coeffs = load_calibration(args.calib_file)
        print(f"Loaded calibration: fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}  "
              f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}")

    # ── 距离修正标定点 ──
    calib_points: list[tuple[float, float]] = []
    if args.calib_csv is not None:
        calib_points = load_calib_csv(args.calib_csv)
        print(f"Loaded {len(calib_points)} distance calibration points")

    # ── 相机模式 ──
    if args.camera is not None:
        if args.capture is None:
            raise SystemExit("--capture is required when --camera is used")
        capture_from_camera(args.camera, args.capture)
        print(f"Captured frame: {args.capture}")
        return 0

    if args.input is None:
        raise SystemExit("--input is required unless --camera is used")

    # ── 配置 ──
    config = ExperimentConfig(
        board_width_mm=args.board_width_mm,
        board_height_mm=args.board_height_mm,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        calib_points=calib_points,
    )
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 处理每张图片 ──
    all_measurements: list[ShapeMeasurement] = []
    all_poses: list[PnPResult] = []

    for image_path in iter_input_images(args.input):
        try:
            # 可选 resize（用于 iPhone 高分辨率照片）
            if args.resize is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                pil_img = pil_img.resize(tuple(args.resize), PILImage.LANCZOS)
                temp_path = output_dir / f"_resized_{image_path.name}"
                pil_img.save(temp_path)
                image_path = temp_path

            measurements, pose = process_image(image_path, output_dir, config)
            all_measurements.extend(measurements)

            stem = image_path.stem
            if pose is not None:
                all_poses.append(pose)
                print(f"Processed {stem}: "
                      f"D_raw={pose.tvec_z_mm:.0f}mm  D_cal={pose.distance_mm:.0f}mm  "
                      f"X={pose.tvec_x_mm:+.0f}  Y={pose.tvec_y_mm:+.0f}  "
                      f"| shapes={len(measurements)}")
            else:
                print(f"Processed {stem}: PnP failed, shapes={len(measurements)}")

        except ValueError as exc:
            print(f"Skipped {image_path.name}: {exc}")

    # ── 保存 CSV ──
    if all_measurements:
        mpath = output_dir / "measurements.csv"
        write_measurements_csv(mpath, all_measurements)
        print(f"Wrote {mpath} ({len(all_measurements)} shapes)")

    if all_poses:
        ppath = output_dir / "pose_results.csv"
        write_pose_csv(ppath, all_poses)
        print(f"Wrote {ppath} ({len(all_poses)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
