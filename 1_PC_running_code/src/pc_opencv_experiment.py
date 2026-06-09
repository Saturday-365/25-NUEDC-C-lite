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


def imread_unicode(path: Path) -> np.ndarray | None:
    """支持中文路径的图片读取（Windows 编码兼容）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """支持中文路径的图片写入。"""
    ext = path.suffix.lower()
    success, buf = cv2.imencode(ext, img, params or [])
    if not success:
        return False
    with open(str(path), "wb") as f:
        f.write(buf.tobytes())
    return True


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
    min_board_area_ratio: float = 0.04
    # ── 相机参数 ──
    camera_matrix: np.ndarray = field(default_factory=lambda: DEFAULT_CAMERA_MATRIX.copy())
    dist_coeffs: np.ndarray = field(default_factory=lambda: DEFAULT_DIST_COEFFS.copy())
    # ── 距离修正（分段线性），格式 [(真实距离mm, 原始Zmm), ...] ──
    calib_points: list[tuple[float, float]] = field(default_factory=list)
    # ── PnP 方法 ──
    pnp_method: int = cv2.SOLVEPNP_IPPE

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


def _quads_from_contours(
    contours, image_area: float, config: ExperimentConfig, eps_factor: float = 0.02
) -> list[tuple[float, np.ndarray]]:
    """从轮廓列表中筛选四边形候选。"""
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * config.min_board_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 边长不能太离谱：最短边 / 最长边 > 0.15
            edges = [
                np.linalg.norm(approx[(i + 1) % 4] - approx[i])
                for i in range(4)
            ]
            if min(edges) / max(edges) > 0.15:
                candidates.append((area, approx))
    return candidates


def _find_quad_in_roi(
    img: np.ndarray, config: ExperimentConfig,
    margin: float = 0.0
) -> np.ndarray | None:
    """在图像（或裁剪后的 ROI）中找 A4 外框四边形。margin>0 时先裁剪边缘。"""
    h, w = img.shape[:2]
    if margin > 0:
        x0, x1 = int(w * margin), int(w * (1 - margin))
        y0, y1 = int(h * margin), int(h * (1 - margin))
        roi = img[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        roi = img

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_area = blur.shape[0] * blur.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # OTSU 找黑框
    binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    morph = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    candidates = _quads_from_contours(cnts, roi_area, config)
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        if margin > 0:
            best[:, :, 0] += x0
            best[:, :, 1] += y0
        return order_quad_points(best)
    return None


def find_board_quad(image: np.ndarray, config: ExperimentConfig) -> np.ndarray | None:
    """多策略查找 A4 外框四边形。依次尝试不同方法，直到找到。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = blur.shape[0] * blur.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # ═══════════════════════════════════════════════════════════
    # 策略 1: 全图 OTSU（原方法，最快）
    # ═══════════════════════════════════════════════════════════
    q = _find_quad_in_roi(image, config, margin=0.0)
    if q is not None:
        return q

    # ═══════════════════════════════════════════════════════════
    # 策略 2: 裁剪四周 15-30%，排除杂乱背景
    # ═══════════════════════════════════════════════════════════
    for margin in [0.20, 0.15, 0.25, 0.30]:
        q = _find_quad_in_roi(image, config, margin=margin)
        if q is not None:
            return q

    # ═══════════════════════════════════════════════════════════
    # 策略 3: OTSU 白纸检测（深色背景上的白色 A4）
    # ═══════════════════════════════════════════════════════════
    binary_white = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    morph_white = cv2.morphologyEx(binary_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_white = cv2.findContours(morph_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    candidates_white = _quads_from_contours(contours_white, image_area, config)
    if candidates_white:
        candidates_white.sort(key=lambda item: item[0], reverse=True)
        return order_quad_points(candidates_white[0][1])
    # 白纸区域找 minAreaRect 作为后备
    if contours_white:
        largest = max(contours_white, key=cv2.contourArea)
        la = cv2.contourArea(largest)
        if la > image_area * config.min_board_area_ratio:
            rect = cv2.minAreaRect(largest)
            rw, rh = rect[1]
            if rw > 0 and rh > 0:
                asp = max(rw, rh) / min(rw, rh)
                if 1.2 < asp < 2.0:
                    box = cv2.boxPoints(rect)
                    return order_quad_points(box)

    # ═══════════════════════════════════════════════════════════
    # 策略 4: 自适应阈值
    # ═══════════════════════════════════════════════════════════
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )
    morph2 = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph2 = cv2.dilate(morph2, dilate_k, iterations=1)
    contours2 = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    candidates2 = _quads_from_contours(contours2, image_area, config)
    if candidates2:
        candidates2.sort(key=lambda item: item[0], reverse=True)
        return order_quad_points(candidates2[0][1])

    # ═══════════════════════════════════════════════════════════
    # 策略 5: Canny 边缘 + 膨胀
    # ═══════════════════════════════════════════════════════════
    edges = cv2.Canny(blur, 30, 100)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=4)
    contours3 = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    candidates3 = _quads_from_contours(contours3, image_area, config, eps_factor=0.025)
    if candidates3:
        candidates3.sort(key=lambda item: item[0], reverse=True)
        return order_quad_points(candidates3[0][1])

    # ═══════════════════════════════════════════════════════════
    # 策略 6: RETR_TREE 嵌套四边形
    # ═══════════════════════════════════════════════════════════
    binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    morph = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_inv = 255 - morph
    contours4, hierarchy = cv2.findContours(morph_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours4):
            area = cv2.contourArea(contour)
            if area < image_area * config.min_board_area_ratio:
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            child = hierarchy[i][2]
            if child >= 0:
                child_area = cv2.contourArea(contours4[child])
                fill_ratio = child_area / area
                if 0.55 < fill_ratio < 0.96:
                    return order_quad_points(approx)

    # ═══════════════════════════════════════════════════════════
    # 策略 7: HSV V 通道多阈值白纸检测
    # ═══════════════════════════════════════════════════════════
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    for thresh_val in range(180, 100, -10):
        _, v_bin = cv2.threshold(v, thresh_val, 255, cv2.THRESH_BINARY)
        v_morph = cv2.morphologyEx(v_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
        c_v = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c_v:
            largest_v = max(c_v, key=cv2.contourArea)
            la = cv2.contourArea(largest_v)
            if la > image_area * 0.15:
                p_v = cv2.arcLength(largest_v, True)
                if p_v > 0:
                    ap_v = cv2.approxPolyDP(largest_v, 0.02 * p_v, True)
                    if len(ap_v) == 4 and cv2.isContourConvex(ap_v):
                        return order_quad_points(ap_v)
                    rect_v = cv2.minAreaRect(largest_v)
                    box_v = cv2.boxPoints(rect_v)
                    rw, rh = rect_v[1]
                    if rw > 0 and rh > 0:
                        asp = max(rw, rh) / min(rw, rh)
                        if 1.2 < asp < 2.0:
                            return order_quad_points(box_v)

    # ═══════════════════════════════════════════════════════════
    # 策略 8: 自适应阈值多 block 大小
    # ═══════════════════════════════════════════════════════════
    s = hsv[:, :, 1]
    for src in [blur, 255 - s, v]:
        for block in [31, 51, 71]:
            try:
                ad = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, block, 10)
            except Exception:
                continue
            ad_morph = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, kernel, iterations=1)
            c_ad = cv2.findContours(ad_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cand_ad = _quads_from_contours(c_ad, image_area, config)
            if cand_ad:
                cand_ad.sort(key=lambda x: x[0], reverse=True)
                return order_quad_points(cand_ad[0][1])

    return None





def process_image(
    image_path: Path, output_dir: Path, config: ExperimentConfig
) -> PnPResult | None:
    """处理单张图像：外框检测 → PnP 位姿解算 → 输出结果。"""
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # ── 1. 外框检测 ──
    binary = preprocess_for_black_contours(image)
    board_quad = find_board_quad(image, config)
    if board_quad is None:
        raise ValueError(f"No rectangular target board found: {image_path}")

    # ── 2. PnP 位姿解算 ──
    pose = solve_pnp_pose(board_quad, config)
    if pose is not None:
        pose.source = image_path.name

    # ── 3. 标注并保存结果图 ──
    raw_annotated = image.copy()
    cv2.polylines(raw_annotated, [board_quad.astype(np.int32)], True, (0, 0, 255), 3)
    for idx, point in enumerate(board_quad.astype(np.int32), start=1):
        cv2.putText(
            raw_annotated, str(idx), tuple(point),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA,
        )
    if pose is not None:
        draw_pose_info(raw_annotated, pose, config.camera_matrix, config.dist_coeffs)

    stem = image_path.stem
    imwrite_unicode(output_dir / f"{stem}_binary.png", binary)
    imwrite_unicode(output_dir / f"{stem}_annotated.png", raw_annotated)
    return pose


def iter_input_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


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
        data = np.load(str(calib_path))
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
        imwrite_unicode(capture_path, frame)
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
    all_poses: list[PnPResult] = []

    for image_path in iter_input_images(args.input):
        try:
            # 可选 resize（用于 iPhone 高分辨率照片）
            if args.resize is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                pil_img = pil_img.resize(tuple(args.resize), PILImage.LANCZOS)
                temp_path = output_dir / f"_resized_{image_path.name}"
                pil_img.save(str(temp_path))
                image_path = temp_path

            pose = process_image(image_path, output_dir, config)

            stem = image_path.stem
            if pose is not None:
                all_poses.append(pose)
                print(f"Processed {stem}: "
                      f"D_raw={pose.tvec_z_mm:.0f}mm  D_cal={pose.distance_mm:.0f}mm  "
                      f"X={pose.tvec_x_mm:+.0f}  Y={pose.tvec_y_mm:+.0f}")
            else:
                print(f"Processed {stem}: PnP failed")

        except ValueError as exc:
            print(f"Skipped {image_path.name}: {exc}")

    # ── 保存 CSV ──
    if all_poses:
        ppath = output_dir / "pose_results.csv"
        write_pose_csv(ppath, all_poses)
        print(f"Wrote {ppath} ({len(all_poses)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
