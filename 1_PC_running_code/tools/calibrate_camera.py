"""
相机标定工具 — 使用棋盘格标定板对 iPhone（或任意相机）进行标定。

工作流程：
  1. 打印棋盘格（用 generate_chessboard.py 生成），贴在硬纸板上
  2. 用 iPhone 从不同角度/距离拍摄 15-25 张棋盘格照片
  3. 将照片传入此脚本进行标定
  4. 输出 .npz 标定文件，可直接用于 pc_opencv_experiment.py

用法：
  # 批量处理标定图片
  python tools/calibrate_camera.py --input data/calib/iphone_images/ --output data/calib/iphone_calib.npz

  # 指定棋盘格参数
  python tools/calibrate_camera.py --input data/calib/iphone_images/ --output data/calib/iphone_calib.npz --cols 9 --rows 7 --square-mm 25

  # 显示标定结果评估
  python tools/calibrate_camera.py --input data/calib/iphone_images/ --output data/calib/iphone_calib.npz --show

拍摄建议：
  - 棋盘格占画面 20-80% 面积
  - 覆盖画面各个区域（四角、中心）
  - 保持棋盘格平坦
  - 至少 15 张，推荐 20-25 张
  - 距离覆盖 30cm-200cm
  - 不同倾斜角度（0°-45°）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ── 支持的图片格式 ──
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_unicode(path: Path) -> np.ndarray | None:
    """支持中文路径的图片读取（Windows 编码兼容）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Camera calibration using chessboard pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input image file or directory containing chessboard images.")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/calib/calibration.npz"),
                        help="Output .npz file path (default: data/calib/calibration.npz).")
    parser.add_argument("--cols", type=int, default=9,
                        help="Number of inner corners per row (default: 9).")
    parser.add_argument("--rows", type=int, default=7,
                        help="Number of inner corners per column (default: 7).")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="Chessboard square side length in mm (default: 25).")
    parser.add_argument("--show", action="store_true",
                        help="Display detected corners on images for visual verification.")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Maximum images to use (default: 50).")
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)  # (width, height) 内角点数
    square_size = args.square_mm

    # ── 准备三维点：棋盘格平面上的角点坐标 ──
    objp = np.zeros((args.rows * args.cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * square_size

    object_points: list[np.ndarray] = []  # 世界坐标系中的点
    image_points: list[np.ndarray] = []   # 图像坐标系中的点
    image_sizes: list[tuple[int, int]] = []

    image_paths = list(iter_images(args.input))
    if not image_paths:
        raise SystemExit(f"No images found in {args.input}")

    print(f"Found {len(image_paths)} images, searching for chessboard corners...")
    print(f"  Pattern: {pattern_size[0]}×{pattern_size[1]} inner corners")
    print(f"  Square size: {square_size} mm")
    print()

    successful = 0
    for path in image_paths[: args.max_images]:
        img = imread_unicode(path)
        if img is None:
            print(f"  ⚠  Cannot read: {path.name}, skipping")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        image_sizes.append((w, h))

        # 使用亚像素精度查找角点
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # 亚像素精炼
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            object_points.append(objp)
            image_points.append(corners_refined)
            successful += 1

            if args.show:
                annotated = cv2.drawChessboardCorners(img.copy(), pattern_size, corners_refined, found)
                cv2.imshow(f"Chessboard: {path.name}", annotated)
                print(f"  [{successful:2d}] ✓ {path.name}  (press any key to continue)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"  [{successful:2d}] ✓ {path.name}")
        else:
            print(f"  [{successful+1:2d}] ✗ {path.name}  — no chessboard found")

    if successful < 4:
        raise SystemExit(
            f"\nOnly {successful} images with detected corners (need at least 4).\n"
            "Tips:\n"
            "  - Ensure the chessboard is fully visible\n"
            "  - Try different lighting conditions\n"
            "  - Check that pattern_size matches your board\n"
            f"  - Pattern size set to: {pattern_size[0]}×{pattern_size[1]}"
        )

    print(f"\n{successful}/{len(image_paths)} images with detected chessboard.")
    print("Running calibration...")

    # ── 执行标定 ──
    # 取中位数图像尺寸（所有图片应一致）
    median_w = sorted(s for s, _ in image_sizes)[len(image_sizes) // 2]
    median_h = sorted(h for _, h in image_sizes)[len(image_sizes) // 2]

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (median_w, median_h),
        cameraMatrix=None, distCoeffs=None,
        flags=(
            cv2.CALIB_RATIONAL_MODEL |   # 允许 k4, k5, k6
            cv2.CALIB_THIN_PRISM_MODEL | # 薄棱镜畸变
            cv2.CALIB_FIX_ASPECT_RATIO   # 固定 fx/fy 比例（保持传感器像素方形）
        )
    )

    # ── 计算重投影误差 ──
    total_error = 0.0
    for i in range(len(object_points)):
        img_points_proj, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error
    mean_error = total_error / len(object_points)

    # ── 打印结果 ──
    print(f"\nCalibration result:")
    print(f"  RMS re-projection error: {ret:.4f} pixels")
    print(f"  Mean corner error: {mean_error:.4f} pixels")
    print(f"  Camera matrix:\n{camera_matrix}")
    print(f"  Distortion coefficients:\n{dist_coeffs.reshape(1, -1)}")

    # ── 保存 ──
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.output),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=ret,
        mean_error=mean_error,
        image_count=successful,
        image_size=(median_w, median_h),
        pattern_size=pattern_size,
        square_mm=square_size,
    )
    print(f"\nSaved: {args.output.resolve()}")

    # ── 使用建议 ──
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    print(f"\n── 在实验中使用的命令 ──")
    print(f"python src/pc_opencv_experiment.py --input data/input --output data/output \\")
    print(f"    --calib-file {args.output.as_posix()} \\")
    print(f"    --resize 1920 1440")
    print(f"\n或直接用内参参数：")
    print(f"    --camera-matrix {fx:.2f} {fy:.2f} {cx:.2f} {cy:.2f}")

    # ── 畸变校正示例 ──
    sample_image = next((p for p in image_paths if p.suffix.lower() in IMAGE_EXTS), None)
    if sample_image:
        sample = imread_unicode(sample_image)
        if sample is not None:
            h, w = sample.shape[:2]
            new_camera, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            dst = cv2.undistort(sample, camera_matrix, dist_coeffs, None, new_camera)
            # 裁剪
            x, y, roi_w, roi_h = roi
            dst = dst[y:y + roi_h, x:x + roi_w]

            undist_path = args.output.parent / (args.output.stem + "_undistort_demo.jpg")
            cv2.imwrite(str(undist_path), dst)
            print(f"\nUndistortion demo: {undist_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
