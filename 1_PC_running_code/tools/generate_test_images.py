"""
生成模拟 A4 目标板的测试图像，用于在没有真实照片时验证代码流程。

生成策略：
  1. 在白色画布上以指定分辨率（ppm）绘制 A4 外框 + 内部图形
  2. 可选施加随机透视变换模拟不同拍摄角度
  3. 添加少量高斯噪声

不同 ppm 值模拟不同距离（ppm 越小 = 目标越小 = 距离越远）。

用法：
  # 生成默认测试图
  python tools/generate_test_images.py

  # 指定输出目录
  python tools/generate_test_images.py --output data/input
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# ── 目标板参数 ──
BOARD_W_MM = 168.1
BOARD_H_MM = 255.1
BORDER_W_MM = 20.0

SHAPE_PARAMS: dict[str, float] = {
    "triangle": 120.0,  # 等边三角形边长 mm
    "square": 100.0,    # 正方形边长 mm
    "circle": 100.0,    # 圆直径 mm
}


def make_board(ppm: float, shape: str, canvas_size: tuple[int, int]) -> np.ndarray:
    """在画布中央绘制 A4 目标板，ppm = 像素/毫米。"""
    cw, ch = canvas_size
    bw = round(BOARD_W_MM * ppm)
    bh = round(BOARD_H_MM * ppm)
    bp = round(BORDER_W_MM * ppm)  # 黑框宽度 px

    img = np.full((ch, cw, 3), 255, dtype=np.uint8)
    cx, cy = cw // 2, ch // 2

    # ── 外框（黑色外矩形 - 白色内矩形 = 黑边框） ──
    cv2.rectangle(img, (cx - bw // 2, cy - bh // 2),
                  (cx + bw // 2, cy + bh // 2), (0, 0, 0), -1)
    cv2.rectangle(img, (cx - bw // 2 + bp, cy - bh // 2 + bp),
                  (cx + bw // 2 - bp, cy + bh // 2 - bp), (255, 255, 255), -1)

    # ── 内部图形 ──
    size_mm = SHAPE_PARAMS[shape]
    size_px = round(size_mm * ppm)

    if shape == "triangle":
        r = size_px / np.sqrt(3)  # 外接圆半径
        pts = np.array([
            [cx, cy - r], [cx - size_px // 2, cy + r // 2],
            [cx + size_px // 2, cy + r // 2]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))
    elif shape == "square":
        half = size_px // 2
        cv2.rectangle(img, (cx - half, cy - half),
                      (cx + half, cy + half), (0, 0, 0), -1)
    elif shape == "circle":
        cv2.circle(img, (cx, cy), size_px // 2, (0, 0, 0), -1)

    return img


def apply_random_perspective(img: np.ndarray, max_angle_deg: float = 35.0) -> np.ndarray:
    """随机透视变换模拟倾斜。"""
    h, w = img.shape[:2]
    margin = w * 0.12 * (max_angle_deg / 35.0)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(180, 180, 180))


def add_noise(img: np.ndarray, level: float = 3.0) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * level
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic A4 target test images.")
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    parser.add_argument("--canvas", type=int, nargs=2, default=(1920, 1440),
                        metavar=("W", "H"), help="Canvas size (default: 1920 1440).")
    parser.add_argument("--noise", type=float, default=3.0)
    parser.add_argument("--angle-max", type=float, default=35.0,
                        help="Max perspective angle deg (default: 35).")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = ["triangle", "square", "circle"]
    rng = random.Random(42)

    # ── 距离模拟：不同 ppm ──
    scenarios = [
        ("100cm", 3.0, False),
        ("125cm", 2.6, False),
        ("150cm", 2.3, False),
        ("175cm", 2.0, False),
        ("200cm", 1.7, False),
    ]
    # ── 角度模拟 ──
    scenarios += [
        ("angle_15deg", 2.6, True),
        ("angle_30deg", 2.6, True),
    ]

    for label, ppm, do_angle in scenarios:
        shape = rng.choice(shapes)
        rng.seed(f"{label}_{shape}")
        np.random.seed(abs(hash(f"{label}_{shape}")) % (2**31))

        img = make_board(ppm, shape, tuple(args.canvas))

        if do_angle:
            img = apply_random_perspective(img, args.angle_max)

        img = add_noise(img, args.noise)

        filename = f"{label}_{shape}.jpg"
        cv2.imwrite(str(output_dir / filename), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {filename}  (ppm={ppm:.1f}, angle={do_angle})")

    print(f"Done — {len(scenarios)} images written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
