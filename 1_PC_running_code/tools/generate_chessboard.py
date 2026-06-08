"""
生成可用于打印的棋盘格标定板图像。

棋盘格贴在硬纸板上，用于 OpenCV 相机标定。
默认生成 A4 尺寸，8×6 内角点棋盘格（9×7 格子），每格 25mm。

用法：
  python tools/generate_chessboard.py
  python tools/generate_chessboard.py --cols 9 --rows 7 --square-mm 30 --output data/calib/chessboard.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a printable chessboard pattern.")
    parser.add_argument("--cols", type=int, default=9,
                        help="Number of inner corners per row (default: 9 → 10×7 grid)")
    parser.add_argument("--rows", type=int, default=7,
                        help="Number of inner corners per column (default: 7 → 10×7 grid)")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="Square side length in mm (default: 25)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    parser.add_argument("--output", type=Path, default=Path("data/calib/chessboard.png"),
                        help="Output image path (default: data/calib/chessboard.png)")
    parser.add_argument("--margin-mm", type=float, default=15.0,
                        help="White margin around board in mm (default: 15)")
    args = parser.parse_args()

    # 棋盘格实际使用 (cols-1)×(rows-1) 个格子
    grid_cols = args.cols - 1  # 格子列数
    grid_rows = args.rows - 1  # 格子行数
    square_px = round(args.square_mm * args.dpi / 25.4)
    margin_px = round(args.margin_mm * args.dpi / 25.4)

    board_w = grid_cols * square_px
    board_h = grid_rows * square_px
    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px

    # 创建白色画布
    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    # 绘制棋盘格
    for i in range(grid_rows):
        for j in range(grid_cols):
            if (i + j) % 2 == 0:  # 黑色格子
                x0 = margin_px + j * square_px
                y0 = margin_px + i * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                img[y0:y1, x0:x1] = 0

    # 转 BGR 供 cv2.imwrite
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 添加标注文字
    info = f"{grid_cols}x{grid_rows}  grid, {args.square_mm:.0f}mm/sq  |  inner corners: {args.cols}x{args.rows}"
    cv2.putText(img_bgr, info, (margin_px, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 同时保存参数说明
    info_path = args.output.with_suffix(".txt")
    with open(info_path, "w") as f:
        f.write(f"Chessboard calibration pattern\n")
        f.write(f"Grid size: {grid_cols} cols × {grid_rows} rows\n")
        f.write(f"Square size: {args.square_mm} mm\n")
        f.write(f"Inner corners: {args.cols} (cols) × {args.rows} (rows)\n")
        f.write(f"Physical board size: {grid_cols * args.square_mm:.0f} × {grid_rows * args.square_mm:.0f} mm\n")
        f.write(f"\nFor OpenCV calibration:\n")
        f.write(f"  pattern_size = ({args.cols}, {args.rows})\n")
        f.write(f"  square_size = {args.square_mm}  # mm\n")

    print(f"Chessboard saved: {args.output.resolve()}")
    print(f"  Grid: {grid_cols}×{grid_rows} cells ({grid_cols * args.square_mm:.0f}×{grid_rows * args.square_mm:.0f} mm)")
    print(f"  Inner corners: {args.cols}×{args.rows}")
    print(f"  Square size: {args.square_mm} mm")
    print(f"  Resolution: {img_w}×{img_h} px @ {args.dpi} DPI")
    print(f"  Info: {info_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
