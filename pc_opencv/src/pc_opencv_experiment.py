from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ExperimentConfig:
    board_width_mm: float = 168.1
    board_height_mm: float = 255.1
    warp_width_px: int = 800
    min_board_area_ratio: float = 0.04
    inner_margin_ratio: float = 0.08
    min_inner_area_px: float = 400.0

    @property
    def warp_height_px(self) -> int:
        return round(self.warp_width_px * self.board_height_mm / self.board_width_mm)


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
) -> list[ShapeMeasurement]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    binary = preprocess_for_black_contours(image)
    board_quad = find_board_quad(binary, config)
    if board_quad is None:
        raise ValueError(f"No rectangular target board found: {image_path}")

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

    warped, _ = warp_board(image, board_quad, config)
    measurements, warped_annotated = detect_inner_shapes(warped, image_path.name, config)

    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_binary.png"), binary)
    cv2.imwrite(str(output_dir / f"{stem}_raw.png"), raw_annotated)
    cv2.imwrite(str(output_dir / f"{stem}_warped.png"), warped_annotated)
    return measurements


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
        description="PC OpenCV experiment for perspective correction and shape measurement."
    )
    parser.add_argument("--input", type=Path, help="Input image file or image directory.")
    parser.add_argument("--output", type=Path, default=Path("pc_opencv/data/output"))
    parser.add_argument("--camera", type=int, help="USB camera index, usually 0.")
    parser.add_argument("--capture", type=Path, help="Path to save one camera frame.")
    parser.add_argument("--board-width-mm", type=float, default=168.1)
    parser.add_argument("--board-height-mm", type=float, default=255.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.camera is not None:
        if args.capture is None:
            raise SystemExit("--capture is required when --camera is used")
        capture_from_camera(args.camera, args.capture)
        print(f"Captured frame: {args.capture}")
        return 0

    if args.input is None:
        raise SystemExit("--input is required unless --camera is used")

    config = ExperimentConfig(
        board_width_mm=args.board_width_mm,
        board_height_mm=args.board_height_mm,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    all_measurements: list[ShapeMeasurement] = []
    for image_path in iter_input_images(args.input):
        try:
            measurements = process_image(image_path, args.output, config)
            all_measurements.extend(measurements)
            print(f"Processed {image_path}: {len(measurements)} shapes")
        except ValueError as exc:
            print(f"Skipped {image_path}: {exc}")

    write_measurements_csv(args.output / "measurements.csv", all_measurements)
    print(f"Wrote {args.output / 'measurements.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
