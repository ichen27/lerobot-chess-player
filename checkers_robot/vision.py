"""Calibrated board rectification and HSV disc ownership detection."""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from checkers_robot.rules import Position


class BoardGeometry:
    def __init__(self, corners, size=6, dark_parity=None, orientation="orange_bottom"):
        parity = (0 if size == 6 else 1) if dark_parity is None else dark_parity
        self.board = Position({}, size=size, dark_parity=parity)
        self.size = size
        if orientation not in ("orange_bottom", "gray_bottom"):
            raise ValueError("orientation must be orange_bottom or gray_bottom.")
        self.orientation = orientation
        self.corners = np.asarray(corners, dtype=np.float32)
        if (
            self.corners.shape != (4, 2)
            or not np.isfinite(self.corners).all()
            or not cv2.isContourConvex(self.corners)
            or cv2.contourArea(self.corners, oriented=True) < 1
        ):
            raise ValueError(
                "Board corners must be a convex TL, TR, BR, BL quadrilateral."
            )
        self.width = size * 100
        dst = np.array(
            [[0, 0], [self.width, 0], [self.width, self.width], [0, self.width]],
            np.float32,
        )
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)

    def point_square(self, x, y):
        if not np.isfinite([x, y]).all():
            return None
        if cv2.pointPolygonTest(self.corners, (float(x), float(y)), False) < 0:
            return None
        point = cv2.perspectiveTransform(
            np.array([[[x, y]]], np.float32), self.transform
        )[0, 0]
        if not np.isfinite(point).all() or not (
            (point >= 0).all() and (point < self.width).all()
        ):
            return None
        col, row = (point // 100).astype(int)
        if self.orientation == "gray_bottom":
            row, col = self.size - 1 - row, self.size - 1 - col
        return self.board.square(row, col)

    def warp(self, image):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("Camera frame must be a uint8 BGR image.")
        height, width = image.shape[:2]
        if (
            (self.corners < 0).any()
            or (self.corners[:, 0] >= width).any()
            or (self.corners[:, 1] >= height).any()
        ):
            raise ValueError(
                "Calibration corners are outside this frame; recalibrate at its resolution."
            )
        result = cv2.warpPerspective(image, self.transform, (self.width, self.width))
        return (
            cv2.rotate(result, cv2.ROTATE_180)
            if self.orientation == "gray_bottom"
            else result
        )


class ColorDetector:
    def __init__(self, geometry, colors, min_fraction=0.2):
        self.geometry = geometry
        if not np.isfinite(min_fraction) or not 0 < min_fraction <= 1:
            raise ValueError("min_fraction must be between 0 and 1.")
        self.min_fraction = min_fraction
        self.ranges = {}
        if set(colors) != {"orange", "gray"}:
            raise ValueError("Provide orange and gray HSV ranges.")
        for name, side in (("orange", 1), ("gray", -1)):
            lower = np.asarray(colors[name]["lower"], dtype=float)
            upper = np.asarray(colors[name]["upper"], dtype=float)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.isfinite([lower, upper]).all()
                or (lower < 0).any()
                or (upper > np.array([179, 255, 255])).any()
                or (lower > upper).any()
                or (lower != np.floor(lower)).any()
                or (upper != np.floor(upper)).any()
            ):
                raise ValueError(f"Invalid HSV range for {name}.")
            self.ranges[side] = (lower.astype(np.uint8), upper.astype(np.uint8))

    def process(self, image):
        warped = self.geometry.warp(image)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        annotated = warped.copy()
        observed = {}
        for row in range(self.geometry.size):
            for col in range(self.geometry.size):
                if (row + col) % 2 != self.geometry.board.dark_parity:
                    continue
                # Sample the inner square to avoid grid lines and neighboring discs.
                roi = hsv[
                    row * 100 + 15 : row * 100 + 85, col * 100 + 15 : col * 100 + 85
                ]
                candidates = []
                for side, (lower, upper) in self.ranges.items():
                    fraction = (
                        np.count_nonzero(cv2.inRange(roi, lower, upper))
                        / roi.shape[0]
                        / roi.shape[1]
                    )
                    if fraction >= self.min_fraction:
                        candidates.append(side)
                square = self.geometry.board.square(row, col)
                if len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous piece colors on {square}; recalibrate HSV ranges."
                    )
                if candidates:
                    side = candidates[0]
                    observed[square] = side
                    color = (0, 165, 255) if side == 1 else (200, 200, 200)
                    cv2.circle(
                        annotated, (col * 100 + 50, row * 100 + 50), 38, color, 3
                    )
                cv2.putText(
                    annotated,
                    square,
                    (col * 100 + 5, row * 100 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
        for n in range(self.geometry.size + 1):
            cv2.line(
                annotated, (n * 100, 0), (n * 100, self.geometry.width), (0, 200, 0), 1
            )
            cv2.line(
                annotated, (0, n * 100), (self.geometry.width, n * 100), (0, 200, 0), 1
            )
        return observed, annotated


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    try:
        geometry = BoardGeometry(
            data["corners"],
            size=data.get("size", 6),
            dark_parity=data.get("dark_parity"),
            orientation=data.get("orientation", "orange_bottom"),
        )
        return ColorDetector(geometry, data["colors"], data.get("min_fraction", 0.2))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Calibration needs corners and orange/gray lower/upper HSV ranges."
        ) from exc


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a checkers image using an explicit calibration."
    )
    parser.add_argument("image")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", help="Optional annotated, rectified image.")
    args = parser.parse_args()
    try:
        frame = cv2.imread(args.image)
        if frame is None:
            raise ValueError(f"Cannot read image: {args.image}")
        detector = load_calibration(args.calibration)
        observed, annotated = detector.process(frame)
        if args.output and not cv2.imwrite(args.output, annotated):
            raise ValueError(f"Cannot write image: {args.output}")
        print(
            json.dumps(
                {
                    "size": detector.geometry.size,
                    "pieces": observed,
                    "orange": sum(p == 1 for p in observed.values()),
                    "gray": sum(p == -1 for p in observed.values()),
                },
                indent=2,
            )
        )
    except (ValueError, OSError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
