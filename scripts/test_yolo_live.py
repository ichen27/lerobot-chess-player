#!/usr/bin/env python3
"""Live camera view with YOLO piece detection and board grid overlay.

Uses the `supervision` library for annotated overlays, NMS, and
temporal smoothing of detections across frames.

Usage:
    python scripts/test_yolo_live.py
    python scripts/test_yolo_live.py --camera 1 --confidence 0.4
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chess_perception.board import BoardStateExtractor

FILES = "abcdefgh"
RANKS = "12345678"

# Board grid colours (BGR)
BOARD_OUTLINE_COLOR = (255, 255, 0)
GRID_COLOR = (180, 180, 0)
SQUARE_LABEL_COLOR = (200, 200, 200)


def draw_board_grid(
    image: np.ndarray,
    corners: np.ndarray,
    orientation: str = "white_bottom",
) -> None:
    """Draw the 8x8 grid lines and square labels on the image."""
    tl, tr, br, bl = corners

    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=BOARD_OUTLINE_COLOR, thickness=3)

    for i in range(1, 8):
        t = i / 8.0
        left = tl + t * (bl - tl)
        right = tr + t * (br - tr)
        cv2.line(image, tuple(left.astype(int)), tuple(right.astype(int)), GRID_COLOR, 1)

        top = tl + t * (tr - tl)
        bottom = bl + t * (br - bl)
        cv2.line(image, tuple(top.astype(int)), tuple(bottom.astype(int)), GRID_COLOR, 1)

    for row in range(8):
        for col in range(8):
            u = (col + 0.5) / 8.0
            v = (row + 0.5) / 8.0
            top_pt = tl + u * (tr - tl)
            bot_pt = bl + u * (br - bl)
            center = top_pt + v * (bot_pt - top_pt)
            cx, cy = int(center[0]), int(center[1])

            if orientation == "white_bottom":
                file_char = FILES[col]
                rank_char = RANKS[7 - row]
            else:
                file_char = FILES[7 - col]
                rank_char = RANKS[row]

            label = f"{file_char}{rank_char}"
            cv2.putText(
                image, label, (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, SQUARE_LABEL_COLOR, 1,
                cv2.LINE_AA,
            )


def draw_hud(image: np.ndarray, num_detections: int, board_found: bool, conf: float) -> None:
    """Draw a heads-up display with detection stats."""
    h, w = image.shape[:2]
    status = f"Pieces: {num_detections}  |  Board: {'YES' if board_found else 'NO'}  |  Conf: {conf:.0%}"
    cv2.rectangle(image, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(
        image, status, (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "q=quit  s=save  +/-=confidence  t=toggle smoothing", (w - 480, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live YOLO chess piece detection viewer")
    parser.add_argument("--weights", default=os.path.expanduser("~/chess/weights/chess_pieces/weights/best.pt"))
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0=Innomaker robot cam)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--orientation", default="white_bottom", choices=["white_bottom", "black_bottom"])
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: weights not found at {args.weights}")
        sys.exit(1)

    # Load YOLO model directly (supervision integrates with ultralytics)
    print("Loading YOLO model...")
    model = YOLO(args.weights)

    board_extractor = BoardStateExtractor(board_orientation=args.orientation)

    # --- Supervision annotators ---
    # White pieces = green, black pieces = orange
    white_color = sv.Color(0, 220, 0)
    black_color = sv.Color(0, 100, 255)

    def color_lookup(det_idx: int, detections: sv.Detections) -> sv.Color:
        class_name = detections.data.get("class_name", [])
        if det_idx < len(class_name) and class_name[det_idx].startswith("white"):
            return white_color
        return black_color

    # Use ColorPalette with piece-aware colors
    # Build a palette: map each class_id to white or black color
    class_names = model.names  # {0: 'bishop', 1: 'black-bishop', ...}
    palette_colors = []
    for cls_id in sorted(class_names.keys()):
        name = class_names[cls_id]
        if name.startswith("white"):
            palette_colors.append(sv.Color(0, 220, 0))
        elif name.startswith("black"):
            palette_colors.append(sv.Color(0, 100, 255))
        else:
            palette_colors.append(sv.Color(200, 200, 0))  # ambiguous class like "bishop"
    palette = sv.ColorPalette(palette_colors)

    box_annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=15, color=palette)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.4,
        text_padding=4,
        color=palette,
        text_position=sv.Position.TOP_LEFT,
    )
    dot_annotator = sv.DotAnnotator(radius=4, color=palette)

    # Detection smoother: stabilises detections across frames
    smoother = sv.DetectionsSmoother(length=5)
    smoothing_enabled = True

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf = args.confidence
    iou_thresh = args.iou_threshold
    print(f"Running. Confidence: {conf:.0%}  |  IoU NMS: {iou_thresh}")
    print("  q = quit | s = save | +/- = confidence | t = toggle smoothing")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display = frame.copy()

        # Run YOLO inference
        results = model(frame, conf=conf, iou=iou_thresh, verbose=False)[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Apply class-agnostic NMS to remove overlapping boxes across classes
        detections = detections.with_nms(threshold=iou_thresh, class_agnostic=True)

        # Temporal smoothing
        if smoothing_enabled:
            detections = smoother.update_with_detections(detections)

        # Detect board corners and draw grid
        corners = board_extractor.detect_board_corners(frame)
        if corners is not None:
            draw_board_grid(display, corners, args.orientation)

        # Build labels: "white-pawn 92%"
        labels = []
        class_name_data = detections.data.get("class_name", [])
        for i in range(len(detections)):
            name = class_name_data[i] if i < len(class_name_data) else "?"
            c = detections.confidence[i] if detections.confidence is not None else 0
            labels.append(f"{name} {c:.0%}")

        # Annotate with supervision
        display = box_annotator.annotate(scene=display, detections=detections)
        display = label_annotator.annotate(scene=display, detections=detections, labels=labels)
        display = dot_annotator.annotate(scene=display, detections=detections)

        # HUD
        draw_hud(display, len(detections), corners is not None, conf)

        cv2.imshow("YOLO Chess Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("yolo_detection_frame.png", display)
            print("Saved yolo_detection_frame.png")
        elif key in (ord("+"), ord("=")):
            conf = min(conf + 0.05, 0.95)
            print(f"Confidence threshold: {conf:.0%}")
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.10)
            print(f"Confidence threshold: {conf:.0%}")
        elif key == ord("t"):
            smoothing_enabled = not smoothing_enabled
            print(f"Temporal smoothing: {'ON' if smoothing_enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
