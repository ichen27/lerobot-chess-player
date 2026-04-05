"""YOLO-based chess piece detection."""

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


# The 12 piece classes expected from the trained YOLO model.
PIECE_CLASSES = [
    "black-bishop", "black-king", "black-knight", "black-pawn",
    "black-queen", "black-rook", "white-bishop", "white-king",
    "white-knight", "white-pawn", "white-queen", "white-rook",
]


@dataclass
class Detection:
    piece_class: str
    confidence: float
    cx: float  # bounding box center x
    cy: float  # bounding box center y
    x1: float
    y1: float
    x2: float
    y2: float


class ChessPieceDetector:
    """Wraps a YOLOv8 model trained on chess piece images."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.model: YOLO | None = None
        self.confidence_threshold = confidence_threshold

    def load_model(self, weights_path: str) -> None:
        """Load trained YOLO weights."""
        self.model = YOLO(weights_path)

    def detect_pieces(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a BGR image and return detections.

        Args:
            image: BGR image (numpy array from OpenCV).

        Returns:
            List of Detection objects for pieces above the confidence threshold.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(image, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append(Detection(
                piece_class=cls_name,
                confidence=conf,
                cx=cx, cy=cy,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return detections
