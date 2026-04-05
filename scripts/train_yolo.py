#!/usr/bin/env python3
"""Fine-tune YOLOv8s on the chess pieces dataset."""

import os
from ultralytics import YOLO


def main():
    data_yaml = os.path.expanduser("~/chess/data/chess_pieces/data.yaml")
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found. Run download_dataset.py first.")
        return

    weights_dir = os.path.expanduser("~/chess/weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Load pretrained YOLOv8s.
    model = YOLO("yolov8s.pt")

    # Train.
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        project=weights_dir,
        name="chess_pieces",
        exist_ok=True,
    )

    # The best weights are saved at weights_dir/chess_pieces/weights/best.pt
    best_weights = os.path.join(weights_dir, "chess_pieces", "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
