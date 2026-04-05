#!/usr/bin/env python3
"""Quick test: capture one frame, run the pipeline, print results."""

import os
import sys
import cv2


def main():
    weights = os.path.expanduser("~/chess/weights/chess_pieces/weights/best.pt")
    if not os.path.exists(weights):
        print(f"Error: YOLO weights not found at {weights}")
        print("Run train_yolo.py first, or pass a custom path as the first argument.")
        if len(sys.argv) > 1:
            weights = sys.argv[1]
        else:
            sys.exit(1)

    stockfish_path = "stockfish"
    camera_index = 0

    # Allow overrides via CLI args.
    if len(sys.argv) > 2:
        stockfish_path = sys.argv[2]
    if len(sys.argv) > 3:
        camera_index = int(sys.argv[3])

    from chess_perception.pipeline import ChessPerceptionPipeline

    print("Initializing pipeline ...")
    pipeline = ChessPerceptionPipeline(
        yolo_weights=weights,
        stockfish_path=stockfish_path,
    )

    print(f"Capturing frame from camera {camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame")
        sys.exit(1)

    print("Running pipeline ...")
    result = pipeline.process_frame(frame)

    print(f"\nDetections: {len(result['detections'])} pieces found")
    print(f"Piece map: {result['piece_map']}")
    print(f"FEN: {result['fen']}")
    print(f"Best move: {result['best_move']}")
    print(f"Description: {result['move_description']}")

    if result["error"]:
        print(f"Error: {result['error']}")

    # Save annotated image.
    out_path = os.path.expanduser("~/chess/annotated_output.png")
    if result["annotated_image"] is not None:
        cv2.imwrite(out_path, result["annotated_image"])
        print(f"\nAnnotated image saved to {out_path}")


if __name__ == "__main__":
    main()
