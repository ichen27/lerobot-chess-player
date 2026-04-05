"""End-to-end chess perception pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from chess_perception.detect import ChessPieceDetector, Detection
from chess_perception.board import BoardStateExtractor
from chess_perception.engine import ChessEngine


class ChessPerceptionPipeline:
    """Camera image -> piece detection -> FEN -> best move."""

    def __init__(
        self,
        yolo_weights: str,
        stockfish_path: str = "stockfish",
        board_orientation: str = "white_bottom",
        confidence_threshold: float = 0.5,
        skill_level: int = 20,
        time_limit: float = 1.0,
    ):
        self.detector = ChessPieceDetector(confidence_threshold=confidence_threshold)
        self.detector.load_model(yolo_weights)

        self.board_extractor = BoardStateExtractor(board_orientation=board_orientation)
        self.engine = ChessEngine(
            stockfish_path=stockfish_path,
            skill_level=skill_level,
            time_limit=time_limit,
        )
        self.board_orientation = board_orientation

    def process_frame(
        self,
        image: np.ndarray,
        color: str = "white",
    ) -> dict:
        """Run the full pipeline on a single BGR image.

        Args:
            image: BGR image (numpy array).
            color: Side to move ('white' or 'black').

        Returns:
            Dict with keys: fen, best_move, move_description, detections,
            piece_map, annotated_image, error.
        """
        result: dict = {
            "fen": None,
            "best_move": None,
            "move_description": None,
            "detections": [],
            "piece_map": {},
            "annotated_image": None,
            "error": None,
        }

        # Step 1: Detect pieces.
        try:
            detections = self.detector.detect_pieces(image)
            result["detections"] = detections
        except Exception as e:
            result["error"] = f"Detection failed: {e}"
            return result

        if not detections:
            result["error"] = "No pieces detected"
            result["annotated_image"] = image.copy()
            return result

        # Step 2: Find board corners.
        corners = self.board_extractor.detect_board_corners(image)
        if corners is None:
            result["error"] = "Could not detect board corners"
            result["annotated_image"] = self._draw_detections(image, detections)
            return result

        # Step 3: Map pieces to squares.
        piece_map = self.board_extractor.map_pieces_to_squares(detections, corners)
        result["piece_map"] = piece_map

        # Step 4: Generate FEN.
        fen = self.board_extractor.to_fen(piece_map)
        result["fen"] = fen

        # Step 5: Validate and get best move.
        if not self.engine.validate_fen(fen):
            result["error"] = f"Invalid FEN: {fen}"
            result["annotated_image"] = self._draw_detections(image, detections, corners)
            return result

        try:
            best_move = self.engine.get_best_move(fen, color=color)
            result["best_move"] = best_move
            if best_move:
                result["move_description"] = self.engine.move_to_description(best_move, fen)
        except Exception as e:
            result["error"] = f"Engine error: {e}"

        result["annotated_image"] = self._draw_detections(image, detections, corners)
        return result

    # ------------------------------------------------------------------
    # Live camera loop
    # ------------------------------------------------------------------

    def run_live(self, camera_index: int = 0, color: str = "white") -> None:
        """Capture from camera and process continuously.

        Press 'q' to quit, 's' to save current frame.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: cannot open camera {camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Chess perception pipeline running. Press 'q' to quit, 's' to save frame.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            result = self.process_frame(frame, color=color)

            # Print info.
            if result["fen"]:
                print(f"FEN: {result['fen']}")
            if result["best_move"]:
                print(f"Best move: {result['best_move']} — {result['move_description']}")
            if result["error"]:
                print(f"Error: {result['error']}")

            annotated = result["annotated_image"]
            if annotated is not None:
                cv2.imshow("Chess Perception", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("chess_frame.png", frame)
                if annotated is not None:
                    cv2.imwrite("chess_annotated.png", annotated)
                print("Saved chess_frame.png and chess_annotated.png")

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_detections(
        image: np.ndarray,
        detections: list[Detection],
        corners: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        annotated = image.copy()

        # Draw board outline if we have corners.
        if corners is not None:
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        for det in detections:
            color = (0, 200, 0) if det.piece_class.startswith("white") else (200, 0, 0)
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.piece_class} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        return annotated
