"""Board detection and piece-to-square mapping."""

from __future__ import annotations

import cv2
import numpy as np

from chess_perception.detect import Detection

# FEN piece symbols keyed by detection class name.
_PIECE_FEN = {
    "black-bishop": "b", "black-king": "k", "black-knight": "n",
    "black-pawn": "p", "black-queen": "q", "black-rook": "r",
    "white-bishop": "B", "white-king": "K", "white-knight": "N",
    "white-pawn": "P", "white-queen": "Q", "white-rook": "R",
}

FILES = "abcdefgh"
RANKS = "12345678"


class BoardStateExtractor:
    """Detects the chessboard grid and maps pieces to squares."""

    def __init__(self, board_orientation: str = "white_bottom"):
        """
        Args:
            board_orientation: "white_bottom" means rank 1 is at the bottom
                of the image, "black_bottom" means rank 8 is at the bottom.
        """
        if board_orientation not in ("white_bottom", "black_bottom"):
            raise ValueError("board_orientation must be 'white_bottom' or 'black_bottom'")
        self.board_orientation = board_orientation

    # ------------------------------------------------------------------
    # Board corner detection
    # ------------------------------------------------------------------

    def detect_board_corners(self, image: np.ndarray) -> np.ndarray | None:
        """Find the four outer corners of the chessboard in the image.

        Uses edge detection + contour detection to find the largest
        quadrilateral, which is assumed to be the board.

        Args:
            image: BGR image.

        Returns:
            4x2 float array of corner points ordered [TL, TR, BR, BL],
            or None if detection fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close gaps in edges.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort by area descending and look for a quadrilateral.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:10]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._order_corners(corners)

        # An arbitrary contour's bounding box is not evidence of a board.
        return None

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order points as [TL, TR, BR, BL]."""
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        ordered[1] = pts[np.argmin(d)]   # top-right
        ordered[3] = pts[np.argmax(d)]   # bottom-left
        return ordered

    # ------------------------------------------------------------------
    # Piece mapping
    # ------------------------------------------------------------------

    def map_pieces_to_squares(
        self,
        detections: list[Detection],
        board_corners: np.ndarray,
    ) -> dict[str, str]:
        """Map detected pieces to algebraic square names.

        Uses a perspective transform to warp piece centres into a
        normalised 8x8 grid.

        Args:
            detections: Output of ChessPieceDetector.detect_pieces().
            board_corners: 4x2 array [TL, TR, BR, BL] of board corners.

        Returns:
            Dict mapping square name (e.g. "e2") to piece class name.
        """
        board_corners = np.asarray(board_corners, dtype=np.float32)
        if (board_corners.shape != (4, 2) or not np.isfinite(board_corners).all()
                or not cv2.isContourConvex(board_corners)
                or cv2.contourArea(board_corners) < 1):
            raise ValueError("Board corners must form a finite, convex quadrilateral.")
        # Destination square in normalised space (800x800).
        dst = np.array([
            [0, 0], [800, 0], [800, 800], [0, 800]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(board_corners, dst)

        piece_map: dict[str, str] = {}
        confidences: dict[str, float] = {}

        for det in detections:
            if not np.isfinite([det.cx, det.cy, det.confidence]).all():
                continue
            # Reject outside points before warping: OpenCV maps a zero
            # homogeneous denominator to (0, 0), which looks like a valid square.
            if cv2.pointPolygonTest(board_corners, (float(det.cx), float(det.cy)), False) < 0:
                continue
            # Transform the detection centre into normalised board space.
            pt = np.array([[[det.cx, det.cy]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, M)[0][0]
            bx, by = warped

            # Off-board pieces must not become phantom pieces on edge squares.
            if not np.isfinite(warped).all() or not (0 <= bx < 800 and 0 <= by < 800):
                continue

            col = int(bx // 100)  # 0-7
            row = int(by // 100)  # 0-7
            col = min(col, 7)
            row = min(row, 7)

            if self.board_orientation == "white_bottom":
                # Image top = rank 8, image bottom = rank 1.
                file_char = FILES[col]
                rank_char = RANKS[7 - row]
            else:
                # Black bottom: image top = rank 1, image bottom = rank 8.
                file_char = FILES[7 - col]
                rank_char = RANKS[row]

            square = f"{file_char}{rank_char}"
            # If two pieces land on the same square, keep the higher-confidence one.
            if square not in piece_map or det.confidence > confidences[square]:
                piece_map[square] = det.piece_class
                confidences[square] = det.confidence

        return piece_map

    # ------------------------------------------------------------------
    # FEN generation
    # ------------------------------------------------------------------

    @staticmethod
    def to_fen(piece_map: dict[str, str], color: str = "white") -> str:
        """Convert a piece-on-square mapping to a FEN position string.

        Args:
            piece_map: Dict of square name -> piece class (e.g. "white-king").

        Returns:
            FEN with requested turn and placeholder history. For an ongoing
            game, use only its placement and preserve history separately.
        """
        if color not in ("white", "black"):
            raise ValueError("color must be white or black")
        for square, piece in piece_map.items():
            if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
                raise ValueError(f"Invalid square: {square}")
            if piece not in _PIECE_FEN:
                raise ValueError(f"Unsupported piece class: {piece}")
        rows: list[str] = []
        for rank_idx in range(7, -1, -1):  # rank 8 down to rank 1
            rank_char = RANKS[rank_idx]
            empty = 0
            row = ""
            for file_idx in range(8):
                file_char = FILES[file_idx]
                square = f"{file_char}{rank_char}"
                if square in piece_map:
                    if empty > 0:
                        row += str(empty)
                        empty = 0
                    piece_cls = piece_map[square]
                    fen_char = _PIECE_FEN[piece_cls]
                    row += fen_char
                else:
                    empty += 1
            if empty > 0:
                row += str(empty)
            rows.append(row)

        # An image cannot recover castling rights, en passant or move counters.
        turn = "w" if color == "white" else "b"
        return "/".join(rows) + f" {turn} - - 0 1"
