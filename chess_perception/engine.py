"""Chess engine wrapper using python-chess and Stockfish."""

from __future__ import annotations

import chess
import chess.engine


class ChessEngine:
    """Wraps python-chess + Stockfish for move analysis."""

    def __init__(
        self,
        stockfish_path: str = "stockfish",
        skill_level: int = 20,
        time_limit: float = 1.0,
    ):
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level
        self.time_limit = time_limit

    def validate_fen(self, fen: str) -> bool:
        """Check whether a FEN string represents a legal board position."""
        try:
            board = chess.Board(fen)
            return board.is_valid()
        except ValueError:
            return False

    def get_best_move(self, fen: str, color: str = "white") -> str | None:
        """Return the best move in UCI notation (e.g. 'e2e4').

        Args:
            fen: FEN string of the current position.
            color: Which side to move ('white' or 'black'). If the FEN
                already encodes the side to move this parameter adjusts it.

        Returns:
            Best move as a UCI string, or None if no legal moves exist.
        """
        board = chess.Board(fen)

        # Override side to move if needed.
        desired_turn = chess.WHITE if color == "white" else chess.BLACK
        if board.turn != desired_turn:
            parts = fen.split()
            parts[1] = "w" if color == "white" else "b"
            board = chess.Board(" ".join(parts))

        if not any(board.legal_moves):
            return None

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            engine.configure({"Skill Level": self.skill_level})
            result = engine.play(board, chess.engine.Limit(time=self.time_limit))
            if result.move is None:
                return None
            return result.move.uci()

    @staticmethod
    def move_to_description(move: str, fen: str) -> str:
        """Convert a UCI move to a human-readable pick-and-place description.

        Args:
            move: UCI string like 'e2e4'.
            fen: Current board FEN (used to identify the piece).

        Returns:
            Description string for the robot arm, e.g.
            "pick white pawn from e2, place on e4".
        """
        board = chess.Board(fen)
        from_sq = chess.parse_square(move[:2])
        to_sq = chess.parse_square(move[2:4])

        piece = board.piece_at(from_sq)
        piece_name = chess.piece_name(piece.piece_type) if piece else "piece"
        color_name = "white" if piece and piece.color == chess.WHITE else "black"

        captured = board.piece_at(to_sq)
        desc = f"pick {color_name} {piece_name} from {move[:2]}, place on {move[2:4]}"

        if captured:
            cap_color = "white" if captured.color == chess.WHITE else "black"
            cap_name = chess.piece_name(captured.piece_type)
            desc += f" (captures {cap_color} {cap_name})"

        # Promotion
        if len(move) == 5:
            promo_map = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
            promo = promo_map.get(move[4], move[4])
            desc += f" (promotes to {promo})"

        return desc
