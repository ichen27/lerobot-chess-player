from types import SimpleNamespace
import chess
import numpy as np
from chess_perception.board import BoardStateExtractor
from chess_perception.pipeline import ChessPerceptionPipeline

def test_position_with_black_in_check_uses_requested_turn():
    pipeline = ChessPerceptionPipeline.__new__(ChessPerceptionPipeline)
    pipeline.detector = SimpleNamespace(detect_pieces=lambda _: [object()])
    pipeline.board_extractor = SimpleNamespace(
        detect_board_corners=lambda _: np.array([[0,0],[800,0],[800,800],[0,800]], dtype=np.float32),
        map_pieces_to_squares=lambda *_: {"e8":"black-king","e1":"white-king","e2":"white-rook"},
        to_fen=BoardStateExtractor.to_fen,
    )
    from chess_perception.engine import ChessEngine
    pipeline.engine = ChessEngine()
    # Avoid launching an external engine; validate its actual input.
    def best_move(fen, color):
        assert chess.Board(fen).turn == chess.BLACK
        return "e8d8"
    pipeline.engine.get_best_move = best_move
    pipeline._draw_detections = lambda image, *_: image
    result = pipeline.process_frame(np.zeros((800,800,3), dtype=np.uint8), color="black")
    assert result["error"] is None
    assert result["best_move"] == "e8d8"

def test_observation_mode_does_not_require_a_standalone_legal_position():
    pipeline = ChessPerceptionPipeline.__new__(ChessPerceptionPipeline)
    pipeline.detector = SimpleNamespace(detect_pieces=lambda _: [object()])
    pipeline.board_extractor = SimpleNamespace(
        detect_board_corners=lambda _: np.array([[0,0],[800,0],[800,800],[0,800]], dtype=np.float32),
        map_pieces_to_squares=lambda *_: {"e8":"black-king","e1":"white-king","e2":"white-rook"},
        to_fen=BoardStateExtractor.to_fen,
    )
    # No engine installed: observation must be usable independently of search.
    pipeline.engine = None
    pipeline._draw_detections = lambda image, *_: image
    result = pipeline.process_frame(np.zeros((800,800,3), dtype=np.uint8), analyze=False)
    assert result["error"] is None
    assert result["fen"].split()[0] == "4k3/8/8/8/8/8/4R3/4K3"
    assert result["best_move"] is None
