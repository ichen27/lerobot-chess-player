import chess
import pytest
from chess_perception.game import ChessGame

@pytest.mark.parametrize("fen,move,commands", [
    (chess.STARTING_FEN, "e2e4", ["pick white pawn from e2, place on e4"]),
    ("4k3/8/8/8/8/4p3/3P4/4K3 w - - 0 1", "d2e3",
     ["remove black pawn from e3", "pick white pawn from d2, place on e3"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1",
     ["pick white king from e1, place on g1", "pick white rook from h1, place on f1"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6",
     ["remove black pawn from d5", "pick white pawn from e5, place on d6"]),
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q",
     ["pick white pawn from a7, place on a8", "promote to white queen on a8"]),
])
def test_legal_move_recovery_and_manipulation_commands(fen, move, commands):
    game = ChessGame("unused.pt")
    game.board = chess.Board(fen)
    after = game.board.copy()
    after.push_uci(move)
    assert ChessGame.detect_human_move(fen, after.fen()) == move
    assert game._build_move_commands(chess.Move.from_uci(move)) == commands

def test_illegal_observation_is_not_accepted_by_fallback():
    board = chess.Board()
    board.remove_piece_at(chess.E2)
    board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))
    assert ChessGame.detect_human_move(chess.STARTING_FEN, board.fen()) is None

def test_wrong_piece_identity_is_not_accepted_as_legal_move():
    board = chess.Board()
    board.remove_piece_at(chess.E2)
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    assert ChessGame.detect_human_move(chess.STARTING_FEN, board.fen()) is None

def test_invalid_observation_is_rejected():
    assert ChessGame.detect_human_move(chess.STARTING_FEN, "not a FEN") is None

def test_failed_robot_execution_propagates():
    game = ChessGame("unused.pt")
    game.pi0_policy = object()
    # Physical camera access is unavailable; fail at that external boundary.
    game.capture_image = lambda: (_ for _ in ()).throw(RuntimeError("camera lost"))
    with pytest.raises(RuntimeError):
        game.execute_robot_move("pick black pawn from e7, place on e5")
