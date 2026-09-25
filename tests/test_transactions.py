from types import SimpleNamespace
import chess
import pytest
from chess_perception.game import ChessGame

@pytest.fixture
def game():
    result = ChessGame("unused.pt")
    result.board = chess.Board()
    result.board.push_uci("e2e4")
    result.prev_fen = result.board.fen()
    return result

def test_robot_failure_keeps_current_position(game):
    before = game.board.fen()
    game.pi0_policy = object()
    game.execute_robot_move = lambda _: (_ for _ in ()).throw(RuntimeError("motor unavailable"))
    with pytest.raises(RuntimeError):
        game.execute_robot_turn(chess.Move.from_uci("e7e5"))
    assert game.board.fen() == before
    assert game.prev_fen == before

def test_manual_move_requires_confirmation(game, monkeypatch):
    before = game.board.fen()
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(RuntimeError):
        game.execute_robot_turn(chess.Move.from_uci("e7e5"))
    assert game.board.fen() == before

def test_confirmed_manual_move_commits_position(game, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    game.execute_robot_turn(chess.Move.from_uci("e7e5"))
    assert game.board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)
    assert game.board.piece_at(chess.E7) is None
    assert game.board.turn == chess.WHITE
    assert game.prev_fen == game.board.fen()

@pytest.mark.parametrize("matching", [False, True])
def test_physical_move_requires_matching_post_move_observation(game, monkeypatch, matching):
    before = game.board.fen()
    game.pi0_policy = object()
    game.execute_robot_move = lambda _: None  # Motor boundary unavailable in CI.
    game.capture_image = lambda: "camera-frame"
    expected = game.board.copy()
    expected.push_uci("e7e5")
    game.pipeline = SimpleNamespace(process_frame=lambda *_, **__: {
        "error": None, "fen": expected.fen() if matching else before,
    })
    monkeypatch.setattr("chess_perception.game.time.sleep", lambda _: None)
    if matching:
        game.execute_robot_turn(chess.Move.from_uci("e7e5"))
        assert game.board.fen() == expected.fen()
    else:
        with pytest.raises(RuntimeError, match="verification"):
            game.execute_robot_turn(chess.Move.from_uci("e7e5"))
        assert game.board.fen() == before

def test_promotion_cannot_start_unvalidated_physical_sequence(game):
    game.board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    game.pi0_policy = object()
    executed = []
    game.execute_robot_move = executed.append
    with pytest.raises(RuntimeError, match="promotion"):
        game.execute_robot_turn(chess.Move.from_uci("a2a1q"))
    assert not executed

def test_illegal_robot_move_never_reaches_execution(game):
    executed = []
    game.execute_robot_move = executed.append
    with pytest.raises(ValueError):
        game.execute_robot_turn(chess.Move.from_uci("e7e4"))
    assert not executed
