import pytest
from checkers_robot.rules import Position
from checkers_robot.game import Session, match_move, plan_move
from checkers_robot.engine import best_move


def test_ai_takes_forced_capture():
    p = Position({"a2": 1, "b3": -1, "f5": -1})
    assert str(best_move(p, depth=3)) == "a2xc4"


def test_ai_returns_none_when_blocked():
    assert best_move(Position({"a6": 1, "f1": -1})) is None


def test_capture_plan_includes_every_jump_and_crown():
    p = Position({"a2": 1, "b3": -1, "d5": -1, "f5": -1})
    commands = plan_move(p, p.parse_move("a2xc4xe6"))
    assert [(c.kind, c.source, c.target) for c in commands] == [
        ("remove", "b3", None),
        ("move", "a2", "c4"),
        ("remove", "d5", None),
        ("move", "c4", "e6"),
        ("crown", "e6", None),
    ]


def test_observation_matches_legal_move_and_preserves_king_identity():
    p = Position({"c4": 2, "f5": -1})
    assert str(match_move(p, {"b3": 1, "f5": -1})) == "c4-b3"


def test_incorrect_piece_color_does_not_match_move():
    p = Position.initial()
    observed = p.ownership()
    del observed["a2"]
    observed["b3"] = -1
    assert match_move(p, observed) is None


def test_failed_executor_does_not_advance():
    session = Session(Position.initial())
    before = session.position

    def fail(_):
        raise RuntimeError("motor unavailable")

    with pytest.raises(RuntimeError):
        session.execute_turn(before.parse_move("a2-b3"), fail, confirm=lambda: True)
    assert session.position is before


def test_unconfirmed_manual_move_does_not_advance():
    session = Session(Position.initial())
    before = session.position
    with pytest.raises(RuntimeError, match="confirmed"):
        session.execute_turn(
            before.parse_move("a2-b3"), lambda _: None, confirm=lambda: False
        )
    assert session.position is before


def test_mismatched_post_move_observation_does_not_advance():
    session = Session(Position.initial())
    before = session.position
    with pytest.raises(RuntimeError, match="verification"):
        session.execute_turn(
            before.parse_move("a2-b3"), lambda _: None, observe=before.ownership
        )
    assert session.position is before


def test_matching_observation_commits_move():
    session = Session(Position.initial())
    move = session.position.parse_move("a2-b3")
    observed = session.position.ownership()
    del observed["a2"]
    observed["b3"] = 1
    session.execute_turn(move, lambda _: None, observe=lambda: observed)
    assert session.position.turn == -1
    assert session.position.pieces["b3"] == 1
    assert "a2" not in session.position.pieces


def test_no_confirmation_strategy_rejected_before_any_command():
    session = Session(Position.initial())
    executed = []
    with pytest.raises(ValueError):
        session.execute_turn(session.position.parse_move("a2-b3"), executed.append)
    assert not executed


def test_matching_observation_does_not_override_declined_confirmation():
    position = Position({"b5": 1, "f5": -1})
    move = position.parse_move("b5-a6")
    session = Session(position)
    with pytest.raises(RuntimeError, match="not confirmed"):
        session.execute_turn(
            move,
            lambda command: None,
            observe=lambda: position.apply(move).ownership(),
            confirm=lambda: False,
        )
    assert session.position is position


def test_crowning_requires_confirmation_before_any_execution():
    position = Position({"b5": 1, "f5": -1})
    move = position.parse_move("b5-a6")
    session = Session(position)
    commands = []
    with pytest.raises(ValueError, match="Crowning"):
        session.execute_turn(
            move, commands.append, observe=lambda: position.apply(move).ownership()
        )
    assert commands == []
    assert session.position is position


def test_confirmed_crown_and_matching_observation_commit():
    position = Position({"b5": 1, "f5": -1})
    move = position.parse_move("b5-a6")
    session = Session(position)
    session.execute_turn(
        move,
        lambda command: None,
        observe=lambda: position.apply(move).ownership(),
        confirm=lambda: True,
    )
    assert session.position.pieces["a6"] == 2
