import pytest
from checkers_robot.rules import Position


def test_initial_six_board_matches_photo_layout():
    p = Position.initial()
    assert p.size == 6
    assert len(p.pieces) == 12
    assert p.pieces["a2"] == 1
    assert p.pieces["a6"] == -1
    assert "a2-b3" in [str(m) for m in p.legal_moves()]


def test_eight_board_has_standard_twelve_pieces_each():
    p = Position.initial(8)
    assert len(p.pieces) == 24
    assert p.pieces["a3"] == 1
    assert p.pieces["b8"] == -1


def test_capture_is_mandatory_even_when_another_piece_can_slide():
    p = Position({"a2": 1, "e2": 1, "b3": -1, "f5": -1})
    assert [str(m) for m in p.legal_moves()] == ["a2xc4"]


def test_complete_multi_jump_removes_both_opponents():
    p = Position({"a2": 1, "b3": -1, "d5": -1, "f5": -1})
    m = p.parse_move("a2xc4xe6")
    assert m.captures == ("b3", "d5")
    after = p.apply(m)
    assert dict(after.pieces) == {"e6": 2, "f5": -1}
    assert after.turn == -1
    assert dict(p.pieces) == {"a2": 1, "b3": -1, "d5": -1, "f5": -1}


def test_partial_jump_cannot_be_played():
    p = Position({"a2": 1, "b3": -1, "d5": -1})
    with pytest.raises(ValueError, match="Illegal"):
        p.parse_move("a2xc4")


def test_men_cannot_move_or_capture_backwards():
    p = Position({"c4": 1, "b3": -1, "f5": -1})
    assert "c4xa2" not in [str(m) for m in p.legal_moves()]
    assert "c4-d3" not in [str(m) for m in p.legal_moves()]


def test_king_can_capture_backwards():
    p = Position({"c4": 2, "b3": -1, "f5": -1})
    assert [str(m) for m in p.legal_moves()] == ["c4xa2"]


def test_crowning_ends_turn_even_if_new_king_could_jump_backwards():
    p = Position({"c6": 1, "d7": -1, "f7": -1}, size=8, dark_parity=0)
    assert [str(m) for m in p.legal_moves()] == ["c6xe8"]
    assert p.apply(p.parse_move("c6xe8")).pieces["e8"] == 2


def test_invalid_or_light_square_piece_rejected():
    for pieces in [{"a1": 1}, {"z9": 1}, {"a2": 3}]:
        with pytest.raises(ValueError):
            Position(pieces)


def test_wrong_turn_move_is_rejected():
    with pytest.raises(ValueError, match="Illegal"):
        Position.initial().parse_move("b5-a4")


def test_no_legal_moves_is_loss():
    assert Position({"a6": 1, "f1": -1}).winner() == -1


def test_position_dictionary_is_immutable():
    p = Position.initial()
    with pytest.raises(TypeError):
        p.pieces["a2"] = -1
