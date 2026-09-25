import numpy as np
import pytest
from chess_perception.board import BoardStateExtractor
from chess_perception.detect import Detection

CORNERS = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)

def detection(name="white-queen", confidence=.95, x=450, y=650):
    return Detection(name, confidence, x, y, x-10, y-10, x+10, y+10)

@pytest.mark.parametrize("reverse", [False, True])
def test_same_square_keeps_stronger_detection(reverse):
    pieces = [detection(), detection("black-pawn", .1)]
    if reverse:
        pieces.reverse()
    assert BoardStateExtractor().map_pieces_to_squares(pieces, CORNERS) == {"e2": "white-queen"}

@pytest.mark.parametrize("x,y", [(-1, 650), (800, 650), (450, -1), (450, 800), (float("nan"), 10)])
def test_off_board_or_nonfinite_detection_does_not_create_piece(x, y):
    assert BoardStateExtractor().map_pieces_to_squares([detection(x=x, y=y)], CORNERS) == {}

def test_black_orientation_rotates_square_mapping():
    assert BoardStateExtractor("black_bottom").map_pieces_to_squares([detection()], CORNERS) == {"d7": "white-queen"}

def test_unknown_piece_is_rejected_instead_of_generating_invalid_fen():
    with pytest.raises(ValueError, match="piece"):
        BoardStateExtractor.to_fen({"e2": "bishop"})

def test_invalid_square_is_rejected_instead_of_silently_disappearing():
    with pytest.raises(ValueError, match="square"):
        BoardStateExtractor.to_fen({"z9": "white-queen"})

def test_degenerate_corners_are_rejected():
    with pytest.raises(ValueError, match="corner"):
        BoardStateExtractor().map_pieces_to_squares([detection()], np.zeros((4, 2), dtype=np.float32))

def test_point_on_projective_horizon_outside_trapezoid_is_rejected():
    corners = np.array([[200,400],[600,400],[800,800],[0,800]], dtype=np.float32)
    assert BoardStateExtractor().map_pieces_to_squares(
        [detection("black-rook", .99, 400, 0)], corners
    ) == {}

def test_triangle_is_not_promoted_to_a_board_bounding_box():
    import cv2
    image = np.zeros((800,800,3), dtype=np.uint8)
    cv2.fillConvexPoly(image, np.array([[400,100],[700,700],[100,700]]), (255,255,255))
    assert BoardStateExtractor().detect_board_corners(image) is None
