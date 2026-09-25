import json
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
from checkers_robot.vision import BoardGeometry, ColorDetector, load_calibration

CORNERS = [[0, 0], [599, 0], [599, 599], [0, 599]]
COLORS = {
    "orange": {"lower": [5, 100, 80], "upper": [35, 255, 255]},
    "gray": {"lower": [0, 0, 70], "upper": [179, 80, 185]},
}


def board_image():
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    for row in range(6):
        for col in range(6):
            image[row * 100 : (row + 1) * 100, col * 100 : (col + 1) * 100] = (
                20 if (row + col) % 2 == 0 else 235
            )
    cv2.circle(image, (50, 450), 30, (0, 140, 240), -1)  # a2 orange
    cv2.circle(image, (150, 150), 30, (130, 130, 130), -1)  # b5 gray
    return image


def test_color_detection_from_pixels():
    detector = ColorDetector(BoardGeometry(CORNERS), COLORS)
    observed, annotated = detector.process(board_image())
    assert observed == {"a2": 1, "b5": -1}
    assert annotated.shape == (600, 600, 3)


def test_empty_board_is_empty_not_dark_squares_as_gray():
    image = board_image()
    image[400:500, 0:100] = 20
    image[100:200, 100:200] = 20
    assert ColorDetector(BoardGeometry(CORNERS), COLORS).process(image)[0] == {}


def test_rotated_camera_maps_back_to_same_game_coordinates():
    rotated = cv2.rotate(board_image(), cv2.ROTATE_180)
    assert ColorDetector(
        BoardGeometry(CORNERS, orientation="gray_bottom"), COLORS
    ).process(rotated)[0] == {"a2": 1, "b5": -1}


@pytest.mark.parametrize(
    "corners",
    [
        [[0, 0]] * 4,
        [[0, 0], [599, 599], [599, 0], [0, 599]],
        [[0, 0], [float("nan"), 0], [599, 599], [0, 599]],
    ],
)
def test_invalid_geometry_is_rejected(corners):
    with pytest.raises(ValueError, match="corners"):
        BoardGeometry(corners)


def test_point_on_projective_horizon_is_not_a_corner_piece():
    g = BoardGeometry([[200, 400], [600, 400], [800, 800], [0, 800]])
    assert g.point_square(400, 0) is None


@pytest.mark.parametrize(
    "x,y", [(-1, 100), (600, 100), (100, -1), (100, 600), (float("nan"), 100)]
)
def test_outside_or_nonfinite_points_rejected(x, y):
    assert BoardGeometry(CORNERS).point_square(x, y) is None


def test_calibration_corners_outside_frame_fail():
    with pytest.raises(ValueError, match="frame"):
        ColorDetector(BoardGeometry(CORNERS), COLORS).process(
            np.zeros((100, 100, 3), np.uint8)
        )


def test_overlapping_color_evidence_is_rejected():
    colors = {name: COLORS["orange"] for name in ("orange", "gray")}
    with pytest.raises(ValueError, match="Ambiguous"):
        ColorDetector(BoardGeometry(CORNERS), colors).process(board_image())


def test_bad_calibration_file_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"corners": CORNERS, "size": 7, "colors": COLORS}))
    with pytest.raises(ValueError):
        load_calibration(path)


def test_color_ranges_must_be_valid_hsv():
    invalid = {
        "orange": {"lower": [190, 0, 0], "upper": [200, 255, 255]},
        "gray": COLORS["gray"],
    }
    with pytest.raises(ValueError, match="HSV"):
        ColorDetector(BoardGeometry(CORNERS), invalid)


def test_annotation_does_not_mutate_input():
    frame = board_image()
    before = frame.copy()
    ColorDetector(BoardGeometry(CORNERS), COLORS).process(frame)
    assert np.array_equal(frame, before)


def test_supplied_calibrated_screenshot_matches_manually_checked_pieces():
    from pathlib import Path
    from checkers_robot.vision import load_calibration

    root = Path(__file__).resolve().parents[1]
    detector = load_calibration(root / "examples/screenshot-calibration.json")
    result, _ = detector.process(
        cv2.imread(str(root / "docs/media/checkers-board.png"))
    )
    assert result == {
        "a6": -1,
        "c6": -1,
        "e6": -1,
        "a4": -1,
        "d3": 1,
        "c2": -1,
        "e2": 1,
        "b1": 1,
        "d1": 1,
        "f1": 1,
    }
