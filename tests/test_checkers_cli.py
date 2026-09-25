import json
import subprocess
import sys


def test_demo_plays_checkers_without_vision_or_chess_dependencies():
    run = subprocess.run(
        [sys.executable, "-m", "checkers_robot.demo"], text=True, capture_output=True
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    assert data["game"] == "checkers"
    assert data["board_size"] == 6
    assert data["human_move"] == "a2-b3"
    assert data["ai_move"]
    assert data["commands"]
    code = "import checkers_robot.demo; import sys; assert not {'chess','torch','cv2'} & set(sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_demo_rejects_illegal_chess_style_move():
    run = subprocess.run(
        [sys.executable, "-m", "checkers_robot.demo", "--move", "e2e4"],
        text=True,
        capture_output=True,
    )
    assert run.returncode != 0


def test_eight_square_demo_runs():
    run = subprocess.run(
        [sys.executable, "-m", "checkers_robot.demo", "--size", "8"],
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["board_size"] == 8


def test_interactive_game_accepts_move_and_ai_replies():
    run = subprocess.run(
        [sys.executable, "-m", "checkers_robot.cli"],
        input="?\ne2e4\na2-b3\nq\n",
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert "Illegal move" in run.stdout
    assert "AI:" in run.stdout
    assert "Preview:" in run.stdout
    assert "no hardware is connected" in run.stdout


def test_interactive_game_eof_exits_cleanly():
    run = subprocess.run(
        [sys.executable, "-m", "checkers_robot.cli"],
        input="",
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
