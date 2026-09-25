import json
import subprocess
import sys

def test_demo_runs_without_model_camera_or_arm():
    result = subprocess.run([sys.executable, "-m", "chess_perception.demo"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["human_move"] == "e2e4"
    assert report["reply"] == "e7e5"
    assert report["commands"] == ["pick black pawn from e7, place on e5"]
    assert report["after"].split()[1] == "w"

def test_demo_rejects_illegal_input():
    result = subprocess.run([sys.executable, "-m", "chess_perception.demo", "--move", "e2e5"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "illegal" in result.stderr.lower()
