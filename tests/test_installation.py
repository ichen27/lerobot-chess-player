import subprocess
import sys
from chess_perception.detect import ChessPieceDetector
import pytest

def test_help_does_not_import_or_initialize_deep_learning():
    code = "import chess_perception; import sys; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def test_missing_model_is_reported_without_implicit_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights"):
        ChessPieceDetector().load_model(str(tmp_path / "missing.pt"))
