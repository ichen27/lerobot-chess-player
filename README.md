# lerobot-chess-player

A robot arm that plays chess against you in the real world. Uses computer vision to see the board, Stockfish to think, and a LeRobot Pi0 policy to physically move the pieces.

**Human plays white. Robot plays black.**

## How It Works

```
Camera frame
    |
    v
YOLOv8 (piece detection) --> Board state (FEN)
    |                              |
    v                              v
Supervision (annotated view)   Stockfish (best move)
                                   |
                                   v
                          Pi0 policy (action chunks)
                                   |
                                   v
                          OMX robot arm (physical move)
```

1. **See** -- A YOLOv8 model detects all 12 chess piece types (king, queen, rook, bishop, knight, pawn x black/white) from the robot's camera. Board corners are detected via edge detection, and a perspective transform maps piece positions to algebraic squares (a1-h8).

2. **Think** -- The detected board state is converted to FEN notation and fed to Stockfish, which returns the best move for black.

3. **Move** -- The move is translated into a natural language instruction (e.g. "pick black knight from g8, place on f6") and sent to a fine-tuned [Pi0](https://github.com/huggingface/lerobot) vision-language-action model. Pi0 generates action chunks that are executed on the OMX follower robot arm at 30 fps.

## Hardware

- **Robot arm**: OMX follower (6-DOF, serial connection)
- **Camera**: Innomaker U20CAM 720P (mounted on arm, 1280x720)
- **Compute**: Any Mac with Apple Silicon (MPS) or CUDA GPU

## Setup

```bash
git clone https://github.com/ichen27/lerobot-chess-player.git
cd lerobot-chess-player
pip install -e .
brew install stockfish   # macOS
```

### Dependencies

- Python 3.10+
- [ultralytics](https://github.com/ultralytics/ultralytics) -- YOLOv8 inference and training
- [supervision](https://github.com/roboflow/supervision) -- detection annotation, NMS, and temporal smoothing
- [python-chess](https://github.com/niklasf/python-chess) -- board logic and Stockfish integration
- [lerobot](https://github.com/huggingface/lerobot) -- Pi0 policy and OMX robot arm control

## Usage

### Play a game (full pipeline)

```bash
python play_chess.py \
  --yolo-weights weights/chess_pieces/weights/best.pt \
  --pi0-model izchen/pi0_chess \
  --robot-port /dev/tty.usbmodem11401
```

Without a robot arm (print moves only):

```bash
python play_chess.py \
  --yolo-weights weights/chess_pieces/weights/best.pt
```

### Live detection viewer

Opens the camera and shows real-time piece detection with board grid overlay:

```bash
python scripts/test_yolo_live.py
```

Controls: `q` quit, `s` save frame, `+`/`-` adjust confidence, `t` toggle smoothing.

### Test Pi0 arm control

```bash
python scripts/test_pi0.py --instruction "pick white pawn from e2, place on e4"
```

### Train YOLO

```bash
# Download dataset (requires ROBOFLOW_API_KEY)
python scripts/download_dataset.py

# Train
python scripts/train_yolo.py
```

Trained weights are saved to `weights/chess_pieces/weights/best.pt`.

## Project Structure

```
chess_perception/
  detect.py      # YOLOv8 piece detector (12 classes)
  board.py       # Board corner detection + piece-to-square mapping
  engine.py      # Stockfish wrapper (FEN validation, best move)
  pipeline.py    # End-to-end: camera -> detection -> FEN -> move
  game.py        # Game orchestrator (human vs robot loop)
scripts/
  test_yolo_live.py    # Live camera viewer with supervision annotations
  test_pi0.py          # Standalone Pi0 + robot arm test
  test_pipeline.py     # Single-frame pipeline test
  train_yolo.py        # YOLO training script
  download_dataset.py  # Roboflow dataset downloader
weights/               # Trained model weights
data/                  # Training data
```

## License

MIT
