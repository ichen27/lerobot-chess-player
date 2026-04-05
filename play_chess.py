#!/usr/bin/env python3
"""CLI entry point for Robot vs Human chess.

Usage:
    python play_chess.py --yolo-weights weights/best.pt --stockfish stockfish
    python play_chess.py --yolo-weights weights/best.pt --pi0-model path/to/pi0
"""

from __future__ import annotations

import argparse
import sys

from chess_perception.game import ChessGame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robot vs Human chess game using perception + Stockfish + Pi0."
    )
    parser.add_argument(
        "--yolo-weights",
        required=True,
        help="Path to trained YOLO weights for piece detection.",
    )
    parser.add_argument(
        "--stockfish",
        default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' on PATH).",
    )
    parser.add_argument(
        "--pi0-model",
        default=None,
        help="Path to fine-tuned Pi0 model for robot arm control (optional).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )
    parser.add_argument(
        "--orientation",
        default="white_bottom",
        choices=["white_bottom", "black_bottom"],
        help="Board orientation in camera view (default: white_bottom).",
    )
    parser.add_argument(
        "--robot-port",
        default="/dev/tty.usbmodem11401",
        help="Serial port for the OMX follower arm (default: /dev/tty.usbmodem11401).",
    )

    args = parser.parse_args()

    game = ChessGame(
        yolo_weights=args.yolo_weights,
        stockfish_path=args.stockfish,
        pi0_model_path=args.pi0_model,
        camera_index=args.camera,
        board_orientation=args.orientation,
        robot_port=args.robot_port,
    )

    try:
        game.setup()
        game.play()
    except KeyboardInterrupt:
        print("\n[interrupted] Game aborted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[fatal] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
