"""Hardware-free example: legal move recovery and manipulation planning."""
from __future__ import annotations
import argparse
import json
import chess
from chess_perception.engine import ChessEngine
from chess_perception.game import ChessGame

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="Position before the human move.")
    parser.add_argument("--move", default="e2e4", help="Human move in UCI notation.")
    parser.add_argument("--reply", default="e7e5", help="Deterministic reply; ignored with --stockfish.")
    parser.add_argument("--stockfish", help="Optional Stockfish executable to choose the reply.")
    args = parser.parse_args()
    try:
        game = ChessGame("unused.pt")
        game.board = chess.Board(args.fen)
        if not game.board.is_valid():
            raise ValueError("Invalid starting position")
        human = chess.Move.from_uci(args.move)
        if human not in game.board.legal_moves:
            raise ValueError(f"Illegal human move: {args.move}")
        before = game.board.fen()
        game.board.push(human)
        recovered = game.detect_human_move(before, game.board.fen())
        reply = args.reply
        if args.stockfish:
            color = "white" if game.board.turn else "black"
            reply = ChessEngine(args.stockfish).get_best_move(game.board.fen(), color=color)
        if reply is None:
            raise ValueError("No reply: the game has ended")
        move = chess.Move.from_uci(reply)
        if move not in game.board.legal_moves:
            raise ValueError(f"Illegal reply: {reply}")
        commands = game._build_move_commands(move)
        game.board.push(move)
        print(json.dumps({
            "mode": "software-only; no camera, model or arm",
            "before": before, "human_move": recovered, "reply": reply,
            "commands": commands, "after": game.board.fen(),
        }, indent=2))
    except (ValueError, OSError, chess.engine.EngineError) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
