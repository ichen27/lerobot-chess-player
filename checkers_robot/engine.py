"""Deterministic depth-limited minimax with alpha-beta pruning."""

from __future__ import annotations
from checkers_robot.rules import Position


def best_move(position: Position, depth: int = 4):
    if type(depth) is not int or not 1 <= depth <= 8:
        raise ValueError("Search depth must be an integer from 1 to 8.")
    side = position.turn

    def search(board, remaining, alpha, beta):
        moves = board.legal_moves()
        if not moves:
            return -10000 - remaining if board.turn == side else 10000 + remaining
        if remaining == 0:
            score = 0
            for square, piece in board.pieces.items():
                row, _ = board.coords(square)
                advance = board.size - 1 - row if piece > 0 else row
                value = 175 if abs(piece) == 2 else 100 + advance
                score += value if piece * side > 0 else -value
            return score
        if board.turn == side:
            value = float("-inf")
            for move in moves:
                value = max(
                    value, search(board.apply(move), remaining - 1, alpha, beta)
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = float("inf")
        for move in moves:
            value = min(value, search(board.apply(move), remaining - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    chosen, score = None, float("-inf")
    for move in position.legal_moves():
        value = search(position.apply(move), depth - 1, float("-inf"), float("inf"))
        if value > score:
            chosen, score = move, value
    return chosen
