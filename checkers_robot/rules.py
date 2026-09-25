"""English-style checkers movement on the 6x6 prototype or an 8x8 board.

Orange moves toward the top (higher ranks); gray moves toward the bottom.
Men are +/-1; kings are +/-2. Board coordinates are explicit, not FEN.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import re

ORANGE, GRAY = 1, -1


@dataclass(frozen=True)
class Move:
    path: tuple[str, ...]
    captures: tuple[str, ...] = ()

    def __str__(self):
        return ("x" if self.captures else "-").join(self.path)


@dataclass(frozen=True)
class Position:
    pieces: Mapping[str, int]
    size: int = 6
    turn: int = ORANGE
    dark_parity: int = 0

    def __post_init__(self):
        if self.size not in (6, 8) or self.turn not in (ORANGE, GRAY):
            raise ValueError("Use a 6 or 8 square board and turn 1 or -1.")
        if self.dark_parity not in (0, 1):
            raise ValueError("dark_parity must be 0 or 1.")
        copied = dict(self.pieces)
        for square, piece in copied.items():
            row, col = self.coords(square)
            if (
                (row + col) % 2 != self.dark_parity
                or type(piece) is not int
                or piece not in (-2, -1, 1, 2)
            ):
                raise ValueError(
                    f"Invalid piece or unplayable square: {square}={piece}"
                )
        object.__setattr__(self, "pieces", MappingProxyType(copied))

    @classmethod
    def initial(cls, size=6, dark_parity=None):
        parity = (0 if size == 6 else 1) if dark_parity is None else dark_parity
        blank = cls({}, size=size, dark_parity=parity)
        pieces = {}
        rows = size // 2 - 1
        for row in range(size):
            for col in range(size):
                if (row + col) % 2 == parity:
                    if row < rows:
                        pieces[blank.square(row, col)] = GRAY
                    elif row >= size - rows:
                        pieces[blank.square(row, col)] = ORANGE
        return cls(pieces, size=size, dark_parity=parity)

    def coords(self, square):
        if (
            not isinstance(square, str)
            or len(square) != 2
            or square[0] not in "abcdefgh"[: self.size]
            or square[1] not in "12345678"[: self.size]
        ):
            raise ValueError(f"Invalid square: {square}")
        return self.size - int(square[1]), ord(square[0]) - ord("a")

    def square(self, row, col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("Square outside board.")
        return f"{chr(97 + col)}{self.size - row}"

    def _directions(self, piece):
        rows = (-1, 1) if abs(piece) == 2 else (-1,) if piece > 0 else (1,)
        return [(dr, dc) for dr in rows for dc in (-1, 1)]

    def _crowns(self, piece, square):
        row, _ = self.coords(square)
        return abs(piece) == 1 and row == (0 if piece > 0 else self.size - 1)

    def legal_moves(self):
        captures = []
        quiet = []

        def jumps(pieces, piece, path, taken):
            square = path[-1]
            row, col = self.coords(square)
            extended = False
            for dr, dc in self._directions(piece):
                nr, nc = row + 2 * dr, col + 2 * dc
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                mid, dst = self.square(row + dr, col + dc), self.square(nr, nc)
                victim = pieces.get(mid, 0)
                if victim * piece >= 0 or dst in pieces:
                    continue
                extended = True
                after = dict(pieces)
                del after[square]
                del after[mid]
                after[dst] = piece
                new_path, new_taken = path + (dst,), taken + (mid,)
                if self._crowns(piece, dst):
                    captures.append(Move(new_path, new_taken))
                else:
                    jumps(after, piece, new_path, new_taken)
            if taken and not extended:
                captures.append(Move(path, taken))

        for square, piece in sorted(self.pieces.items()):
            if piece * self.turn <= 0:
                continue
            jumps(self.pieces, piece, (square,), ())
            row, col = self.coords(square)
            for dr, dc in self._directions(piece):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    dst = self.square(nr, nc)
                    if dst not in self.pieces:
                        quiet.append(Move((square, dst)))
        return tuple(sorted(captures or quiet, key=str))

    def parse_move(self, text):
        path = tuple(re.split("[-x]", text.strip().lower()))
        for move in self.legal_moves():
            if move.path == path:
                return move
        raise ValueError(f"Illegal move: {text}")

    def apply(self, move):
        if move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}")
        pieces = dict(self.pieces)
        piece = pieces.pop(move.path[0])
        for captured in move.captures:
            del pieces[captured]
        if self._crowns(piece, move.path[-1]):
            piece *= 2
        pieces[move.path[-1]] = piece
        return Position(pieces, self.size, -self.turn, self.dark_parity)

    def ownership(self):
        return {s: (ORANGE if p > 0 else GRAY) for s, p in self.pieces.items()}

    def winner(self):
        return -self.turn if not self.legal_moves() else None

    def render(self):
        symbols = {1: "o", -1: "g", 2: "O", -2: "G", 0: "."}
        lines = [
            f"{self.size - r} "
            + " ".join(
                symbols[self.pieces.get(self.square(r, c), 0)] for c in range(self.size)
            )
            for r in range(self.size)
        ]
        return "\n".join(lines + ["  " + " ".join("abcdefgh"[: self.size])])
