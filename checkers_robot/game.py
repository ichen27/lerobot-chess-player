"""Legal observation matching and confirmed manipulation transactions."""

from dataclasses import dataclass
from checkers_robot.rules import Position, Move


@dataclass(frozen=True)
class Command:
    kind: str
    source: str
    target: str | None = None

    def __str__(self):
        if self.kind == "remove":
            return f"remove captured piece from {self.source} to the capture tray"
        if self.kind == "crown":
            return f"manually crown the piece on {self.source}"
        return f"pick piece from {self.source}, place on {self.target}"


def plan_move(position: Position, move: Move):
    if move not in position.legal_moves():
        raise ValueError("Cannot plan an illegal move.")
    commands = []
    for i, (source, target) in enumerate(zip(move.path, move.path[1:])):
        if move.captures:
            commands.append(Command("remove", move.captures[i]))
        commands.append(Command("move", source, target))
    if position._crowns(position.pieces[move.path[0]], move.path[-1]):
        commands.append(Command("crown", move.path[-1]))
    return tuple(commands)


def match_move(position: Position, observed):
    """Vision observes ownership only; retain kings and turn from legal history."""
    if any(type(v) is not int or v not in (-1, 1) for v in observed.values()):
        return None
    matches = [
        m for m in position.legal_moves() if position.apply(m).ownership() == observed
    ]
    # Different jump routes can end in the same layout; require an explicit route.
    return matches[0] if len(matches) == 1 else None


class Session:
    def __init__(self, position=None):
        self.position = position if position is not None else Position.initial()

    def execute_turn(self, move, execute, *, observe=None, confirm=None):
        if observe is None and confirm is None:
            raise ValueError("An observation or explicit confirmation is required.")
        expected = self.position.apply(move)
        commands = plan_move(self.position, move)
        if any(command.kind == "crown" for command in commands) and confirm is None:
            raise ValueError("Crowning requires explicit manual confirmation.")
        for command in commands:
            execute(command)
        if observe is not None:
            if observe() != expected.ownership():
                raise RuntimeError(
                    "Post-move verification failed; inspect the physical board before restarting."
                )
        if confirm is not None and confirm() is not True:
            raise RuntimeError("Move was not confirmed; game state is unchanged.")
        self.position = expected
