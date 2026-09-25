"""Run one checkers turn without a camera, model, or arm."""

import argparse
from dataclasses import asdict
import json
from checkers_robot.rules import Position
from checkers_robot.engine import best_move
from checkers_robot.game import match_move, plan_move


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(6, 8), default=6)
    parser.add_argument(
        "--move", help="Human move, for example a2-b3 on the 6x6 board."
    )
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    try:
        before = Position.initial(args.size)
        human = before.parse_move(args.move or ("a2-b3" if args.size == 6 else "a3-b4"))
        current = before.apply(human)
        recovered = match_move(before, current.ownership())
        reply = best_move(current, args.depth)
        commands = plan_move(current, reply) if reply else ()
        after = current.apply(reply) if reply else current
        print(
            json.dumps(
                {
                    "game": "checkers",
                    "mode": "software-only",
                    "board_size": args.size,
                    "human_move": str(recovered),
                    "ai_move": str(reply) if reply else None,
                    "commands": [asdict(c) for c in commands],
                    "instructions": [str(c) for c in commands],
                    "after": {
                        "pieces": dict(after.pieces),
                        "turn": after.turn,
                        "board": after.render(),
                    },
                },
                indent=2,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
