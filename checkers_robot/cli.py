"""Interactive software checkers; prints robot commands without operating hardware."""

import argparse
from checkers_robot.engine import best_move
from checkers_robot.game import Session
from checkers_robot.rules import Position, ORANGE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play checkers against minimax. No robot is connected."
    )
    parser.add_argument("--size", type=int, choices=(6, 8), default=6)
    parser.add_argument("--depth", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    session = Session(Position.initial(args.size))
    print("Software checkers: you are orange (o), AI is gray (g). O/G are kings.")
    print("Enter a move such as a2-b3 or a full jump path such as a2xc4xe6. q quits.")
    print("Robot commands are a preview only; no hardware is connected.")
    try:
        while session.position.winner() is None:
            print("\n" + session.position.render())
            if session.position.turn == ORANGE:
                text = input("Your move (or ? for legal moves): ").strip()
                if text.lower() in ("q", "quit"):
                    return 0
                if text == "?":
                    print(", ".join(map(str, session.position.legal_moves())))
                    continue
                try:
                    move = session.position.parse_move(text)
                except ValueError as exc:
                    print(exc)
                    continue
                session.position = session.position.apply(move)
            else:
                move = best_move(session.position, args.depth)
                print(f"AI: {move}")
                # Software-only confirmation. Hardware callers must supply actual verification.
                session.execute_turn(
                    move,
                    lambda command: print(f"  Preview: {command}"),
                    confirm=lambda: True,
                )
        print("\n" + session.position.render())
        print("Orange wins." if session.position.winner() == ORANGE else "Gray wins.")
    except (EOFError, KeyboardInterrupt):
        print("\nGame ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
