# Board perception and calibration

Install the vision extra and run the README's `checkers-inspect` command. The command accepts an image file, not a live camera stream. It returns JSON ownership values: orange = 1, gray = -1. It can also save a rectified, annotated image.

## Calibration fields

See [the screenshot example](../examples/screenshot-calibration.json).

- `size`: 6 or 8.
- `corners`: the four outside corners of the chosen playing region in image pixels, ordered top-left, top-right, bottom-right, bottom-left. For the prototype, select the 6×6 region, not the entire physical board.
- `orientation`: `orange_bottom` or `gray_bottom`. The latter rotates the rectified image 180 degrees into canonical coordinates.
- `dark_parity`: playable squares satisfy `(row + column) % 2 == dark_parity`, with zero-based rows from the top. Default 0 for 6×6, 1 for 8×8.
- `colors`: measured lower/upper HSV values for orange and gray. OpenCV uses H 0–179, S and V 0–255.
- `min_fraction`: minimum fraction of the inner part of a square matching a color; the example uses 0.2.

The detector samples the inner 70% of each playable square's width and height. A square matching both colors is rejected. Off-board points, non-convex corners, out-of-frame calibration, and malformed ranges are rejected. Calibration is resolution-specific.

The supplied image is a screenshot with existing overlays. Its example color ranges were selected for that image and must not be reused blindly on a camera. Lighting, shadows, glare, occlusion, and off-center pieces can cause missed or false detections. The exact-layout legal-move check is a further consistency check, not a guarantee of visual correctness.

## Connecting observations to a game

```python
from checkers_robot.game import match_move
from checkers_robot.vision import load_calibration

detector = load_calibration("your-calibration.json")
observed, annotated = detector.process(frame)  # uint8 BGR image
move = match_move(position, observed)
if move is None:
    raise ValueError("Observation does not identify exactly one legal move")
position = position.apply(move)
```

Initialize the game from a known board with the correct turn and king status. The supplied screenshot is a mid-game fixture, not the initial position. A photo cannot recover turn or king history from disc colors. Multiple legal jump routes can have the same final layout; in that case, explicitly enter the route rather than guessing.

The API is available for integration, but this revision does not connect image capture to the interactive terminal game.
