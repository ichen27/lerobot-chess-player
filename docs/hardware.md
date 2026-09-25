# Hardware integration

The repository currently runs software checkers, image inspection, and command planning. It does not connect to motors or load a LeRobot policy. The supplied arm footage documents an earlier physical prototype, not a hardware test of this revision.

## Adapter boundary

`plan_move(position, move)` emits typed commands:
- `remove`: remove a captured disc to a designated tray.
- `move`: pick a disc from a square and place it on another.
- `crown`: manually mark or stack a promoted piece.

A multi-jump produces each leg and each removal explicitly. Coordinates are board squares, not robot poses. An adapter must calibrate square centers, pickup heights, gripper settings, clearance paths, and the capture tray to the actual arm.

`Session.execute_turn(move, execute, observe=...)` calls an adapter for each command and compares a fresh observation to the expected final ownership map before updating game state. An alternative `confirm=...` callback must return exactly `True` after an operator verifies the move. No confirmation means no execution.

An exception or mismatch leaves software state unchanged, but **cannot roll back physical motion**. Stop and reconcile the physical board before resuming; do not automatically replay a partially completed move. Observation comparison cannot verify a crown. Promotion therefore requires a `confirm=...` callback even when `observe=...` is provided. When both callbacks are supplied, both must succeed before state advances.

## First hardware session

1. Record the actual arm model, controller, firmware, camera, and compatible control library. Fix the board and camera in place.
2. Calibrate the selected 6×6 region and HSV ranges at the camera's operating resolution. Verify every occupied square on multiple saved frames.
3. Establish square-to-arm coordinates and a capture-tray pose. Test unloaded paths and a single pickup/place before playing.
4. Implement the execution callback with explicit errors and completion reporting. Keep manual crowning in the loop.
5. Initialize a known position. Test a normal move, a capture, a complete multi-jump, and promotion. Capture before/after images for each case.
6. Exercise failure recovery: failed grip, interrupted command, displaced disc, incorrect camera reading. Verify state does not advance and reconcile the board manually.
7. Only then connect the camera, legal-move matching, engine, and execution adapter into a full game loop. Record outcomes and limitations in the validation log.

No arm, camera stream, manipulation success rate, or end-to-end gameplay was tested for this revision.
