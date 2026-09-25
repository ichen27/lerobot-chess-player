# Hardware integration and validation

## Current boundary

No robot or camera was connected during the September 2026 software revision. Passing software tests does not establish grasp reliability, calibration quality, motor limits, inference latency or autonomous chess gameplay.

The game adapter targets the ROBOTIS fork's LeRobot 0.3.4 interface. Its local reference exposes six action fields:

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

These names and their order must match the actual follower configuration and the training dataset. Do not infer compatibility merely from the name OMX. A newer LeRobot install is explicitly rejected by this adapter because preprocessing and checkpoint formats differ.

The optional `vision` dependencies are sufficient for camera perception, not arm control. A compatible ROBOTIS LeRobot environment, calibrated hardware, and a compatible Pi0 checkpoint must be supplied separately. There is no verified one-command hardware installation yet.

## Existing model compatibility issue

The original research scripts reference `izchen/pi0_chess` and contain a compatibility shim for a checkpoint trained with a newer LeRobot version. That shim deletes configuration fields and modifies cached files. It has not been adopted into the game runner as a verified conversion.

Use a checkpoint exported for the installed policy version, with validated input/output normalization statistics. A failed load stops the game; it no longer falls back silently to manual mode. The historical `scripts/test_pi0.py` utility remains research code and can connect to motors. It is not part of the automated test suite.

## What the adapter now enforces

- A model and connected robot must exist before execution.
- A new instruction resets queued policy actions.
- Each inference step reads real joint feedback; missing/non-finite fields fail.
- OpenCV BGR images are converted to RGB; language is passed as a batch of one.
- LeRobot 0.3.4 `select_action` is treated as one `(1, 6)` action, not a full trajectory.
- Unexpected shapes or non-finite outputs stop execution.
- The loop targets 30 Hz for at most 300 steps per instruction. This is a step budget, not a measured throughput guarantee or completion detector.
- Any execution exception stops the game. A partial physical move is not automatically retried.
- A matching camera observation is required before committing the robot move to game history.
- Promotion stops before arm motion because replacement of a pawn has not been validated.

These software guards do not implement collision avoidance or calibrated joint/velocity limits. Those must be checked against the real setup.

## Validation when the arm is available

Record the software commit, follower model, motor configuration, calibration identifier, checkpoint revision, normalization files, camera pose, board orientation, piece geometry and compute device.

1. **Perception only:** save representative chessboard images. Check board corners, both orientations, all piece classes, occlusion, and lighting. Checkers footage cannot validate a chess detector.
2. **Coordinate and camera agreement:** confirm that training and inference use the same camera feature, image convention, joint order and action units. An arm-mounted camera changes the board viewpoint after motion.
3. **Single action:** validate the installed model and processors in the hardware-specific environment with an operator present. Confirm motor limits and a reliable way to stop motion.
4. **Single pick-and-place:** test a known source and destination before a game. Record attempts, successes, misses and timings instead of inferring a rate from one clip.
5. **Closed-loop move:** verify that the final observed piece placement matches the planned position and that a mismatch halts without advancing state.
6. **Special moves:** test capture disposal and both castling moves separately. A capture-removal instruction has no validated disposal-location convention yet. Keep promotion manual.
7. **Short game:** start from a known position, exercise a human move and robot reply, and retain video plus logs identifying the exact revision.

Only after these checks should the README describe this revision as demonstrated on hardware. Do not reuse the course checkers video as that evidence.
