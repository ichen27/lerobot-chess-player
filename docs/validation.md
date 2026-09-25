# Software validation

Checkers conversion, September 24, 2026.

## Recorded results

Fresh Python 3.12 environments on the Mac mini:
- Core install: **30 passed, 1 optional vision module skipped**.
- Core plus vision install: **48 passed**.
- Default 6×6 JSON demo and interactive CLI help: passed.
- Source distribution and wheel build: passed.
- Supplied screenshot: exact expected ten-square ownership map, 5 orange and 5 gray.

These are software results, not arm measurements.

## Reproduce

```bash
python -m pip install -e ".[dev,vision]"
python -m pytest -q
checkers-demo
checkers-demo --size 8
checkers-inspect docs/media/checkers-board.png \
  --calibration examples/screenshot-calibration.json
```

The software tests exercise:
- 6×6 and 8×8 setup, mandatory captures, complete multi-jumps, kings, promotion, illegal moves, and immutable state.
- Minimax legal replies and terminal positions.
- Capture removal, each jump leg, manual crowning, legal observation matching, and transaction failures.
- CLI play and a JSON demo without vision, chess, or neural-network packages.
- Perspective mapping, orientation, off-board points, invalid calibration, synthetic color detection, and image non-mutation.
- Exact detection of the ten discs in the owner-supplied, calibrated screenshot: five orange and five gray.

Core-only installations intentionally skip the optional vision test module. CI runs both core and vision installations.

The screenshot check is a fixture regression, not a measured accuracy rate across independent images. No robot or live camera was connected. Physical reliability, timing, gripper control, and closed-loop gameplay remain unverified. See [the hardware procedure](hardware.md).
