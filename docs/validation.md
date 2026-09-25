# Software validation — September 24, 2026

This report records software checks on the Mac mini. No camera or robot was connected.

| Check | Result |
| --- | --- |
| Full regression suite, Python 3.11 | 40 passed |
| Clean core environments (Python 3.11 and locked Python 3.12), no torch/ultralytics/LeRobot | 35 passed; 1 optional policy test module skipped |
| Scripted hardware-free demo | Legal e2e4/e7e5 sequence and pick/place instruction emitted |
| Real Stockfish demo | Stockfish 19 selected a legal reply through the engine wrapper |
| Wheel and source distribution build | Succeeded |
| CLI help in core environment | Succeeded without model or camera initialization |
| Local YOLO checkpoint smoke test | Loaded and inferred one white-bishop detection on one archived test image |
| Physical execution / live perception | Not run |

The detector smoke test used the locally available, untracked `weights/chess_pieces/weights/best.pt` on `0b47311f426ff926578c9d738d683e76_jpg.rf.999ad103e4382fd86b69052c2c35b46e.jpg`. This is a loading/inference check, not a new labeled accuracy evaluation. These assets are not included in a fresh clone.

## Regression coverage

- Same-square confidence ordering and both board orientations.
- Off-board coordinates, non-finite coordinates, degenerate quadrilaterals, and a point on the projective horizon outside a trapezoid.
- Rejection of an arbitrary triangle instead of inventing a board from its bounding box.
- Unsupported labels and invalid square identifiers.
- Exact legal move recovery, including capture, castling, en passant and promotion.
- Rejection of illegal moves, misidentified pieces, and malformed FEN.
- Correct side-to-move handling for checks, and observation without an engine dependency.
- Manual confirmation and physical post-move mismatch/failure without advancing history.
- Pi0 RGB conversion, batch language input, repeated feedback, action-queue reset, expected output shape and finite values.
- Lightweight imports, explicit missing-weight errors, and the installed software demo.

The policy contract tests use synthetic tensors and fake hardware boundaries. They validate our adapter behavior, not the real model's manipulation ability.

A separate read-only code review found the projective-horizon issue; a failing regression test reproduced it before the containment fix. The original software regressions were also reproduced before correction.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
chess-demo
# Optional full suite (downloads vision dependencies, not model weights):
python -m pip install -e ".[dev,vision]"
python -m pytest -q
# Optional real search:
chess-demo --stockfish /path/to/stockfish
```

GitHub Actions additionally defines core checks on Python 3.10/3.12 and a vision/policy-contract job on Python 3.11. Remote workflow status must be checked separately from these local results.
