# Models, data and reproducibility

## Chess detector

The original training artifacts are retained in:

- `weights/chess_pieces/args.yaml`
- `weights/chess_pieces/results.csv`
- `data/chess_pieces/data.yaml`
- `data/chess_pieces/README.roboflow.txt`

The dataset descriptor identifies Roboflow-100's `chess-pieces-mjzgj`, version 2, and lists CC BY 4.0. Preserve the dataset's attribution when redistributing it.

Trained `.pt` files and raw dataset images are not tracked in Git. Supply a checkpoint locally with `--yolo-weights`. The detector checks that the file exists before loading it; it does not silently substitute a general-purpose YOLO model.

### Class-label mismatch

The historical dataset has **13 labels**, including the ambiguous label `bishop`. The chess serializer supports the 12 explicit black/white piece classes. An uncolored `bishop` cannot be safely interpreted, so it produces a board-mapping error instead of an invalid `?` in FEN.

Do not simply delete a label from the YAML: that would shift class IDs relative to the annotations and trained checkpoint. Correct/relabel the dataset and retrain, or use a verified checkpoint with the required 12 named classes.

The checked-in training metrics are historical logs, not a fresh evaluation of the physical setup. No new accuracy or gameplay success rate is claimed.

### Historical training utilities

```bash
python -m pip install -e ".[training]"
# Set ROBOFLOW_API_KEY in your shell; never commit it.
python scripts/download_dataset.py
python scripts/train_yolo.py
```

Review paths and class labels before training. These original research scripts are retained for provenance; the reliable reviewer entry point is `chess-demo`.

## Pi0

The original scripts reference `izchen/pi0_chess` and dataset `izchen/chess_pick_place`. Their current remote availability and suitability for this revision are not established here.

The game adapter requires a compatible LeRobot 0.3.4 checkpoint and normalization statistics. See [hardware.md](hardware.md) for the unverified compatibility and hardware boundaries. Model loading success alone does not demonstrate useful physical behavior.
