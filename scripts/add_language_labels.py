#!/usr/bin/env python3
"""Add per-episode language labels to the chess_pick_place dataset for Pi0 fine-tuning.

This script:
1. Maps each episode (0-50) to a natural language instruction
2. Updates tasks.jsonl with all unique task descriptions
3. Updates episodes.jsonl with per-episode task assignments
4. Updates info.json with the new total_tasks count
5. Adds a 'language_instruction' column to each parquet file
6. Re-uploads the dataset to HuggingFace Hub
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

# Dataset root
DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot/ivanchen/chess_pick_place"
META_DIR = DATASET_ROOT / "meta"
DATA_DIR = DATASET_ROOT / "data/chunk-000"

# Episode-to-move mapping based on the recording list the user followed.
EPISODE_LABELS = {
    # Pawns (0-11)
    0: "pick white pawn from e2, place on e4",
    1: "pick white pawn from d2, place on d4",
    2: "pick white pawn from a2, place on a4",
    3: "pick white pawn from h2, place on h4",
    4: "pick white pawn from b2, place on b3",
    5: "pick white pawn from g2, place on g3",
    6: "pick white pawn from c2, place on c4",
    7: "pick white pawn from f2, place on f4",
    8: "pick black pawn from e7, place on e5",
    9: "pick black pawn from d7, place on d5",
    10: "pick black pawn from a7, place on a5",
    11: "pick black pawn from h7, place on h5",
    # Pieces (12-25)
    12: "pick white knight from b1, place on c3",
    13: "pick white knight from g1, place on f3",
    14: "pick black knight from b8, place on c6",
    15: "pick black knight from g8, place on f6",
    16: "pick white bishop from c1, place on f4",
    17: "pick white bishop from f1, place on c4",
    18: "pick black bishop from c8, place on f5",
    19: "pick black bishop from f8, place on c5",
    20: "pick white queen from d1, place on d3",
    21: "pick black queen from d8, place on d6",
    22: "pick white rook from a1, place on a3",
    23: "pick white rook from h1, place on h3",
    24: "pick black rook from a8, place on a6",
    25: "pick black rook from h8, place on h6",
    # Center moves (26-35)
    26: "pick piece from e4, place on d5",
    27: "pick piece from d4, place on e5",
    28: "pick piece from c3, place on d5",
    29: "pick piece from f3, place on e5",
    30: "pick piece from c4, place on f7",
    31: "pick piece from f4, place on d2",
    32: "pick piece from e4, place on e5",
    33: "pick piece from d5, place on d4",
    34: "pick piece from c6, place on e5",
    35: "pick piece from f6, place on d5",
    # Long moves (36-42)
    36: "pick piece from a1, place on a8",
    37: "pick piece from h1, place on h8",
    38: "pick piece from a1, place on h1",
    39: "pick piece from a8, place on h8",
    40: "pick piece from a1, place on h8",
    41: "pick piece from h1, place on a8",
    42: "pick piece from e1, place on e8",
    # Captures (43-49)
    43: "remove piece from e5, pick piece and place on e5",
    44: "remove piece from d5, pick piece and place on d5",
    45: "remove piece from c4, pick piece and place on c4",
    46: "remove piece from f4, pick piece and place on f4",
    47: "remove piece from a5, pick piece and place on a5",
    48: "remove piece from h5, pick piece and place on h5",
    49: "remove piece from d7, pick piece and place on d7",
    # Extra episode
    50: "pick and place chess piece",
}


def main():
    print("Adding language labels to chess_pick_place dataset...")

    # 1. Build unique task list
    unique_tasks = sorted(set(EPISODE_LABELS.values()))
    task_to_index = {task: i for i, task in enumerate(unique_tasks)}
    print(f"  {len(unique_tasks)} unique task descriptions")

    # 2. Write tasks.jsonl
    tasks_path = META_DIR / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for i, task in enumerate(unique_tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")
    print(f"  Written {tasks_path}")

    # 3. Update episodes.jsonl
    episodes_path = META_DIR / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    with open(episodes_path, "w") as f:
        for ep in episodes:
            idx = ep["episode_index"]
            label = EPISODE_LABELS.get(idx, "pick and place chess piece")
            ep["tasks"] = [label]
            f.write(json.dumps(ep) + "\n")
    print(f"  Updated {episodes_path}")

    # 4. Update info.json
    info_path = META_DIR / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_tasks"] = len(unique_tasks)

    # Add language_instruction feature if not present
    if "language_instruction" not in info["features"]:
        info["features"]["language_instruction"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"  Updated {info_path}")

    # 5. Add language_instruction column to each parquet file
    for ep_idx in range(len(episodes)):
        label = EPISODE_LABELS.get(ep_idx, "pick and place chess piece")
        parquet_path = DATA_DIR / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping")
            continue

        table = pq.read_table(parquet_path)
        n_rows = table.num_rows

        # Add or replace language_instruction column
        if "language_instruction" in table.column_names:
            table = table.drop("language_instruction")

        lang_col = pa.array([label] * n_rows, type=pa.string())
        table = table.append_column("language_instruction", lang_col)

        # Also update task_index column to match new mapping
        new_task_idx = task_to_index[label]
        if "task_index" in table.column_names:
            table = table.drop("task_index")
        task_col = pa.array([new_task_idx] * n_rows, type=pa.int64())
        table = table.append_column("task_index", task_col)

        pq.write_table(table, parquet_path)

    print(f"  Updated {len(episodes)} parquet files with language_instruction column")

    print("\nDone! Dataset now has per-episode language labels.")
    print(f"To re-upload: huggingface-cli upload izchen/chess_pick_place {DATASET_ROOT} --repo-type dataset")


if __name__ == "__main__":
    main()
