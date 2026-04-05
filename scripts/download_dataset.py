#!/usr/bin/env python3
"""Download the Chess Pieces dataset from Roboflow in YOLOv8 format."""

import os
import sys


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: set ROBOFLOW_API_KEY environment variable.")
        print("  Get a free key at https://app.roboflow.com/settings/api")
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
    version = project.version(2)

    dest = os.path.expanduser("~/chess/data/chess_pieces")
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading dataset to {dest} ...")
    dataset = version.download("yolov8", location=dest)
    print(f"Dataset downloaded: {dataset.location}")


if __name__ == "__main__":
    main()
