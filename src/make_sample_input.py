from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train import load_data


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "sample_input.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample JSON payload for the FastAPI /predict endpoint.")
    parser.add_argument(
        "--label",
        choices=["pass", "fail"],
        default="fail",
        help="Choose the first pass or fail sample from the SECOM dataset.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Use an explicit dataframe row index instead of selecting by label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_data()

    if args.row_index is not None:
        if args.row_index not in x.index:
            raise ValueError(f"row-index {args.row_index} is not present in the dataset.")
        selected_index = args.row_index
    else:
        target = 1 if args.label == "fail" else 0
        selected_index = int(y[y == target].index[0])

    row = x.loc[selected_index]
    sensors = [None if pd.isna(value) else float(value) for value in row.to_list()]
    payload = {
        "sample_index": selected_index,
        "source_label": "fail" if int(y.loc[selected_index]) == 1 else "pass",
        "sensors": sensors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved sample input: {args.output}")


if __name__ == "__main__":
    main()
