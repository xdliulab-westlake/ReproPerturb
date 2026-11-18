"""Score a single direct-head test prediction against labeled ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from direct_metrics import score


def table(path, classes=None):
    if path.suffix == ".csv":
        frame = pd.read_csv(path).set_index("sample_id")
    else:
        with np.load(path) as data:
            frame = pd.DataFrame(data["proportions"], index=data["sample_ids"], columns=data["classes"] if "classes" in data else classes)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"Duplicate condition or class in {path}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prediction", "truth", "split"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text())
    ids, classes = manifest["splits"]["test"], manifest["classes"]
    pred, truth = table(args.prediction), table(args.truth, classes)
    if len(ids) != 14 or len(classes) != 10 or set(pred.index) != set(ids) or set(pred.columns) != set(classes):
        raise ValueError("Expected the 14 test conditions and 10 annotated classes")
    print(json.dumps(score(pred.loc[ids, classes], truth.loc[ids, classes]), indent=2))
