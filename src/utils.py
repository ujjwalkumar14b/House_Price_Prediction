import os
import json


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
