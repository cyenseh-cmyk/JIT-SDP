"""Create the temporal splits from the saved cleaned dataset."""
from pathlib import Path

from jit_sdp.config import load
from jit_sdp.data import prepare_splits


if __name__ == "__main__":
    config = load(Path(__file__).resolve().parent / "configs/default.json")
    manifest = prepare_splits(config)
    print({name: partition["count"] for name, partition in manifest["partitions"].items()})
