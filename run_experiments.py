"""Train and compare the configured models and seeds."""
from pathlib import Path

from jit_sdp.config import load
from jit_sdp.runner import run


if __name__ == "__main__":
    config = load(Path(__file__).resolve().parent / "configs/default.json")
    run(config)
    print("Completed:", Path(config["run_dir"]) / config["run_id"] / "comparison.md")
