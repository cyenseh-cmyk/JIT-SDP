"""Validate and save the eligible dataset and its quality reports."""
from pathlib import Path

from jit_sdp.config import load
from jit_sdp.data import clean_dataset


if __name__ == "__main__":
    config = load(Path(__file__).resolve().parent / "configs/default.json")
    summary = clean_dataset(config)
    print(f"Saved {summary['clean_records']} eligible records and quality reports to "
          f"{Path(config['work_dir']) / 'cleaned'}")
