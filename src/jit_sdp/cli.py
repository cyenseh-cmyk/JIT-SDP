import argparse
import sys
from pathlib import Path
from .config import load


def main(argv=None):
    parser = argparse.ArgumentParser(description="ApacheJIT preparation, training and comparison")
    parser.add_argument("--config", default="configs/default.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "acquire", "prepare", "train", "compare", "run", "doctor"):
        sub.add_parser(command)
    enrich = sub.add_parser("enrich")
    enrich.add_argument("--limit", type=int)
    enrich.add_argument("--retry-errors", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--run-folder", required=True)
    predict.add_argument("--input", required=True, help="Canonical enriched JSONL")
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load(args.config)
    try:
        if args.command == "audit":
            from .data import audit_archive
            tables = audit_archive(cfg)
            print({name: len(rows) for name, rows in tables.items()})
        elif args.command == "acquire":
            from .gitdata import acquire
            acquire(cfg)
        elif args.command == "enrich":
            from .gitdata import enrich
            print(enrich(cfg, args.limit, args.retry_errors))
        elif args.command == "prepare":
            from .data import prepare
            manifest = prepare(cfg)
            print({name: p["count"] for name, p in manifest["partitions"].items()})
        elif args.command in ("train", "run"):
            if args.command == "run":
                from .gitdata import enrich
                from .data import prepare
                enrich(cfg)
                prepare(cfg)
            from .runner import run
            run(cfg)
            print("Completed:", Path(cfg["run_dir"]) / cfg["run_id"] / "comparison.md")
        elif args.command == "compare":
            from .analysis import analyze
            analyze(Path(cfg["run_dir"]) / cfg["run_id"], cfg["analysis"])
        elif args.command == "doctor":
            from .runner import provenance
            import torch
            print(provenance(cfg))
            print("CUDA available:", torch.cuda.is_available())
        elif args.command == "predict":
            from .io import read_jsonl
            from .runner import load_pipeline, prediction_rows, write_predictions
            pipeline, threshold = load_pipeline(args.run_folder)
            records = list(read_jsonl(args.input))
            folder = Path(args.run_folder)
            write_predictions(args.output, prediction_rows(pipeline.cfg, pipeline, records,
                folder.parent.name, int(folder.name.removeprefix("seed_")), threshold, stress=False))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
