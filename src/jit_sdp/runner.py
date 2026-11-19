"""Sequential, reproducible experiment execution and reloadable prediction."""
import csv
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path
from .data import key, load_partitions
from .features import JITLineAdapter, NeuralAdapter
from .io import digest, file_hash, read_json, write_json
from .metrics import classification, select_threshold
from .stress import CONDITIONS, transform


def provenance(cfg):
    revisions = {}
    for name in ("DeepJIT", "CC2Vec", "JITLine-replication-package"):
        path = Path(cfg["baseline_dir"]) / name
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Missing baseline Git checkout: {path}")
        revisions[name] = {"revision": result.stdout.strip(), "files": {
            str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*.py"))}}
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "imbalanced-learn", "torch", "joblib"):
        packages[name] = importlib.metadata.version(name)
    if cfg.get("deepjit", {}).get("cuda_backward") == "triton" and "deepjit" in cfg["models"] and cfg["neural"]["device"].startswith("cuda"):
        packages["triton"] = importlib.metadata.version("triton")
    return {"upstream": revisions, "python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "package_source": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def write_predictions(path, rows):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    if not rows:
        raise ValueError("No predictions")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def prediction_rows(cfg, pipeline, records, model, seed, threshold, stress=True):
    probabilities = pipeline.predict(records)
    # Validate original scores before any output is saved.
    classification([r["buggy"] for r in records], probabilities, threshold)
    rows = []
    adapter = pipeline.adapter
    for record, probability in zip(records, probabilities):
        original_hash = adapter.input_hash(record, model)
        truncated = adapter.encode(record, adapter.input_names(model))["truncated"] if model != "jitline_apachejit" else False
        base = {"run_id": cfg["run_id"], "model": model, "seed": seed, "project": record["project"],
            "commit_id": record["commit_id"], "partition": "test", "y_original": int(record["buggy"]),
            "condition": "original", "status": "original", "status_reason": "", "p_original": probability,
            "p_transformed": probability, "yhat_original": int(probability >= threshold),
            "yhat_transformed": int(probability >= threshold), "threshold": threshold,
            "input_hash_original": original_hash, "input_hash_transformed": original_hash,
            "num_hunks_changed": 0, "truncation_flag": truncated, "metric_scope_ok": True,
            "t1_t2_same_encoding": ""}
        rows.append(base)
    if not stress:
        return rows
    originals = list(rows)
    condition_rows = {}
    for condition in CONDITIONS:
        transformed, indices, output = [], [], []
        for index, (record, base) in enumerate(zip(records, originals)):
            changed, reason, hunks = transform(record, condition, model)
            row = {**base, "condition": condition, "num_hunks_changed": hunks}
            if reason:
                row.update(status="structural_na", status_reason=reason, p_transformed=None,
                           yhat_transformed=None, input_hash_transformed=None,
                           metric_scope_ok=reason != "metric_scope_mismatch")
            else:
                new_hash = adapter.input_hash(changed, model)
                row.update(status="encoding_noop" if new_hash == base["input_hash_original"] else "effective_change",
                           input_hash_transformed=new_hash)
                if model != "jitline_apachejit":
                    row["truncation_flag"] |= adapter.encode(changed, adapter.input_names(model))["truncated"]
                transformed.append(changed)
                indices.append(index)
            output.append(row)
        if transformed:
            scores = pipeline.predict(transformed)
            classification([r["buggy"] for r in transformed], scores, threshold)
            for index, score in zip(indices, scores):
                output[index].update(p_transformed=score, yhat_transformed=int(score >= threshold))
        condition_rows[condition] = output
        rows.extend(output)
    for a, b in zip(condition_rows["T1"], condition_rows["T2"]):
        if a["input_hash_transformed"] and b["input_hash_transformed"]:
            same = a["input_hash_transformed"] == b["input_hash_transformed"]
            a["t1_t2_same_encoding"] = b["t1_t2_same_encoding"] = same
    return rows


def run(cfg):
    print("Loading experiment dependencies...", flush=True)
    import torch
    from .neural import NeuralPipeline
    from .forest import ForestPipeline, tune
    print("Loading and verifying dataset splits...", flush=True)
    groups, manifest = load_partitions(cfg)
    counts = ", ".join(f"{name}={len(records):,}" for name, records in groups.items())
    print(f"Verified splits ({counts})", flush=True)
    train, validation, test = (groups[n] for n in ("train", "validation", "test"))
    root = Path(cfg["run_dir"]) / cfg["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    sources = provenance(cfg)
    identity = {"config": cfg, "split_hash": manifest["hash"], "sources": sources}
    if (root / "experiment.json").exists() and read_json(root / "experiment.json") != identity:
        raise ValueError("Run identity changed; choose a new run_id")
    write_json(root / "experiment.json", identity)
    write_json(root / "hardware.json", {"cpu": platform.processor(), "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": True, "device": cfg["neural"]["device"], "parallel_training_runs": 1})
    print("Building neural vocabulary from the training split...", flush=True)
    neural_adapter = NeuralAdapter(cfg["encoding"]).fit(train)
    forest_adapter, selected = None, None
    if "jitline_apachejit" in cfg["models"]:
        print("Preparing JITLine features...", flush=True)
        forest_adapter = JITLineAdapter(cfg["jitline"]["max_tokens"], cfg["jitline"]["min_df"]).fit(train)
        selection_path = root / "jitline_selection.json"
        if selection_path.exists():
            selected = read_json(selection_path)["selected"]
        else:
            print("Tuning JITLine on the validation split...", flush=True)
            selected, candidates = tune(cfg, forest_adapter, train, validation, cfg["seeds"][0])
            write_json(selection_path, {"selected": selected, "candidates": candidates,
                                        "tuning_seed": cfg["seeds"][0]})
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            out = root / model / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            marker = out / "complete.json"
            if marker.exists():
                saved = read_json(marker)
                if saved["identity"] != digest(identity) or any(
                        not (out / filename).exists() or file_hash(out / filename) != checksum
                        for filename, checksum in saved["artifacts"].items()):
                    raise ValueError(f"Completed run was modified: {out}")
                print(f"Verified completed run: {model}/{seed}", flush=True)
                continue
            started = time.perf_counter()
            print(f"Starting {model}, seed {seed}...", flush=True)
            try:
                adapter = forest_adapter if model == "jitline_apachejit" else neural_adapter
                pipeline = (ForestPipeline(cfg, adapter, seed, selected) if model == "jitline_apachejit" else
                            NeuralPipeline(cfg, adapter, model, seed))
                if model != "jitline_apachejit":
                    write_json(out / "adapter.json", neural_adapter.state)
                pipeline.fit(train, validation, out)
                print(f"Evaluating {model}, seed {seed}...", flush=True)
                validation_scores = (pipeline.validation_scores if isinstance(pipeline, NeuralPipeline)
                                     else pipeline.predict(validation))
                threshold = select_threshold([r["buggy"] for r in validation], validation_scores,
                                             cfg["threshold_grid_step"])
                write_json(out / "threshold.json", {"threshold": threshold, "partition": "validation",
                    "metrics": classification([r["buggy"] for r in validation], validation_scores, threshold)})
                write_json(out / "history.json", pipeline.history)
                write_json(out / "encoding_audit.json", {
                    name: {"n": len(records), "truncated": sum(
                        adapter.encode(r, adapter.input_names(model))["truncated"] for r in records)
                           if model != "jitline_apachejit" else 0,
                           "empty_messages": sum(not r["message"] for r in records)}
                    for name, records in groups.items()})
                rows = prediction_rows(cfg, pipeline, test, model, seed, threshold, cfg["stress_tests"])
                write_predictions(out / "predictions.csv", rows)
                write_json(out / "runtime.json", {"seconds": time.perf_counter() - started,
                                                  "seed": seed, "model": model})
                artifacts = {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name not in ("complete.json", "failure.json")}
                write_json(marker, {"identity": digest(identity), "artifacts": artifacts})
                print(f"Completed {model}, seed {seed} in {time.perf_counter() - started:.1f}s",
                      flush=True)
            except Exception as exc:
                write_json(out / "failure.json", {"error": type(exc).__name__, "message": str(exc),
                                                  "seconds": time.perf_counter() - started})
                raise
    print("Comparing experiment results...", flush=True)
    from .analysis import analyze
    return analyze(root, cfg["analysis"])


def load_pipeline(run_folder):
    """Load trusted local artifacts only; joblib is not safe for untrusted files."""
    import joblib
    from .neural import NeuralPipeline
    from .forest import ForestPipeline
    out = Path(run_folder).resolve()
    experiment = read_json(out.parent.parent / "experiment.json")
    cfg = experiment["config"]
    model = out.parent.name
    seed = int(out.name.removeprefix("seed_"))
    if model == "jitline_apachejit":
        saved = joblib.load(out / "forest.joblib")
        pipeline = ForestPipeline(cfg, saved["adapter"], seed, {})
        pipeline.model = saved["model"]
    else:
        state = read_json(out / "adapter.json")
        pipeline = NeuralPipeline(cfg, NeuralAdapter(state["encoding"], state), model, seed).load(out)
    return pipeline, read_json(out / "threshold.json")["threshold"]
