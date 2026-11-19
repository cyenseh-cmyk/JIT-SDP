from copy import deepcopy
from pathlib import Path
import re
from .io import read_json

MODELS = ["deepjit", "cc2vec", "jitline_apachejit"]
DEFAULT = {
    "archive": "apachejit_dataset_replication.zip", "work_dir": "data",
    "run_dir": "runs", "run_id": "experiment_v2", "baseline_dir": "baseline implementations",
    "models": MODELS, "seeds": [11, 23, 37], "projects": [],
    "repositories": {}, "validation_fraction": 0.2, "temporal_overlap": "error",
    "threshold_grid_step": 0.01,
    "encoding": {"msg_length": 256, "code_lines": 10, "code_length": 512,
                 "cc_files": 2, "cc_lines": 10, "cc_length": 64,
                 "max_msg_vocab": 20000, "max_code_vocab": 50000},
    "neural": {"embedding_dim": 64, "num_filters": 64, "filter_sizes": [1, 2, 3],
               "hidden_units": 512, "dropout": 0.5, "learning_rate": 0.0001,
               "weight_decay": 0.00001, "batch_size": 32, "max_epochs": 20,
               "patience": 5, "min_delta": 0.0001, "device": "cpu", "threads": 4},
    "deepjit": {"cuda_backward": "torch", "early_stopping": True},
    "cc2vec": {"embed_size": 64, "hidden_size": 32, "batch_size": 8,
               "max_epochs": 20, "patience": 5},
    "jitline": {"trees": 300, "max_features": "sqrt", "n_jobs": 4,
                "max_tokens": 50000, "min_df": 3, "candidate_evaluations": 30,
                "population": 10, "smote_max_neighbors": 20},
    "stress_tests": True,
    "analysis": {"bootstrap_draws": 1000, "permutation_draws": 9999, "seed": 2026}
}


def merge(base, updates):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(base[key], dict) and key != "repositories":
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load(path):
    path = Path(path).resolve()
    cfg = merge(deepcopy(DEFAULT), read_json(path))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["run_id"]):
        raise ValueError("run_id must be a simple directory name")
    if cfg["deepjit"]["cuda_backward"] not in ("torch", "triton"):
        raise ValueError("Unknown DeepJIT CUDA backward implementation")
    if type(cfg["deepjit"]["early_stopping"]) is not bool:
        raise ValueError("DeepJIT early_stopping must be boolean")
    seeds = cfg["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty, unique integers")
    models = cfg["models"]
    if (not isinstance(models, list) or not models or any(model not in MODELS for model in models)
            or len(models) != len(set(models))):
        raise ValueError("models must be a nonempty list of unique supported model names")
    if not 0 < cfg["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if cfg["temporal_overlap"] not in ("error", "exclude_late_training"):
        raise ValueError("Unknown temporal_overlap policy")
    if cfg["cc2vec"]["embed_size"] != 2 * cfg["cc2vec"]["hidden_size"]:
        raise ValueError("Upstream CC2Vec requires embed_size = 2 * hidden_size")
    if not 0 < cfg["threshold_grid_step"] < 0.5:
        raise ValueError("Invalid threshold grid")
    for group, keys in (("encoding", list(cfg["encoding"])),
                        ("neural", ["batch_size", "max_epochs", "patience", "threads"]),
                        ("cc2vec", ["batch_size", "max_epochs", "patience"]),
                        ("jitline", ["trees", "population", "smote_max_neighbors"])):
        if any(type(cfg[group][k]) is not int or cfg[group][k] < 1 for k in keys):
            raise ValueError(f"{group} sizes must be positive integers")
    if not 1 <= cfg["jitline"]["candidate_evaluations"] <= 30:
        raise ValueError("JITLine permits 1 to 30 candidate evaluations")
    kernels = cfg["neural"]["filter_sizes"]
    if not kernels or min(kernels) < 1 or max(kernels) > min(
            cfg["encoding"][k] for k in ("msg_length", "code_lines", "code_length")):
        raise ValueError("CNN kernels exceed encoded dimensions")
    # Configuration paths are relative to the config's parent project directory.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    for key in ("archive", "work_dir", "run_dir", "baseline_dir"):
        cfg[key] = str((root / cfg[key]).resolve())
    for project, repo in cfg["repositories"].items():
        cfg["repositories"][project] = str((root / repo).resolve())
    return cfg
