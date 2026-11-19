"""Saved-prediction-only analysis; seeds never serve as independent projects."""
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .io import read_json, write_json
from .metrics import classification, sensitivity

VALID = {"encoding_noop", "effective_change"}


def read_predictions(root):
    rows = []
    for file in sorted(Path(root).glob("*/seed_*/predictions.csv")):
        if not (file.parent / "complete.json").exists():
            raise ValueError(f"Incomplete run: {file.parent}")
        with file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for name in ("p_original", "p_transformed", "threshold"):
                    row[name] = None if row[name] == "" else float(row[name])
                for name in ("seed", "y_original", "yhat_original", "yhat_transformed"):
                    row[name] = None if row[name] == "" else int(row[name])
                rows.append(row)
    if not rows:
        raise ValueError("No completed predictions")
    return rows


def mcc(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(set(y.tolist())) < 2:
        return None
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def holm(entries, pkey="p", outkey="p_holm"):
    valid = sorted((e for e in entries if e.get(pkey) is not None), key=lambda e: e[pkey])
    previous = 0.0
    for index, entry in enumerate(valid):
        previous = max(previous, min(1.0, entry[pkey] * (len(valid) - index)))
        entry[outkey] = previous


def interval(values, requested):
    return {"low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "retained": len(values), "undefined": requested - len(values),
            "unstable": len(values) < 0.9 * requested}


def paired_arrays(rows):
    if not rows:
        return None
    groups = defaultdict(dict)
    for row in rows:
        groups[row["seed"]][(row["project"], row["commit_id"])] = row
    common = sorted(set.intersection(*(set(g) for g in groups.values())))
    if not common:
        return None
    seeds = sorted(groups)
    y = np.array([groups[seeds[0]][k]["y_original"] for k in common])
    original = np.array([[groups[s][k]["yhat_original"] for k in common] for s in seeds])
    changed = np.array([[groups[s][k]["yhat_transformed"] for k in common] for s in seeds])
    return common, y, original, changed


def delta_stat(y, original, changed):
    values = []
    for a, b in zip(original, changed):
        first, second = mcc(y, a), mcc(y, b)
        if first is None or second is None:
            return None
        values.append(first - second)
    return float(np.mean(values))


def uncertainty(rows, settings, rng):
    arrays = paired_arrays(rows)
    if arrays is None:
        return {"delta_mcc": None, "p": None, "n": 0}
    keys, y, original, changed = arrays
    observed = delta_stat(y, original, changed)
    project_groups = defaultdict(list)
    for index, (project, _) in enumerate(keys):
        project_groups[project].append(index)
    groups = list(project_groups.values())
    values = []
    for _ in range(settings["bootstrap_draws"]):
        selected = np.concatenate([rng.choice(groups[i], len(groups[i]), replace=True)
                                   for i in rng.integers(len(groups), size=len(groups))])
        value = delta_stat(y[selected], original[:, selected], changed[:, selected])
        if value is not None:
            values.append(value)
    extreme = 0
    if observed is not None:
        for _ in range(settings["permutation_draws"]):
            # One paired swap per commit, shared across every training seed.
            swap = rng.random(len(y)) < 0.5
            permuted = delta_stat(y, np.where(swap, changed, original), np.where(swap, original, changed))
            extreme += abs(permuted) >= abs(observed) - 1e-12
    return {"n": len(y), "projects": len(groups), "delta_mcc": observed,
            "bootstrap_95": interval(values, settings["bootstrap_draws"]),
            "p": (extreme + 1) / (settings["permutation_draws"] + 1) if observed is not None else None}


def performance_bootstrap(rows, settings, rng):
    arrays = paired_arrays(rows)
    if arrays is None:
        return {"n": 0}
    keys, y, predictions, _ = arrays
    projects = defaultdict(list)
    for i, (project, _) in enumerate(keys):
        projects[project].append(i)
    groups = list(projects.values())
    values = []
    for _ in range(settings["bootstrap_draws"]):
        indices = np.concatenate([rng.choice(groups[i], len(groups[i]), replace=True)
                                  for i in rng.integers(len(groups), size=len(groups))])
        scores = [mcc(y[indices], p[indices]) for p in predictions]
        if all(s is not None for s in scores):
            values.append(float(np.mean(scores)))
    return {"n": len(y), "seed_mean_mcc_95": interval(values, settings["bootstrap_draws"])}


def paired_performance(rows, settings, rng):
    grouped = defaultdict(dict)
    for row in rows:
        if row["condition"] == "original":
            grouped[(row["model"], row["seed"])][(row["project"], row["commit_id"])] = row
    keys = sorted(set.intersection(*(set(v) for v in grouped.values())))
    models = sorted({m for m, _ in grouped})
    seeds = sorted({s for _, s in grouped})
    y = np.array([next(iter(grouped.values()))[k]["y_original"] for k in keys])
    predictions = {m: np.array([[grouped[(m, s)][k]["yhat_original"] for k in keys] for s in seeds]) for m in models}
    project_groups = defaultdict(list)
    for index, (project, _) in enumerate(keys):
        project_groups[project].append(index)
    groups = list(project_groups.values())
    pairs = list(itertools.combinations(models, 2))
    samples = {pair: [] for pair in pairs}

    def means(indices):
        result = {}
        for model in models:
            values = [mcc(y[indices], p[indices]) for p in predictions[model]]
            result[model] = float(np.mean(values)) if all(v is not None for v in values) else None
        return result

    observed = means(np.arange(len(y)))
    for _ in range(settings["bootstrap_draws"]):
        # The same resampled project/commit indices reach every method and seed.
        indices = np.concatenate([rng.choice(groups[i], len(groups[i]), replace=True)
                                  for i in rng.integers(len(groups), size=len(groups))])
        values = means(indices)
        for pair in pairs:
            if all(values[m] is not None for m in pair):
                samples[pair].append(values[pair[0]] - values[pair[1]])
    return [{"first": a, "second": b, "n": len(y),
             "seed_mean_mcc_difference": observed[a] - observed[b] if observed[a] is not None and observed[b] is not None else None,
             "paired_bootstrap_95": interval(samples[(a, b)], settings["bootstrap_draws"])} for a, b in pairs]


def friedman_model_pvalue(matrix):
    """Compare model rows using the same project columns as paired blocks."""
    from scipy.stats import friedmanchisquare

    if np.all(matrix == matrix[:1, :]):
        return None
    value = float(friedmanchisquare(*matrix).pvalue)
    return value if np.isfinite(value) else None


def analyze(root, settings):
    from scipy.stats import wilcoxon
    root = Path(root)
    rows = read_predictions(root)
    experiment = read_json(root / "experiment.json")
    expected_models = set(experiment["config"]["models"])
    expected_seeds = set(experiment["config"]["seeds"])
    found = {(r["model"], r["seed"]) for r in rows}
    if found != set(itertools.product(expected_models, expected_seeds)):
        raise ValueError("Comparison requires every configured model/seed to complete")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["seed"], row["condition"])].append(row)
    original_keys = []
    for (model, seed, condition), values in grouped.items():
        identities = [(r["project"], r["commit_id"]) for r in values]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate prediction keys")
        if condition == "original":
            original_keys.append(set(identities))
    if any(keys != original_keys[0] for keys in original_keys):
        raise ValueError("Original test IDs differ across methods/seeds")
    report = {"performance": [], "stress": [], "seed_summary": [], "matched_stress": [],
              "rq2": [], "rq3": [], "performance_uncertainty": {},
              "assumptions": ["Hierarchical project/commit bootstrap, paired across models and seeds.",
                              "Paired permutations assume within-commit exchangeability; commits and related projects may be dependent.",
                              "Counterfactual labels are unverified; delta MCC uses original labels.",
                              "Three training seeds describe variability, not independent datasets."]}
    rng = np.random.default_rng(settings["seed"])
    report["paired_performance"] = paired_performance(rows, settings, rng)
    stress_all = defaultdict(list)
    for (model, seed, condition), values in sorted(grouped.items()):
        if condition == "original":
            project_groups = {"pooled": values}
            for project in sorted({r["project"] for r in values}):
                project_groups[project] = [r for r in values if r["project"] == project]
            for project, selected in project_groups.items():
                report["performance"].append({"model": model, "seed": seed, "project": project,
                    **classification([r["y_original"] for r in selected], [r["p_original"] for r in selected], selected[0]["threshold"])})
        else:
            statuses = Counter(r["status"] for r in values)
            applicable = [r for r in values if r["status"] in VALID]
            effective = [r for r in values if r["status"] == "effective_change"]
            report["stress"].append({"model": model, "seed": seed, "condition": condition,
                "total": len(values), "statuses": dict(statuses),
                "applicability_coverage": len(applicable) / len(values),
                "encoding_noop_rate": statuses["encoding_noop"] / len(applicable) if applicable else None,
                "structurally_applicable": sensitivity(applicable), "effective_change": sensitivity(effective),
                "t1_t2_identical": sum(r["t1_t2_same_encoding"] == "True" for r in applicable)})
            stress_all[(model, condition)].extend(applicable)
    for model in sorted(expected_models):
        model_rows = [r for r in rows if r["model"] == model and r["condition"] == "original"]
        report["performance_uncertainty"][model] = performance_bootstrap(model_rows, settings, rng)
        for metric in ("mcc", "ap", "precision", "recall", "f1"):
            scores = [r[metric] for r in report["performance"] if r["model"] == model and r["project"] == "pooled" and r[metric] is not None]
            project_means = []
            for seed in sorted(expected_seeds):
                projects = [r[metric] for r in report["performance"] if r["model"] == model and r["seed"] == seed
                            and r["project"] != "pooled" and r[metric] is not None]
                if projects:
                    project_means.append(float(np.mean(projects)))
            report["seed_summary"].append({"model": model, "metric": metric, "seeds_defined": len(scores),
                "mean": float(np.mean(scores)) if scores else None,
                "sample_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "project_macro_mean": float(np.mean(project_means)) if project_means else None})
    for (model, condition), values in sorted(stress_all.items()):
        report["rq2"].append({"model": model, "condition": condition, **uncertainty(values, settings, rng)})
    holm(report["rq2"])
    # Shared structural cohort per condition; T4 compares only message-using models.
    for condition in sorted({c for _, c in stress_all}):
        models = sorted(m for m in expected_models if stress_all.get((m, condition)))
        if len(models) < 2:
            continue
        cohorts = [set((r["project"], r["commit_id"]) for r in stress_all[(m, condition)]) for m in models]
        common = set.intersection(*cohorts)
        matched = {m: [r for r in stress_all[(m, condition)] if (r["project"], r["commit_id"]) in common] for m in models}
        for model, values in matched.items():
            for seed in sorted(expected_seeds):
                selected = [r for r in values if r["seed"] == seed]
                report["matched_stress"].append({"model": model, "seed": seed, "condition": condition,
                                                 **sensitivity(selected)})
        project_scores = defaultdict(dict)
        for project in sorted({p for p, _ in common}):
            for model in models:
                values = []
                for seed in sorted(expected_seeds):
                    selected = [r for r in matched[model] if r["project"] == project and r["seed"] == seed]
                    score = sensitivity(selected)["delta_mcc"]
                    if score is not None:
                        values.append(score)
                if len(values) == len(expected_seeds):
                    project_scores[project][model] = float(np.mean(values))
        blocks = [v for v in project_scores.values() if set(v) == set(models)]
        omnibus = None
        if len(models) == 3 and len(blocks) >= 6:
            matrix = np.array([[b[m] for b in blocks] for m in models])
            omnibus = friedman_model_pvalue(matrix)
        comparisons = []
        for first, second in itertools.combinations(models, 2):
            differences = np.array([b[first] - b[second] for b in blocks])
            nonzero = int(np.count_nonzero(differences))
            allowed = nonzero >= 6 and (len(models) == 2 or (omnibus is not None and omnibus < 0.05))
            p = float(wilcoxon(differences).pvalue) if allowed else None
            comparisons.append({"first": first, "second": second, "projects": len(blocks),
                "nonzero_projects": nonzero, "mean_delta_difference": float(differences.mean()) if len(differences) else None,
                "p": p, "reason": "" if allowed else "insufficient_nonzero_projects_or_omnibus"})
        report["rq3"].append({"condition": condition, "matched_commits": len(common),
            "evaluable_projects": len(blocks), "friedman_p": omnibus, "comparisons": comparisons,
            "project_effects": dict(project_scores)})
    holm([comparison for result in report["rq3"] for comparison in result["comparisons"]])
    write_json(root / "comparison.json", report)
    lines = ["# ApacheJIT comparison", "", "Seed-level metrics precede averaging; AP is average precision.", "",
             "| Method | MCC mean | MCC sample SD | AP mean |", "|---|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    for model in sorted(expected_models):
        entries = {e["metric"]: e for e in report["seed_summary"] if e["model"] == model}
        lines.append(f"| {model} | {fmt(entries['mcc']['mean'])} | {fmt(entries['mcc']['sample_sd'])} | {fmt(entries['ap']['mean'])} |")
    lines += ["", "See comparison.json for per-project scores, stress coverage, matched cohorts, confidence intervals and corrected tests.",
              "", "Intervals are conditional on this benchmark; related projects and commits limit independence."]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
