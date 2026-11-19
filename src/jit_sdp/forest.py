"""Sparse JITLine-ApacheJIT adaptation and capped DE/SMOTE selection."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def forest(cfg, seed):
    return RandomForestClassifier(n_estimators=cfg["trees"], criterion="gini", max_depth=None,
        min_samples_split=2, min_samples_leaf=1, max_features=cfg["max_features"],
        bootstrap=True, class_weight=None, n_jobs=cfg["n_jobs"], random_state=seed)


def resample(x, y, ratio, neighbors, seed):
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(y, minlength=2)
    if min(counts) == 0:
        raise ValueError("Training needs both classes")
    if min(counts) / max(counts) >= ratio:
        return x, y, "not_needed"
    if min(counts) < 2:
        raise ValueError("SMOTE needs at least two minority examples")
    k = min(int(neighbors), int(min(counts)) - 1)
    smote = SMOTE(sampling_strategy=float(ratio), k_neighbors=k, random_state=seed)
    sampled_x, sampled_y = smote.fit_resample(x, y)
    return sampled_x, sampled_y, f"smote_k={k}"


def tune(cfg, adapter, train, validation, seed):
    settings = cfg["jitline"]
    x, v = adapter.transform(train), adapter.transform(validation)
    y = np.asarray([r["buggy"] for r in train], dtype=int)
    labels = [r["buggy"] for r in validation]
    counts = np.bincount(y, minlength=2)
    floor = min(counts) / max(counts)
    if floor == 1 or min(counts) < 2:
        return {"ratio": float(floor), "neighbors": 1, "reason": "balanced_or_insufficient_minority"}, []
    rng = np.random.default_rng(seed)
    population = rng.random((settings["population"], 1))
    scores = np.full(len(population), -np.inf)
    log, best = [], None
    # Initial population + at most two DE generations; count actual fitting calls.
    for generation in range(3):
        for index in range(len(population)):
            if len(log) >= settings["candidate_evaluations"]:
                return best, log
            candidate = population[index].copy()
            if generation and len(population) >= 4:
                others = [i for i in range(len(population)) if i != index]
                a, b, c = rng.choice(others, 3, replace=False)
                mutant = np.clip(population[a] + 0.7 * (population[b] - population[c]), 0, 1)
                mask = rng.random(1) < 0.3
                mask[0] = True
                candidate = np.where(mask, mutant, candidate)
            ratio = 1.0
            upper = min(settings["smote_max_neighbors"], int(min(counts)) - 1)
            neighbors = 1 + int(round(candidate[0] * (upper - 1)))
            rx, ry, reason = resample(x, y, ratio, neighbors, seed)
            clf = forest(settings, seed).fit(rx, ry)
            score = float(roc_auc_score(labels, clf.predict_proba(v)[:, 1]))
            entry = {"evaluation": len(log) + 1, "generation": generation, "ratio": ratio,
                     "neighbors": neighbors, "validation_roc_auc": score, "resampling": reason}
            log.append(entry)
            if best is None or score > best["validation_roc_auc"]:
                best = {"ratio": ratio, "neighbors": neighbors, "validation_roc_auc": score}
            if score > scores[index]:
                population[index], scores[index] = candidate, score
    return best, log


class ForestPipeline:
    def __init__(self, cfg, adapter, seed, selected):
        self.cfg, self.adapter, self.seed, self.selected = cfg, adapter, seed, selected
        self.history = {}

    def fit(self, train, validation, out):
        import joblib
        x = self.adapter.transform(train)
        y = np.asarray([r["buggy"] for r in train], dtype=int)
        x, y, reason = resample(x, y, self.selected["ratio"], self.selected["neighbors"], self.seed)
        self.history = {"resampling": reason, "fitted_rows": len(y)}
        self.model = forest(self.cfg["jitline"], self.seed).fit(x, y)
        joblib.dump({"model": self.model, "adapter": self.adapter}, out / "forest.joblib")
        return self

    def predict(self, records):
        return self.model.predict_proba(self.adapter.transform(records))[:, list(self.model.classes_).index(1)].tolist()
