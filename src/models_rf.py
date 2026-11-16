import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np

SEED = 42
SELECTED_PATH = Path(__file__).resolve().parent.parent / "results" / "selected_features.json"

# Reference forest
BASELINE = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
}

# One-at-a-time sweeps
SWEEPS = {
    "n_estimators": [50, 100, 200, 300, 500, 800],
    "max_depth": [3, 5, 8, 12, 16, None],
    "min_samples_leaf": [1, 2, 3, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
    "class_weight": [None, "balanced", "balanced_subsample", {0: 1, 1: 5}, {0: 1, 1: 10}],
}

# Combined random search
SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 8, 12, 16, None],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
    "class_weight": [None, "balanced", "balanced_subsample", {0: 1, 1: 5}, {0: 1, 1: 10}],
}


def make_rf(**params):
    cfg = {**BASELINE, **params}
    return RandomForestClassifier(random_state=SEED, n_jobs=-1, **cfg)


def load_selected_features(path=SELECTED_PATH):
    with open(path) as f:
        data = json.load(f)
    return data["features"] if isinstance(data, dict) else list(data)

BASELINE_MULTI = {**BASELINE, "class_weight": "none"}

SWEEPS_MULTI = {
    "n_estimators": [50, 100, 200, 300, 500, 800],
    "max_depth": [3, 5, 8, 12, 16, None],
    "min_samples_leaf": [1, 2, 3, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
    "class_weight": ["none", "balanced", "balanced_subsample", "sqrt_balanced"],
}

SEARCH_SPACE_MULTI = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 8, 12, 16, None],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
    "class_weight": ["none", "balanced", "balanced_subsample", "sqrt_balanced"],
}


def make_class_weights(y):
    """Class-weight options for the multiclass task, built from the training labels (0..k-1).
    "sqrt_balanced": the weight of a class is the square root of (largest class / its size),
    a milder version of "balanced"."""
    counts = np.bincount(np.asarray(y))
    sqrt_balanced = {c: float(np.sqrt(counts.max() / counts[c])) for c in range(len(counts))}
    return {"none": None, "balanced": "balanced",
            "balanced_subsample": "balanced_subsample", "sqrt_balanced": sqrt_balanced}


def resolve_params(params, class_weights):
    """Replace the class-weight NAME in `params` by the object scikit-learn expects."""
    p = dict(params)
    if "class_weight" in p:
        p["class_weight"] = class_weights.get(p["class_weight"], p["class_weight"])
    return p