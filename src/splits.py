from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from .data import load_raw, preprocess

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
TASKS = ["binary", "multi"]
SEED = 42
TEST_SIZE = 0.25

def save_splits(df=None):
    """Create and save train/test indices for both tasks."""

    df = load_raw() if df is None else df
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    for task in TASKS:
        X, y = preprocess(df, task)
        train_idx, test_idx = train_test_split(X.index.to_numpy(), test_size=TEST_SIZE, stratify=y, random_state=SEED)

        assert not set(train_idx) & set(test_idx), "train/test overlap"
        np.save(SPLIT_DIR / f"{task}_train.npy", train_idx)
        np.save(SPLIT_DIR / f"{task}_test.npy", test_idx)

def load_split(task, df=None):
    """Return X_train, X_test, y_train, y_test from the saved indices."""

    df = load_raw() if df is None else df
    X, y = preprocess(df, task)
    train_idx = np.load(SPLIT_DIR / f"{task}_train.npy")
    test_idx = np.load(SPLIT_DIR / f"{task}_test.npy")
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

def get_cv(n_repeats=1):
    """
    Stratified 5-fold CV on the training set.
    n_repeats > 1 gives repeated runs for mean ± std.
    """

    return RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
