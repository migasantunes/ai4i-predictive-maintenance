import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from pathlib import Path


def _fold_metrics(y_true, pred):
    """Metrics for class 1 of one fold. Undefined metrics count as 0."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

###############################################################################


def evaluate(model, X, y, n_splits=5, n_repeats=3, threshold=0.5, random_state=42):
    """
    Repeated stratified k-fold cross-validation on the TRAINING set only.

    For each fold, a fresh clone of the model is trained on the training part and
    evaluated on the held-out validation part, so the model passed in is never
    modified. The test set must never be given to this function. The final model
    (trained on the whole training set) is a separate step, done only after the
    configuration is frozen.

    Parameters:
    model: unfitted scikit-learn estimator or Pipeline with fit() and predict_proba().
        If the features need preprocessing, put the preprocessor inside a Pipeline
        so it is fitted on each training fold only (no data leakage).
    X: pd.DataFrame, training features (kept as a DataFrame, rows are selected by position).
    y: training labels (0/1), same length as X.
    n_splits: number of folds (default 5).
    n_repeats: number of times the k-fold split is repeated with a different shuffle (default 3).
    threshold: probability from which a row is predicted as failure (class 1).
        It only affects the metrics in the returned table. To try other
        thresholds without retraining, use metrics_at_threshold() or threshold_table().
    random_state: seed, so the folds are reproducible.

    Returns:
    results: pd.DataFrame, one row per fold (n_splits * n_repeats rows) with
        repeat, fold, accuracy, precision, recall, f1 (class 1) and tn, fp, fn, tp.
        Undefined metrics count as 0.
    fold_probas: list of dicts, one per fold, with repeat, fold, y_val and proba
        (predicted probability of class 1). Pass it to metrics_at_threshold().

    Example:
    >>> model = make_pipeline(make_preprocessor(),
    ...                       MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=42))
    >>> X_train, X_test, y_train, y_test = load_split("binary")
    >>> results, probas = evaluate(model, X_train, y_train, threshold=0.2)
    >>> results.mean(numeric_only=True)     # average over the 15 folds
    """



    y = np.asarray(y)
    # Create the cross-validation folds 
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # Initialize lists to store the metrics for each fold
    rows = []
    fold_probas = []
    # running through the folds and evaluating the model
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        #separate the training and validation folds
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        #create a clone of the model to avoid data leakage and fit it to the training data
        m = clone(model)
        #train the model with the folds
        m.fit(X_tr, y_tr)
        #probabilities of the classes for the validation set
        proba = m.predict_proba(X_val)[:, 1]
        #saving the probalities, so we can try with diferent thresholds later
        fold_probas.append({
            "repeat": i // n_splits,
            "fold": i % n_splits,
            "y_val": y_val,
            "proba": proba,
        })

        # probabilities of the positive class due to the threshold
        pred = (proba >= threshold).astype(int)
        rows.append({
            "repeat": i // n_splits,
            "fold": i % n_splits,
            **_fold_metrics(y_val, pred),
        })

    return pd.DataFrame(rows), fold_probas


##########################################################################


def metrics_at_threshold(fold_probas, threshold):
    """
    Recompute the validation metrics of every fold for a new decision threshold,
    without retraining the model.

    evaluate() trains the model once per fold and keeps the predicted probabilities.
    Changing the threshold only changes how those probabilities become 0/1
    predictions, so we can test many thresholds on the same probabilities
    instead of retraining 15 models for each one. The probabilities depend on the
    trained model: if the model or its hyperparameters change, run evaluate() again.

    Parameters:
    fold_probas: list of dicts, the second value returned by evaluate().
        Each dict has "repeat", "fold", "y_val" (true labels) and "proba"
        (predicted probability of class 1) for one validation fold.
    threshold: float between 0 and 1. A row is predicted as failure
        (class 1) when its probability is >= threshold.

    Returns:
    pd.DataFrame, one row per fold with accuracy, precision, recall, f1 and the
    confusion counts (tn, fp, fn, tp). Same format as the table from evaluate().

    Example:
    >>> results, probas = evaluate(model, X_train, y_train)
    >>> metrics_at_threshold(probas, 0.2)   # same table, now with threshold 0.2
    """

    rows = []
    for f in fold_probas:
        pred = (f["proba"] >= threshold).astype(int)
        rows.append({
            "repeat": f["repeat"],
            "fold": f["fold"],
            **_fold_metrics(f["y_val"], pred),
        })
    return pd.DataFrame(rows)


###############################################################################


def summarize(results):
    """
    Mean and std, over the folds, of the main validation metrics.

    Internal helper used by threshold_table() and to build the row for
    log_result(). The std shows how stable the model is across folds.

    Parameters:
    results: pd.DataFrame with one row per fold, as returned by evaluate() or
        metrics_at_threshold(). It needs the columns accuracy, precision,
        recall, f1, fn and fp.

    Returns:
    dict with <metric>_mean and <metric>_std for each of those metrics, the same
    names used as columns in results/tuning_log.csv.
    """

    metrics = ["accuracy", "precision", "recall", "f1", "fn", "fp"]
    summary = {}
    for m in metrics:
        summary[f"{m}_mean"] = results[m].mean()
        summary[f"{m}_std"] = results[m].std()
    return summary


##############################################################################


def threshold_table(fold_probas, thresholds):
    """
    Compare several decision thresholds side by side, without retraining.

    Use it after evaluate() to choose the threshold on validation data. Lower
    thresholds usually raise recall (fewer missed failures) and lower precision
    (more false alarms). The table shows that trade-off. Never choose the
    threshold with the test set.

    Parameters:
    fold_probas: list of dicts, the second value returned by evaluate().
    thresholds: list of floats between 0 and 1, e.g. [0.1, 0.2, 0.3, 0.4, 0.5].

    Returns:
    pd.DataFrame with one row per threshold: the threshold, plus the mean and
    std of each metric over the folds (see _summarize()).

    Example:
    >>> results, probas = evaluate(model, X_train, y_train)
    >>> tt = threshold_table(probas, [0.1, 0.2, 0.3, 0.4, 0.5])
    """
    rows = []
    for t in thresholds:
        table = metrics_at_threshold(fold_probas, t)
        rows.append({"threshold": t, **summarize(table)})
    return pd.DataFrame(rows)


###########################################################################

LOG_PATH = Path(__file__).resolve().parent.parent / "results" / "tuning_log.csv"

def log_result(model_name, task, config, threshold, summary, path=LOG_PATH):
    """
    Append one row to results/tuning_log.csv, the shared record of every
    configuration we tried.

    Use it after evaluating each configuration during tuning (not only the one
    we end up choosing), so the log shows how the metrics evolve as the
    hyperparameters change. That log is the source of the tuning figures in the
    report and of the model selection meeting. Only validation results go here,
    never test results.

    Parameters:
    model_name: str, e.g. "MLP" or "RF".
    task: str, "binary" or "multi".
    config: dict with the hyperparameters of this configuration. It is saved as
        a JSON string in a single column.
    threshold: float, the decision threshold the summary was computed at.
    summary: dict with the mean and std of the metrics, as returned by
        _summarize() on the table from evaluate() or metrics_at_threshold().
    path: csv file to append to (default: results/tuning_log.csv). The header is
        written only if the file is new or empty.

    Example:
    >>> results, probas = evaluate(model, X_train, y_train)
    >>> t = 0.2
    >>> summary = _summarize(metrics_at_threshold(probas, t))
    >>> log_result("MLP", "binary", {"hidden_layer_sizes": (16,), "max_iter": 200}, t, summary)
    """

    row = {
        "model": model_name,
        "task": task,
        "config": json.dumps(config, sort_keys=True, default=str),
        "threshold": threshold,
        **summary,
    }
    new_file = (not path.exists()) or path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(path, mode="a", header=new_file, index=False)