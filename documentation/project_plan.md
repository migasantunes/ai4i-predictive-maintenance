# ML Project Plan – AI4I Predictive Maintenance

Nov 11, 2026 · @José Miguel Luís Antunes

## Overview

We train two models (MLP and Random Forest) on two tasks (binary and 5-class) and submit by **Tue 17 Nov, 23:59**. The individual written defence is on Wed 18 Nov in the morning. That leaves 6.5 working days, so the plan front-loads data work and freezes model choices on Sun 15 Nov.

**Decisions taken**

- Models: **MLP** (mandatory neural network) and **Random Forest** (our second method), used on both tasks. That gives 4 trained models.
- Binary target: `Machine failure` (about 3.4% positives).
- Multiclass target: 5 classes. **No failure, TWF, HDF, PWF, OSF**. RNF is excluded because it is random and cannot be learned from the features. Rows with several modes at once, or with `Machine failure = 1` and no mode, are dropped or resolved by a documented rule.
- Engineered features: temperature difference (process − air), power (torque × angular speed), and strain (tool wear × torque). UDI and Product ID are dropped; Type (L/M/H) is one-hot encoded.
- Mandatory split rules: 75/25 train/test, `random_state=42`, stratified on the target. Split indices are saved to disk.
- Required metrics per model: accuracy, precision, recall and F1 for class 1, on validation and test (0 when undefined). Test confusion matrices are shown as counts, actual in rows and predicted in columns.

## Roles

Each person owns one stream end to end, but everyone must be able to explain the whole project, because the defence is individual.

| Member | Stream | Owns |
| --- | --- | --- |
| **A** | Data and final evaluation | Setup, data quality check, preprocessing and features, splits, final test evaluation and error analysis, report sections on data and methods |
| **B** | Random Forest and context | EDA and plots, feature selection, RF binary and multiclass, literature comparison, report sections on results and discussion |
| **C** | Neural network and packaging | Shared evaluation code, MLP binary and multiclass, executed notebook, README, dependencies, AI-use and contribution declaration |

## Repository and notebooks

Shared code lives in `src/`, each stream experiments in its own notebook, and one final notebook, `main.ipynb`, runs everything end to end for submission. This keeps the three of us from editing the same file at once.

```text
ml-ai4i-project/
├── README.md                    # setup + execution order
├── requirements.txt             # pinned versions
├── main.ipynb                   # SUBMITTED notebook, executed; AI-use + contribution declaration at the end (C)
├── data/
│   ├── raw/ai4i2020.csv         # original data, or download steps in README
│   └── splits/                  # saved indices: binary_train/test, multi_train/test (.npy)
├── src/
│   ├── data.py                  # load, quality checks, preprocess(), engineered features (A)
│   ├── splits.py                # create, save and load the 75/25 splits (A)
│   ├── evaluate.py              # metrics, repeated CV mean ± std, confusion matrices (C)
│   ├── models_rf.py             # Random Forest builder + tuning grid (B)
│   └── models_mlp.py            # MLP architecture + training loop (C)
├── notebooks/
│   ├── 01_data_quality_eda.ipynb      # T2, T3 (A, B)
│   ├── 02_feature_selection.ipynb     # T7 (B)
│   ├── 03_rf_experiments.ipynb        # T8, T10 (B)
│   ├── 04_mlp_experiments.ipynb       # T9, T11 (C)
│   └── 05_final_evaluation.ipynb      # T13 (A)
├── results/
│   ├── tuning_log.csv           # every config tried, with mean ± std
│   ├── best_params.json         # frozen configs from T12
│   ├── final_metrics.csv        # validation + test, all 4 models
│   └── figures/                 # EDA plots, tuning curves, confusion matrices
└── report/
    └── report.pdf               # max 3 pages
```

**How `main.ipynb` is organised**

1. Setup: imports, seeds, paths.
2. Data import and quality check (calls `src/data.py`).
3. EDA: the key figures only; the full exploration stays in notebook 01.
4. Preprocessing, features and the saved splits.
5. Binary task: RF and MLP with validation results (mean ± std).
6. Multiclass task: RF and MLP with validation results.
7. Final test evaluation: metrics tables and confusion matrices as counts.
8. Error analysis: missed failures and false alarms.
9. AI-use and contribution declaration.

A flag `RUN_TUNING = False` at the top loads `best_params.json` so the notebook runs in minutes; set to `True`, it repeats the full search.

**Working rules**

- Each person edits only the files marked with their letter; changes to someone else's file go through them.
- Clear notebook outputs before committing (or use `nbstripout`), so Git merges stay clean. Only `main.ipynb` is committed executed, at the end.
- Every experiment appends a row to `results/tuning_log.csv`. That log becomes the tuning-trend figures in the report.
- The README lists the execution order: install `requirements.txt`, place or download the data, then run `main.ipynb` top to bottom.

## Tasks and dependencies

A task can start only when everything in its "Depends on" column is done. T6 and T14 have no data dependency, so they run in parallel with the data work from day one.

| # | Task | Owner | Dates | Depends on | Output |
| --- | --- | --- | --- | --- | --- |
| T1 | Project setup: shared repo or Drive folder, Python environment, `requirements.txt`, data download, folder structure from "Repository and notebooks" | A | Wed 11 | none | Repo everyone can run |
| T2 | Data quality check: missing values, duplicates, types, inconsistent labels (failure with no mode, several modes, RNF rows) | A | Wed 11 to Thu 12 | T1 | List of rows to drop, with reasons |
| T3 | EDA: histograms, pie chart of classes, box plots per class, correlation matrix, imbalance analysis | B | Thu 12 | T1 | Figures plus 5 to 6 written findings |
| T4 | Preprocessing and features: drop UDI and Product ID, one-hot Type, add temp difference, power and strain, build the 5-class label, scaling for the MLP | A | Thu 12 | T2 | `preprocess()` function |
| T5 | Splits: 75/25 stratified, `random_state=42`, one split per task, saved indices; stratified 5-fold CV on train for validation | A | Thu 12 | T4 | Saved index files |
| T6 | Shared evaluation code: metrics with `zero_division=0`, repeated CV with mean and std, confusion matrices as counts, results logging table | C | Wed 11 to Thu 12 | T1 | `evaluate()` used by both models |
| T7 | Feature selection: RF importance and mutual information; compare all features, raw only, and engineered only | B | Fri 13 to Sat 14 | T5, T6 | Table of feature sets vs F1 |
| T8 | RF binary: baseline, then tune `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`; pick the decision threshold on validation | B | Fri 13 to Sat 14 | T5, T6 | Tuned RF with tuning curves |
| T9 | MLP binary: architecture, weighted BCE, tune layers, units, learning rate, dropout; early stopping; loss curves | C | Fri 13 to Sat 14 | T5, T6 | Tuned MLP with tuning curves |
| T10 | RF multiclass: same pipeline with the 5-class label, class weights, per-class metrics | B | Sat 14 to Sun 15 | T8 | Tuned multiclass RF |
| T11 | MLP multiclass: softmax output, weighted cross-entropy, per-class metrics | C | Sat 14 to Sun 15 | T9 | Tuned multiclass MLP |
| T12 | Model selection meeting: compare validation mean and std, freeze the final configs before touching the test set | A, B, C | Sun 15 (evening) | T7, T10, T11 | Frozen configs, written down |
| T13 | Final test evaluation and error analysis: test metrics, confusion matrices, inspect missed failures and false alarms | A | Mon 16 | T12 | Final results tables and figures |
| T14 | Literature comparison: 2 to 3 published results on AI4I 2020, compared with ours in our own words | B | Thu 12 to Mon 16 | none (final step needs T13) | Comparison paragraph with references |
| T15 | Report, max 3 pages PDF: data, methods, results, discussion, recommendation, one limitation, validation vs test disagreement | A, B | Mon 16 to Tue 17 | T13, T14 | Report PDF |
| T16 | Code package: main.ipynb executed top to bottom, README with execution order, dependencies, saved indices, data instructions, AI-use and contribution declaration | C | Mon 16 to Tue 17 | T13 | Zip ready to submit |
| T17 | Final review and submission | A, B, C | Tue 17, by 20:00 | T15, T16 | Submitted zip and PDF |
| T18 | Defence prep: cross-explain each stream, each person writes their own A4 sheet (both sides, handwritten) | A, B, C | Tue 17 (evening) | T15 | One A4 sheet each |

## Timeline

&#91;embedded content: Work plan by day, 11 to 18 Nov 2026\]

Thursday is the busiest day for A: the saved splits (T5) unblock both model streams. Saturday and Sunday overlap binary and multiclass work so the freeze can happen on Sunday evening.

## Critical path, risks and rules

The critical path is **T1 → T2 → T4 → T5 → T9 → T11 → T12 → T13 → T15 → T17**. Any delay on it moves the submission, so the MLP stream (C) is the one to watch. A must have the saved splits ready by Thursday night.

**Risks and fallbacks**

- *Splits late.* B and C build T6, T8 and T9 against a temporary split, then swap in the saved indices.
- *MLP tuning drags.* Cap it at about 10 configurations per task, logged in the results table, and stop on Saturday night for binary.
- *TWF scores near zero.* Expected, because TWF happens at a random point in its tool-wear window. Report it honestly as the project's limitation; don't chase it.
- *Report runs over 3 pages.* Results go in tables, not prose. A and B cut to fit on Tuesday morning, not Tuesday night.

**Rules for the whole team**

- Never pick models, thresholds or features from test results. The test set is touched once, in T13.
- Run every experiment over several seeds or CV folds and report mean ± std.
- For multiclass, report per-class precision, recall and F1, plus macro F1.
- Short daily check-in, about 15 minutes, to unblock dependencies.

## Submission checklist

- [ ] Executed notebook that runs top to bottom without errors
- [ ] Supporting scripts, if any
- [ ] Original data, or clear download instructions
- [ ] Saved split indices for both tasks
- [ ] Dependencies file (`requirements.txt`)
- [ ] README with the execution order
- [ ] AI-use and contribution declaration inside the notebook: tools used, their role, how output was checked, each member's contribution
- [ ] Report PDF, max 3 pages, including the final recommendation and one limitation
- [ ] Validation and test metrics for all 4 models: accuracy, precision, recall, F1 (class 1)
- [ ] Test confusion matrices as counts, with the change in missed failures and false alarms explained
- [ ] Comparison with published results, written in our own words
- [ ] Each member's handwritten A4 sheet for the defence
