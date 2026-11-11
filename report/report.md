## Data

### Data quality and label handling

The dataset has 10 000 rows and 14 columns. It has no missing values, no duplicate rows (also when UDI and Product ID are ignored), and no values outside plausible ranges. `Type` always matches the first letter of `Product ID`, so no imputation or cleaning of the features was needed.

The labels are less consistent. By definition, `Machine failure` is 1 if and only if at least one failure mode is active, but:

| Issue | Rows |
| --- | --- |
| `Machine failure = 1` with no mode (`failure_no_mode`) | 9 |
| RNF active but `Machine failure = 0` (`mode_no_failure`) | 18 |
| More than one mode active (`multiple_modes`) | 24 |

**Binary task.** We keep all 10 000 rows and use `Machine failure` as given. The 9 `failure_no_mode` rows are valid failures. The 18 `mode_no_failure` rows are the only true inconsistency. We keep them as negatives, because relabelling them as failures would add positives that depend only on chance (RNF) and that no feature can predict.

**Multiclass task.** We predict five classes: No failure, TWF, HDF, PWF and OSF. RNF is excluded because it occurs at random, independently of the process variables. We drop 27 rows that have no learnable label: the 9 `failure_no_mode` rows and the 18 rows that are RNF only.

The 24 multi-mode rows are not errors. The modes are independent, so a machine can fail in two ways at once. A single-label classifier still needs exactly one label per row, so we give each of these rows its **rarest** mode, ranked by single-mode frequency (TWF < OSF < PWF < HDF). The order is a fixed constant, so it does not depend on the split. The final classes are 9 643 No failure, 106 HDF, 95 OSF, 83 PWF and 46 TWF.

We preferred this to dropping the multi-mode rows. Dropping them would remove 24 of the roughly 330 failures. It would also leave a test set with no multi-mode cases, which would make the evaluation optimistic. The cost is that on these rows, predicting the other mode that is also present counts as an error. In the error analysis we count how many multiclass errors are of this kind: [X] of [Y] test errors. (X and Y are placeholders for the actual numbers computed in the final eval)
