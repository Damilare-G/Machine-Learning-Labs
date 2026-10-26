# Problem definition (Task 1)

**Goal.** Predict whether a person's annual income exceeds $50K from demographic and employment attributes in the UCI Adult (Census Income) dataset, and build the data pipeline that makes this prediction trustworthy.

**Inputs.** 14 features per person: 6 numeric (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). `income` is the label and is never used as an input.

**Output.** Binary label `income`: `>50K` (positive) or `<=50K`.

**Success metrics.** F1 on the `>50K` class and accuracy, both on a held-out test set. About a quarter of rows are positive, so accuracy alone is misleading and F1 is the primary metric. The lab compares both with and without two derived features (`capital_net`, `hours_per_age`).

**Constraints.**
1. *No leakage.* The 70/30 stratified split happens before any cleaning. Imputation values, clipping bounds, scaling and encoding are all learned from the training split only. Duplicates are dropped from train only; the test set stays untouched.
2. *Reproducible.* Fixed seed (42), pinned library versions, a YAML config, and one command (`python lab04/src/run_pipeline.py`). A SHA256 hash of the processed data is stored with the config so two runs can be compared.
3. *Visible data quality.* Validation reports violations and never silently drops rows.

**Out of scope.** Model tuning and fairness mitigation. The model is only a baseline to show that the pipeline works.
