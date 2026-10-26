"""Task 7:"""


def pct(x):
    return f"{100 * x:.1f}%"


def write(path, m, stats):
    smoke = "> **SMOKE TEST on synthetic data - not valid for submission.**\n\n" if m["synthetic_data"] else ""
    text = f"""{smoke}# Data Card: UCI Adult (Census Income), as processed in Lab 4

## Source
- Dataset: Adult / Census Income, donated to the UCI Machine Learning Repository (Becker & Kohavi, 1996;
  doi:10.24432/C5XW20). Extracted by Barry Becker from the 1994 US Census Bureau database.
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590).
- Task: predict whether annual income exceeds $50K (`>50K`) from 14 demographic and employment features.

## License
- CC BY 4.0, as listed on the UCI repository page for this dataset (confirm on the page before submission).

## Size
- Raw: {m['raw_rows']} rows x {m['raw_cols']} columns (14 features + income).
- Train/test split 70/30, stratified, `random_state={m['seed']}`: {m['train_rows_before_dedup']} / {m['test_rows']} rows.
- After dropping {m['train_duplicates_removed']} exact duplicate rows from train only: {m['train_rows_final']} train rows.
- Processed-data SHA256 (combined): `{m['sha256_combined']}`

## Composition and measured class balance
- Positive class (`>50K`): {pct(stats['positive_rate'])} of rows (imbalanced; accuracy alone is misleading, hence F1).
- Sex: {pct(stats['male_share'])} male / {pct(1 - stats['male_share'])} female.
- Race: {pct(stats['white_share'])} White; remaining groups are small.
- Native country: {pct(stats['us_share'])} United-States.
- Missing values (`?`/NaN): workclass {pct(stats['miss_workclass'])}, occupation {pct(stats['miss_occupation'])}, native-country {pct(stats['miss_country'])}.

## Known biases and limitations
- **Dated and local:** 1994 US data. Incomes are not inflation-adjusted and the $50K threshold means something very different today.
- **Skewed demographics:** men, White and US-born respondents dominate, so models can perform worse for under-represented groups; positive rates differ strongly by sex and race (see the label audit), and a model can reproduce those disparities.
- **Sensitive attributes** (`sex`, `race`, `native-country`) and proxies for them (`relationship`, `marital-status`, `occupation`) are present.
- **Missingness is not random:** missing values occur in workclass, occupation and native-country, so imputing the mode is an assumption, not a fact.
- **`fnlwgt`** is a census sampling weight, not a personal attribute; treating it as an ordinary feature is questionable.
- **Label quality:** the label is a binary threshold of a continuous quantity. The audit found {stats['conflicting_groups_14']} groups of identical feature rows with conflicting labels (all 14 features) and {stats['conflicting_groups_13']} when `fnlwgt` is ignored, so some label noise is irreducible.

## Privacy concerns
- Public, anonymised data without names or direct identifiers, but combinations of age, sex, race, occupation and country are quasi-identifiers; re-identification risk is low but not zero, especially for rare combinations.
- Do not link this dataset with other personal data. Fairness-sensitive attributes should only be used for auditing or with a documented justification.
- No consent or opt-out information is available for the individuals in the original census records.

## Processing and reproducibility
- Pipeline: `python lab04/src/run_pipeline.py` (config: `lab04/config.yaml`; manifest with hashes: `lab04/results/run_manifest.yaml`).
- Cleaning is fit on the training split only: `{m['preprocessor_file']}`.
- Derived features: `capital_net`, `hours_per_age`.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)