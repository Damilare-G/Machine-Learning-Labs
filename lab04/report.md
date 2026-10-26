# Lab 4: Data pipeline and quality

## 1. Goal
The goal of this lab is to build a fully reproducible, leakage‑free data pipeline for the UCI Adult dataset and evaluate a baseline classifier with and without engineered features. The pipeline must validate schema and ranges, handle missing values, prevent leakage, audit label quality, and produce deterministic processed data with SHA256 hashes. See `problem_definition.md` for the full half‑page description.

## 2. Method
- One command: `python lab04/src/run_pipeline.py`; configuration in `lab04/config.yaml`.
- **Ingestion:** Loaded via `fetch_openml(name='adult', version=2, as_frame=True)`; raw data left unmodified; row/column counts logged.
- **Validation:** Checked schema types, numeric ranges, null policy (`?`), and allowed categories. Violations reported; no rows dropped.
- **Split:** 70/30 stratified split (seed 42) performed *before* any cleaning.
- **Cleaning (train‑only fit):**
  - Median imputation for numeric, mode imputation for categorical.
  - Outlier clipping using 1st/99th percentiles from train only.
  - Exact duplicate removal from train only.
  - Test set left untouched.
  - Fitted transformer saved to `lab04/results/preprocessor.joblib`.
- **Derived features:** `capital_net`, `hours_per_age`. LogisticRegression baseline evaluated with and without them.
- **Label audit:** Duplicate‑conflict audit using all 14 features and again without `fnlwgt`; rule‑based checks for impossible values; 50 random rows manually inspected.
- **Reproducibility:** SHA256 hashes of processed train/test stored in `lab04/results/run_manifest.yaml`.
- **Environment:** Python 3.11.8, pandas 2.2.2, numpy 1.26.4, scikit‑learn 1.5.0, pyyaml 6.0.3 (per manifest). Pyarrow was 15.0.2 as noted in Lab 1.

## 3. Results

### **Ingestion**
- Rows = **48842**
- Columns = **15**

### **Validation report** (`results/validation_report.md`)
- **47 checks**, **0 violations**.
- All schema types correct.
- All numeric ranges valid.
- Missing values:
  - workclass: **2799** (5.73%)
  - occupation: **2809** (5.75%)
  - native-country: **857** (1.75%)
- 52 exact duplicate rows detected (kept for cleaning stage).

### **Cleaning**
- Train rows before dedup: **34189**
- Train rows after dedup: **34165** (24 removed)
- Test rows: **14653**
- Imputed cells:
  - Train: **4446**
  - Test: **2013**
- Clipped cells:
  - Train: **2357**
  - Test: **1026**

### **Model performance**

| Features          | Accuracy | F1 (>50K) |
|------------------|----------|-----------|
| without derived   | **0.8540** | **0.6601** |
| with derived      | **0.8578** | **0.6696** |

Derived features improved both accuracy and F1.

### **Label-quality audit**
- **Duplicate-conflict groups (all 14 features):** 5 groups (10 rows)
- **Duplicate-conflict groups (13 features, without fnlwgt):** 1029 groups (3873 rows)
- **Rule violations (full data):**
  - husband_female_or_wife_male: **4**
  - young_with_postgrad_degree: **2**
  - All other rules: **0**
- **Manual re-check of 50 rows:**  
  - 50/50 rows OK  
  - No impossible values found.

### **Reproducibility**
- Combined SHA256 = **8cb87468caf8ee62e8c259905c56df14a1954c24ce2698e8b382bcbc4bc257a2**
- Second run produced the same hash: **Yes**
- `pytest lab04/src`: **All tests passed** (pipeline executed successfully).

## 4. Three conclusions
1. The Adult dataset is structurally clean and passes all schema and range checks, but contains meaningful missingness and small pockets of label noise.
2. A leakage‑free pipeline with train‑only fitting significantly improves reliability and ensures reproducible results, confirmed by stable SHA256 hashes.
3. Simple engineered features (`capital_net`, `hours_per_age`) provide measurable improvements in both accuracy and F1, demonstrating the value of thoughtful feature engineering even in baseline models.