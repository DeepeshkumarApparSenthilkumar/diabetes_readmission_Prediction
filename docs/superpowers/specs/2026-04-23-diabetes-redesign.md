# Diabetes Readmission Prediction — Full Redesign Spec
**Date:** 2026-04-23
**Author:** Deepeshkumar Appar Senthilkumar
**Deliverable:** Academic submission — code + results + report + GitHub repo

---

## 1. Goal

Produce a complete, professional-grade data science project predicting 30-day
hospital readmission in diabetic patients. Every stage of the pipeline must be
correct, reproducible, and documented. The final submission includes:

- 7 R scripts (full pipeline, runnable in order)
- 1 Python script (SHAP analysis on the actual evaluated model)
- A <5-page academic report (R Markdown → HTML)
- A README.md covering setup, structure, results
- A public GitHub repo as the submission URL

---

## 2. Dataset

**Source:** UCI ML Repository — Diabetes 130-US hospitals for years 1999–2008
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**Files:**
- `data/raw/diabetic_data.csv` — 101,766 encounters × 50 features
- `data/raw/IDS_mapping.csv` — lookup table for coded admission/discharge IDs

**Known issues addressed in preprocessing:**
- Deceased/hospice patients (discharge_disposition_id ∈ {11,13,14,19,20,21}) — removed
- weight (96.9% missing), payer_code (39.6% missing) — dropped
- Duplicate patients (16,773 patients appear >1 time) — keep first encounter only
- ICD-9 codes (700+ levels) — grouped to 9 clinical categories
- age stored as range strings — converted to numeric midpoints
- `>30` readmissions coded as negative class (correct — not readmitted within 30 days)

---

## 3. File Structure

```
diabetes_readmission_Prediction/
├── R/
│   ├── 01_data_audit.R           # audit, missing summary, cardinality
│   ├── 02_preprocessing.R        # cleaning, encoding, SMOTE
│   ├── 03_eda.R                  # 20+ plots, statistical tests
│   ├── 04_feature_engineering.R  # interaction terms, risk scores
│   ├── 05_modeling.R             # LR + RF + XGB + LightGBM, grid search
│   ├── 06_ensemble.R             # stacking, threshold optimization
│   └── 07_evaluation.R           # full eval: AUC, PR, calibration, DeLong
├── python/
│   ├── shap_analysis.py          # SHAP on the actual R XGB model
│   └── requirements.txt
├── data/
│   ├── raw/                      # diabetic_data.csv, IDS_mapping.csv
│   └── processed/                # diabetes_clean.csv, diabetes_featured.csv
├── outputs/
│   ├── figures/                  # all plots (01–25+)
│   └── results/                  # CSVs, RDS models
├── docs/
│   └── report.Rmd                # academic report → HTML
├── README.md
└── renv.lock                     # reproducible R deps
```

---

## 4. Preprocessing Pipeline (`02_preprocessing.R`)

**Step 1 — Remove deceased/hospice patients**
Filter out discharge_disposition_id ∈ {11, 13, 14, 19, 20, 21}.
Rationale: these patients cannot be readmitted; including them creates label noise
and inflates the importance of discharge_disposition_id artificially.

**Step 2 — Deduplication**
Keep first encounter per patient_nbr. Prevents data leakage across train/test.
Expected rows after: ~69,600 (down from 101,766 raw).

**Step 3 — Drop high-missing columns**
Remove weight (96.9%), payer_code (39.6%), encounter_id (ID column).

**Step 4 — Impute remaining missing**
- race → "Unknown"
- medical_specialty → "Unknown"
- diag_1/2/3 NA → "Other" (handled in ICD-9 grouping)

**Step 5 — Binary target**
readmitted == "<30" → 1, everything else → 0.
Actual positive rate after cleaning: ~7.1%.

**Step 6 — Age encoding**
Range strings [0-10), [10-20) … [90-100) → numeric midpoints 5, 15, …, 95.

**Step 7 — ICD-9 grouping**
diag_1/2/3 collapsed to 9 categories: Circulatory, Respiratory, Digestive,
Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other.

**Step 8 — Medical specialty**
Keep top 10 by frequency, collapse rest to "Other".

**Step 9 — Admission ID decoding**
Use IDS_mapping.csv to convert admission_type_id, discharge_disposition_id,
admission_source_id from numeric codes to meaningful factor levels.

**Step 10 — Medication encoding**
A1Cresult: None/Norm/>7/>8 → 0/1/2/3 ordinal.
max_glu_serum: None/Norm/>200/>300 → 0/1/2/3 ordinal.
Medication change columns (24 drug columns): steady/up/down → ordinal -1/0/1.

**Step 11 — Class balancing (SMOTE)**
Applied to TRAINING SET ONLY after train/test split.
Library: smotefamily. Target: balanced classes for model training.
Test set stays at natural distribution (~7.1% positive).

---

## 5. EDA (`03_eda.R`) — 20+ Visualizations

**Class & target:**
- 01: Class distribution (bar + %)
- 02: Readmission rate by age group
- 03: Time in hospital by class (boxplot)
- 04: Prior inpatient visits by class (boxplot)
- 05: Number of medications (density by class)
- 06: Readmission rate by primary diagnosis (bar)
- 07: Insulin prescription vs readmission rate
- 08: A1C result vs readmission rate
- 09: Medication change vs readmission rate
- 10: Readmission rate by discharge disposition (top 10)

**Distributions:**
- 11: Missing data heatmap (before preprocessing)
- 12: Correlation heatmap (numeric features)
- 13: Number of diagnoses distribution
- 14: Emergency vs outpatient visits (scatter by class)
- 15: Race breakdown by readmission
- 16: Gender vs readmission

**Statistical tests:**
- Chi-square tests for all categorical features vs target — table of p-values
- Mann-Whitney U tests for all numeric features vs target — table of p-values
- Effect size (Cramer's V for categorical, rank-biserial r for numeric)
- 17: Significance plot (lollipop of p-values, threshold line at 0.05)

**Feature relationships:**
- 18: Polypharmacy distribution (num_medications buckets)
- 19: High utilizer breakdown (prior inpatient visits ≥ 3)
- 20: Diagnosis complexity (number_diagnoses vs readmission)

---

## 6. Feature Engineering (`04_feature_engineering.R`)

New features added on top of cleaned data:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `high_utilizer` | number_inpatient >= 3 → 1 | Strong prior signal |
| `polypharmacy` | num_medications > 10 → 1 | Complexity indicator |
| `age_x_inpatient` | age_numeric × number_inpatient | Interaction: old + frequent |
| `med_x_diagnoses` | num_medications × number_diagnoses | Complexity interaction |
| `total_visits` | number_outpatient + number_emergency + number_inpatient | Utilization total |
| `diab_primary` | diag_1 == "Diabetes" → 1 | Diabetes as primary DX |
| `any_change` | any medication changed during encounter → 1 | Management signal |
| `chronic_score` | sum of chronic condition indicators | Risk composite |

Save as `data/processed/diabetes_featured.csv`.

---

## 7. Modeling (`05_modeling.R`)

**Split:** Stratified 70/30. Set seed = 42.
**Class balancing:** SMOTE on train only. K=5 neighbors.
**Cross-validation:** 5-fold CV, metric = ROC-AUC, `twoClassSummary`.

**Models and tuning grids:**

| Model | Package | Tuning Grid |
|-------|---------|-------------|
| Logistic Regression | glm (caret) | C regularization via glmnet: alpha=1, lambda=10 values |
| Random Forest | randomForest (caret) | mtry ∈ {5,8,12,15}, ntree=200 |
| XGBoost | xgboost | max_depth ∈ {3,5,6}, eta ∈ {0.05,0.1}, subsample=0.8, colsample_bytree=0.8, nrounds=200 |
| LightGBM | lightgbm | num_leaves ∈ {31,63}, learning_rate ∈ {0.05,0.1}, n_iter=200 |

**Saved artifacts:**
- model_lr.rds, model_rf.rds, model_xgb.rds, model_lgbm.rds
- xgb_col_names.rds (fix for bug in original)
- test_set.rds, test_label.rds

---

## 8. Ensemble + Threshold Optimization (`06_ensemble.R`)

**Stacking:**
- Base learners: RF + XGB + LightGBM
- OOF (out-of-fold) predictions from 5-fold CV used as meta-features
- Meta-learner: Logistic Regression trained on OOF predictions
- Test predictions: average base learner probs → meta-learner

**Threshold optimization:**
- Compute F1 at every threshold 0.01–0.99
- Report: default threshold (0.5), F1-optimal threshold, recall-optimal threshold (for clinical use where missing a readmission is costlier than a false alarm)
- Save: `optimal_thresholds.csv`

---

## 9. Evaluation (`07_evaluation.R`)

**Metrics for all 5 models (LR, RF, XGB, LightGBM, Ensemble):**
- AUC (ROC)
- PR-AUC (Precision-Recall Area Under Curve)
- Precision, Recall, F1 at default threshold (0.5)
- Precision, Recall, F1 at F1-optimal threshold
- Specificity, NPV
- Accuracy (reported but not used for selection)

**Plots:**
- 21: ROC curves (all 5 models + random baseline)
- 22: PR curves (all 5 models)
- 23: Calibration curves (reliability diagrams)
- 24: Confusion matrices at optimal threshold (grid of 5)
- 25: Threshold vs F1/precision/recall curve (XGB + ensemble)

**Statistical tests:**
- DeLong test: pairwise AUC comparison between best model and others
- Report which differences are significant (p < 0.05)

**Save:** `model_comparison.csv` with all metrics.

---

## 10. SHAP Analysis (`python/shap_analysis.py`)

**Approach:** Export the R XGBoost model as JSON (`xgb.save.raw` or `xgb::xgb.save`),
load in Python via `xgb.Booster.load_model()`. This ensures SHAP runs on the
**exact same model** evaluated in step 7, not a retrained copy.

**Plots:**
- 10: SHAP bar (mean absolute SHAP, top 15)
- 11: SHAP beeswarm (direction + magnitude, top 15)
- 26: SHAP waterfall (single high-risk patient explanation)
- 27: SHAP force plot (interactive HTML)

**Save:** `shap_values.csv`, `shap_top15.csv`

---

## 11. Report (`docs/report.Rmd`)

Sections matching submission requirements:

| Section | Content |
|---------|---------|
| Abstract | 150-word summary: problem, dataset, best model (AUC/F1), key finding |
| Overview | Problem statement, HRRP context, literature (Strack 2014 + 2 others), methodology overview |
| Data Processing | Pipeline steps 1–11, data issues table, assumptions, class imbalance strategy |
| Data Analysis | Key EDA findings, 6 embedded figures, statistical test results table |
| Model Training | Feature engineering table, SMOTE rationale, model configs, CV strategy |
| Model Validation | Test set results, confusion matrices, PR curves, bias/risk discussion |
| Model Performance | Full comparison table (all 5 models × all metrics), DeLong test, threshold analysis |
| Conclusion | Best model recommendation, clinical implications, limitations, future work |
| Data Sources | UCI URL, dataset citation, IDS_mapping source |

Target: <5 pages when knitted to HTML (use `fig.width`, concise prose).

---

## 12. README.md

Structure:
- Title + one-line description
- Badges (optional: R version, Python version)
- Dataset description + UCI link
- Repo structure (file tree, one-line description each)
- Setup: R packages (renv::restore()), Python (pip install -r requirements.txt)
- How to run: numbered steps (source R scripts in order, then python shap_analysis.py, then knit report)
- Results table: all 5 models × AUC / PR-AUC / F1
- Key findings: 3 bullets
- Citation

---

## 13. Bugs Fixed

| Bug | Location | Fix |
|-----|----------|-----|
| xgb_col_names never saved | 04→05 boundary | Add saveRDS in 05_modeling.R |
| SHAP on retrained Python model | shap_analysis.py | Export R model as JSON, load in Python |
| Deceased patients in negative class | 02_preprocessing.R | Filter discharge_disposition_id |
| Wrong class balance % in report | report.Rmd | Recompute: 7.1% not 8.8% |
| dtest.rds saved but never loaded | 04_modeling.R | Remove dead saveRDS |
| data_audit.py empty | python/ | Either populate or remove |

---

## 14. R Dependencies

```r
tidyverse, skimr, janitor,   # audit + preprocessing
smotefamily,                  # SMOTE
ggplot2, scales, gridExtra,  # EDA plots
caret, randomForest,          # LR + RF
xgboost, lightgbm,            # gradient boosting
pROC, PRROC,                  # ROC + PR curves
rmdformats, knitr             # report
```

Lock with `renv::snapshot()` → `renv.lock`.

## 15. Python Dependencies (`requirements.txt`)

```
xgboost>=2.0
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```

---

## 16. Success Criteria

- All 7 R scripts run top-to-bottom without error
- Best model AUC ≥ 0.68 (up from 0.658)
- PR-AUC reported for all models
- SHAP on same model as evaluation
- Report knits to HTML without errors
- README has correct setup + run instructions
- All bugs listed in Section 13 are fixed
- Repo pushed and link is valid
