# 🏭 Industrial Casting AI — Quality Monitoring Platform

A production-ready, end-to-end AI system for predicting defects in grey/SG iron
casting batches, detecting process anomalies, clustering process patterns, and
generating metallurgically-grounded QA reports.

---

## Architecture Overview

```
melting_cleaned_final.xlsx
         │
         ▼
stage1_data_cleaning.py         → outputs/melting_cleaned_stage1.csv
         │
         ▼
stage2_feature_engineering.py   → outputs/melting_features_stage2.csv
         │
         ├──▶ stage3_supervised_model.py  → models/best_classifier.pkl
         │                                   models/feature_columns.pkl
         │
         ├──▶ stage4_pca_clustering.py    → models/pca_model.pkl
         │                                   models/kmeans_model.pkl
         │
         ├──▶ stage5_anomaly_detection.py → models/isolation_forest.pkl
         │                                   outputs/melting_with_anomalies_stage5.csv
         │
         ├──▶ validation_and_kpi.py       → outputs/validation_report.txt
         │
         └──▶ stage9_streamlit_app.py     ← Interactive Dashboard
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (in order)
```bash
python3 stage1_data_cleaning.py
python3 stage2_feature_engineering.py
python3 stage3_supervised_model.py
python3 stage4_pca_clustering.py
python3 stage5_anomaly_detection.py
python3 validation_and_kpi.py
```

.\venv\Scripts\Activate.ps1

### 3. Launch Dashboard
```bash
streamlit run stage9_streamlit_app.py
```

---

## Execution Order (Critical)

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 1 | `stage1_data_cleaning.py`     | `melting_cleaned_final.xlsx` | `outputs/melting_cleaned_stage1.csv` |
| 2 | `stage2_feature_engineering.py` | `stage1` output | `outputs/melting_features_stage2.csv` |
| 3 | `stage3_supervised_model.py`  | `stage2` output | `models/best_classifier.pkl`, plots |
| 4 | `stage4_pca_clustering.py`    | `stage2` output | `models/pca_model.pkl`, `models/kmeans_model.pkl` |
| 5 | `stage5_anomaly_detection.py` | `stage4` output | `models/isolation_forest.pkl`, `outputs/melting_with_anomalies_stage5.csv` |
| 6 | `validation_and_kpi.py`       | `stage5` output + models | `outputs/validation_report.txt`, plots |
| 9 | `stage9_streamlit_app.py`     | all models + `stage5` output | Live dashboard |

---

## File Structure

```
casting_ai/
├── melting_cleaned_final.xlsx      ← Raw input data
│
├── stage1_data_cleaning.py
├── stage2_feature_engineering.py
├── stage3_supervised_model.py
├── stage4_pca_clustering.py
├── stage5_anomaly_detection.py
├── interpretation_rules.py
├── validation_and_kpi.py
├── stage9_streamlit_app.py
│
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── melting_cleaned_stage1.csv
│   ├── melting_features_stage2.csv
│   ├── melting_clustered_stage4.csv
│   ├── melting_with_anomalies_stage5.csv
│   ├── cluster_profiles.csv
│   ├── model_comparison.csv
│   ├── traceability_columns.csv
│   ├── cleaning_report.txt
│   └── validation_report.txt
│
├── models/
│   ├── best_classifier.pkl
│   ├── all_classifiers.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   └── isolation_forest.pkl
│
└── plots/
    ├── roc_curves.png
    ├── pr_curves.png
    ├── pca_variance.png
    ├── pca_clusters.png
    ├── elbow_silhouette.png
    ├── anomaly_distribution.png
    ├── anomaly_pca.png
    └── validation_*.png
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Overview | KPI cards, defect distribution, anomaly severity pie, chemistry summary |
| 🔬 Single Batch | Gauge, QA report, process parameters for any selected batch |
| 📤 Upload & Predict | Upload new Excel/CSV → full pipeline runs → download predictions |
| 🗂️ Cluster Explorer | Interactive cluster selection, PCA scatter, chemistry profiles |
| ⚠️ Anomaly Report | Severity filter, anomaly vs defect scatter, top anomalous batches |
| 🧠 Risk Intelligence | Feature importance, recommendation counts, heatmap, QA reports |

---

## ML Architecture

### Defect Classification
- **Models compared**: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees
- **Imbalance handling**: `class_weight='balanced'` (upweights minority class = defective)
- **Leakage prevention**: `sklearn.Pipeline` — scaler fitted only on training split
- **Validation**: 5-fold stratified cross-validation, ROC-AUC + Recall optimisation
- **Expected performance**: ROC-AUC ~0.92, Recall ~0.70–0.75

### Anomaly Detection
- **Isolation Forest**: random partitioning, short paths = anomalies
- **Local Outlier Factor**: density comparison to neighbours
- **Ensemble**: weighted average of both scores

### Clustering
- **PCA**: 90% variance retained, dimensionality reduced for clustering
- **KMeans**: optimal k chosen by silhouette score (elbow as sanity check)
- **DBSCAN**: run for comparison, identifies noise points

### Feature Engineering (Stage 2)
All features are metallurgically motivated:

| Feature | Metallurgical Meaning |
|---------|----------------------|
| `feat_ce_calculated` | Carbon Equivalent — solidification mode |
| `feat_ce_hypo_risk` | Hypoeutectic → micro-shrinkage risk |
| `feat_mn_s_ratio` | Mn neutralises S; ratio < 3 = risk |
| `feat_sulfur_risk` | S > 0.015% impairs nodularisation |
| `feat_mg_recovery_risk` | Mg recovery < 40% → flake graphite |
| `feat_temp_loss` | Tapping − Pouring temp (ladle heat loss) |
| `feat_shrinkage_risk_index` | Composite: CE, Si, pouring temp |
| `feat_gas_risk_index` | N%, Mg excess, moisture indicators |
| `feat_chemistry_instability` | Deviation from target chemistry ranges |

---

## Leakage Columns (Excluded from ML)

The following columns were identified as **process decision codes** that perfectly
encode the defect label and must never be used for training:

| Column | Reason |
|--------|--------|
| `Sulphur` | Additive weight (kg) — coded per batch type |
| `Phos` | Additive weight (kg) — coded per batch type |
| `Crom` | Additive weight (kg) — coded per batch type |
| `Copper` | 450 kg for healthy, 10 kg for defective |
| `Heel_Metal` | 268 kg for healthy, 0.4 for defective |
| `Nickel` | 11 for healthy, 15 for defective |
| `Mg` (raw) | Bimodal: 373 (additive kg) vs 0.04 (chemistry %) |

These are kept in `outputs/traceability_columns.csv` for reference but not used in any model.

---

## Retraining Instructions

### When new data arrives:
```bash
# 1. Place new Excel file in project root
# 2. Update RAW_FILE in stage1_data_cleaning.py if filename changed
# 3. Re-run the full pipeline:
python3 stage1_data_cleaning.py
python3 stage2_feature_engineering.py
python3 stage3_supervised_model.py
python3 stage4_pca_clustering.py
python3 stage5_anomaly_detection.py
python3 validation_and_kpi.py
```

### Adding more models (if xgboost/lightgbm become available):
In `stage3_supervised_model.py`, add to `build_pipelines()`:
```python
"XGBoost": Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        scale_pos_weight=sum(y==0)/sum(y==1),  # handles imbalance
        n_estimators=300, learning_rate=0.05,
        max_depth=6, random_state=42
    )),
]),
```

---

## Debugging Common Issues

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `KeyError: 'defect'` | Column name changed in new dataset | Check `target_col_raw` in stage1 |
| Perfect AUC (1.0) | Data leakage from additive columns | Add leaking col to `LEAKAGE_COLUMNS_EXACT` in stage1 |
| Dashboard shows no data | Pipeline not run | Execute stages 1–5 first |
| Streamlit `ModuleNotFoundError` | Missing dependency | `pip install streamlit` |
| `Model file not found` | Models not trained | Run stage3 before dashboard |
| Poor recall on new data | Distribution shift | Retrain with combined old+new data |

---

## Adding Future Datasets

To add historical or new plant data:
1. Concatenate new data to existing cleaned CSV: `pd.concat([old_df, new_df])`
2. Ensure the `DEFECTED` column exists (0/1) in new data
3. Re-run the pipeline from stage1

To integrate a **second furnace** or **different grade**:
- Add a `furnace_type` or `grade` categorical feature to clustering
- Retrain with stratification on both defect label AND furnace type

---

## Industrial QA Thresholds (Configurable)

All thresholds are defined in `interpretation_rules.py → THRESH` dict:

```python
THRESH = {
    "s_high":          0.015,   # Sulfur % — warning
    "s_critical":      0.025,   # Sulfur % — critical
    "mn_s_low":        2.0,     # Mn/S ratio — dangerous
    "ce_hypo":         4.20,    # Carbon Equivalent — hypoeutectic
    "ce_hyper":        4.30,    # Carbon Equivalent — hypereutectic
    "mg_rec_low":      0.35,    # Mg Recovery — critical
    "temp_loss_high":  80,      # Temperature loss °C — critical
    "pour_temp_low":   1290,    # Pouring temperature °C — cold pour
    "defect_prob_crit":0.75,    # AI classifier — critical risk
    ...
}
```

Adjust these values to match your foundry's specific control plan.
