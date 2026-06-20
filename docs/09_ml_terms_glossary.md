# ML Terms Glossary

| Term | Simple Explanation | Technical Meaning | Casting AI Usage |
|---|---|---|---|
| Accuracy | How often the model is correct. | `(TP + TN) / total`. | Shown on ML Performance page. |
| Precision | How often a defect alarm is correct. | `TP / (TP + FP)`. | Measures false alarm quality. |
| Recall | How many real defects are caught. | `TP / (TP + FN)`. | Very important for avoiding escaped defects. |
| F1 Score | Balance of precision and recall. | Harmonic mean of precision and recall. | Used to judge balanced classifier quality. |
| ROC | Curve showing defect-catching vs false alarms. | TPR vs FPR across thresholds. | Shown as ROC Curve Section. |
| ROC-AUC | One score for ROC curve quality. | Area under ROC curve. | Used for model comparison and validation. |
| Precision-Recall Curve | Shows precision and recall tradeoff. | Precision vs recall across thresholds. | Useful when defect class is imbalanced. |
| Average Precision | Summary of PR curve. | Weighted mean of precision over recall changes. | Stored in model comparison. |
| PCA | Compress many features into fewer dimensions. | Principal Component Analysis. | Used before clustering and visualization. |
| Isolation Forest | Finds unusual points. | Tree-based anomaly detection using isolation depth. | Runtime anomaly scoring. |
| Local Outlier Factor | Finds points unusual compared with neighbors. | Density-based anomaly method. | Used during offline Stage 5 ensemble scoring. |
| Threshold | Cutoff for decision. | Probability or score boundary. | Converts probabilities into MONITOR/HOLD/STOP behavior. |
| False Positive | Healthy batch flagged as defective. | Predicted 1, actual 0. | Causes unnecessary inspection or hold. |
| False Negative | Defective batch missed. | Predicted 0, actual 1. | Dangerous because bad casting may proceed. |
| True Positive | Defective batch correctly caught. | Predicted 1, actual 1. | Desired defect detection. |
| True Negative | Healthy batch correctly accepted. | Predicted 0, actual 0. | Desired release decision. |
| Feature | Input variable for model. | Numeric column in model matrix. | Chemistry, process, and engineered signals. |
| Feature Importance | Which inputs influence predictions most. | Tree importances, coefficient magnitudes, or correlation fallback. | Used for explainability. |
| Confidence | How certain a decision is. | In this app, risk confidence from signal agreement. | Shown as `risk_confidence`. |
| Classification Report | Table of precision, recall, F1, support. | sklearn report by class. | Displayed in ML Performance. |
| Support | Number of samples in a class. | Count of actual examples per class. | Helps interpret metrics. |
| Class Imbalance | One class has fewer examples. | Unequal target distribution. | Defective class may be minority. |
| Stratified Split | Keeps class ratio during train/test split. | Sampling preserving label proportions. | Used in Stage 3. |
| Cross-Validation | Repeated train/evaluate splits. | K-fold validation. | 5-fold stratified CV used. |
| StandardScaler | Normalizes feature scale. | `(x - mean) / std`. | Used in sklearn pipelines, PCA, anomaly detection. |
| Imputation | Filling missing values. | Median fill in this project. | Prevents model crashes on blanks. |
| Leakage | Model sees information it should not know. | Features encode target directly. | Additive/code columns are removed. |
| Inference | Running model on new data. | Applying trained model without retraining. | Upload prediction flow. |
| Training | Learning patterns from labeled data. | Fitting model parameters. | Stage 3 classifier training. |
| Model Artifact | Saved model file. | Pickle/JSON used later. | Files in `models/`. |
| Schema Alignment | Match runtime columns to training features. | Exact names/order with fills for missing. | Prevents inference mismatch. |
| Anomaly Score | How unusual a batch is. | Normalized outlier score 0-1. | Higher score means more unusual. |
| Cluster | Group of similar batches. | KMeans label. | Used for process families and cluster defect rate. |
| Silhouette Score | Quality of clustering separation. | Measures cohesion and separation. | Used to choose KMeans `k`. |
| Confusion Matrix | Table of correct and incorrect predictions. | TN, FP, FN, TP counts. | Displayed in ML Performance. |
| Probability | Model-estimated likelihood. | `predict_proba` output. | `defect_prob`. |
| Calibration | Whether probabilities match real frequencies. | Reliability of predicted probabilities. | Future improvement area. |
