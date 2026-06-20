"""Cluster naming and behavioral signatures from data."""

from __future__ import annotations

import pandas as pd

from dashboard.risk_scoring import compute_cluster_stats


def describe_cluster(df: pd.DataFrame, cluster_id: int) -> dict:
    sub = df[df["cluster"] == cluster_id] if "cluster" in df.columns else df.iloc[:0]
    stats = compute_cluster_stats(df).get(int(cluster_id), {})
    defect_rate = stats.get("defect_rate", 0.0)

    name = f"Cluster {cluster_id}"
    signature = "Standard production mix"
    tendency = "Typical defect exposure for this operating mode"

    if defect_rate >= 0.35:
        name = f"High-Risk Cluster {cluster_id}"
        tendency = "Historically elevated defect rate — prioritize HOLD review"
    elif defect_rate >= 0.20:
        name = f"Elevated-Risk Cluster {cluster_id}"
        tendency = "Moderate defect tendency — monitor chemistry and temperature"

    if len(sub) == 0:
        return {"name": name, "signature": signature, "tendency": tendency, "defect_rate": defect_rate}

    # Temperature / chemistry characterization
    if len(sub) >= 2 and "pouring_temp" in sub.columns and "tapping_temp" in sub.columns:
        temp_std = pd.to_numeric(sub["pouring_temp"], errors="coerce").std()
        if pd.notna(temp_std) and temp_std < 15:
            signature = "High temperature stable production cluster"
        elif sub["feat_temp_loss"].mean() > 70 if "feat_temp_loss" in sub.columns else False:
            signature = "High ladle heat-loss / extended transfer cluster"

    if "feat_chemistry_instability" in sub.columns:
        if sub["feat_chemistry_instability"].mean() > 0.5:
            signature = "Chemistry drift / out-of-spec cluster"

    if "anomaly_score" in sub.columns and sub["anomaly_score"].mean() > 0.55:
        signature = "Anomalous process signature — rare operating point"

    return {
        "name": name,
        "signature": signature,
        "tendency": tendency,
        "defect_rate": defect_rate,
        "count": len(sub),
        "avg_defect_prob": float(sub["defect_prob"].mean()) if "defect_prob" in sub.columns else 0,
        "avg_anomaly": float(sub["anomaly_score"].mean()) if "anomaly_score" in sub.columns else 0,
    }
