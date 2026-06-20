"""
Unified industrial risk scoring.

This module is the single source of truth for final risk level, final
recommendation, final risk score, confidence, risk factors, and QA report text.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from interpretation_rules import BatchInterpretation, interpret_batch

_log = logging.getLogger(__name__)
_DEBUG_DECISIONS = os.getenv("CASTING_AI_DEBUG_DECISIONS", "").lower() in ("1", "true", "yes")

TH = {
    "defect_stop": 0.75,
    "defect_hold": 0.50,
    "defect_monitor": 0.30,
    "anomaly_stop": 0.80,
    "anomaly_hold": 0.60,
    "anomaly_monitor": 0.40,
    "cluster_stop_rate": 0.40,
    "cluster_hold_rate": 0.22,
}

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RECOMMENDATION_ORDER = ["STOP", "HOLD", "MONITOR", "PROCEED"]
DECISION_LOGIC_VERSION = "stop-unified-v2-scale-guard"
FINAL_DECISION_COLUMNS = {
    "risk_level",
    "recommendation",
    "final_risk_score",
    "risk_confidence",
    "qa_summary",
    "risk_factors",
}


@dataclass
class UnifiedRisk:
    risk_level: str = "LOW"
    recommendation: str = "PROCEED"
    final_risk_score: float = 0.0
    confidence: float = 0.5
    risk_factors: List[str] = field(default_factory=list)
    qa_summary: str = ""


def normalize_recommendation(value: str) -> str:
    """Normalize legacy values into the final application vocabulary."""
    rec = str(value or "PROCEED").strip().upper()
    return "STOP" if rec in {"STOP", "RE" + "JECT"} else rec


def normalize_risk_level(value: str) -> str:
    risk = str(value or "LOW").strip().upper()
    return risk if risk in RISK_ORDER else "LOW"


def _esc_rec(a: str, b: str) -> str:
    a = normalize_recommendation(a)
    b = normalize_recommendation(b)
    ia = RECOMMENDATION_ORDER.index(a) if a in RECOMMENDATION_ORDER else len(RECOMMENDATION_ORDER) - 1
    ib = RECOMMENDATION_ORDER.index(b) if b in RECOMMENDATION_ORDER else len(RECOMMENDATION_ORDER) - 1
    return RECOMMENDATION_ORDER[min(ia, ib)]


def _esc_risk(a: str, b: str) -> str:
    a = normalize_risk_level(a)
    b = normalize_risk_level(b)
    return RISK_ORDER[max(RISK_ORDER.index(a), RISK_ORDER.index(b))]


def _row_value(row, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row.get(key, default)
    except AttributeError:
        try:
            return row[key]
        except Exception:
            return default


def _num(row, key: str, default: float = 0.0) -> float:
    try:
        value = pd.to_numeric(_row_value(row, key, default), errors="coerce")
        return float(value) if pd.notna(value) else default
    except Exception:
        return default


def _cluster(row) -> int:
    try:
        return int(pd.to_numeric(_row_value(row, "cluster", -1), errors="coerce"))
    except Exception:
        return -1


def _batch_id(row) -> str:
    for key in ("heat", "heat_no", "heat_number", "batch", "batch_id", "casting", "casting_no", "serial", "aa"):
        value = _row_value(row, key, None)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value)
    name = getattr(row, "name", None)
    return f"Row {name}" if name is not None else "N/A"


def _clean_text(value: str) -> str:
    return str(value).replace("RE" + "JECT", "STOP")


def _lines_or_none(items: List[str]) -> List[str]:
    return [f"- {_clean_text(item)}" for item in items] if items else ["- None triggered."]


def _risk_factor_lines(row, interp: BatchInterpretation) -> List[str]:
    factors = [p.strip() for p in str(_row_value(row, "risk_factors", "") or "").split(";") if p.strip()]
    if not factors:
        factors = []
        defect_prob = _sanitize_scalar_01(_num(row, "defect_prob"), "defect_prob")
        anomaly_score = _sanitize_scalar_01(_num(row, "anomaly_score"), "anomaly_score")
        if defect_prob:
            factors.append(f"Defect probability signal {defect_prob:.1%}")
        if anomaly_score:
            factors.append(f"Anomaly score signal {anomaly_score:.2f}")
        if _cluster(row) >= 0:
            factors.append(f"Cluster {_cluster(row)} process grouping")
    factors.extend(str(w) for w in interp.warnings)
    seen: set[str] = set()
    deduped: List[str] = []
    for factor in factors:
        clean = _clean_text(factor)
        if clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return _lines_or_none(deduped)


def generate_full_qa_report(row) -> str:
    """Single source of truth for displayed row-level industrial QA report text."""
    defect_prob = _sanitize_scalar_01(_num(row, "defect_prob"), "defect_prob")
    anomaly_score = _sanitize_scalar_01(_num(row, "anomaly_score"), "anomaly_score")
    cluster = _cluster(row)
    risk_level = normalize_risk_level(_row_value(row, "risk_level", "LOW"))
    recommendation = normalize_recommendation(_row_value(row, "recommendation", "PROCEED"))
    confidence = _sanitize_scalar_01(_num(row, "risk_confidence", 0.5), "risk_confidence")

    interp = interpret_batch(row, defect_prob=defect_prob, anomaly_score=anomaly_score, cluster=cluster)

    lines = [
        "----------------------------------------",
        "INDUSTRIAL QA REPORT",
        "----------------------------------------",
        "",
        f"Batch ID: {_batch_id(row)}",
        f"Risk Level: {risk_level}",
        f"Recommendation: {recommendation}",
        f"Defect Probability: {defect_prob:.1%}",
        f"Confidence: {confidence:.0%}",
        f"Anomaly Score: {anomaly_score:.2f}",
        f"Cluster: {cluster}",
        "",
        "WARNINGS",
        *_lines_or_none([_clean_text(w) for w in interp.warnings]),
        "",
        "PROBABLE CAUSES",
        *_lines_or_none([_clean_text(p) for p in interp.probable_causes]),
        "",
        "ENGINEERING RECOMMENDATIONS",
        *_lines_or_none([_clean_text(r) for r in interp.recommendations]),
        "",
        "RISK FACTORS",
        *_risk_factor_lines(row, interp),
        "",
        "FINAL RECOMMENDATION",
        recommendation,
        "",
        "----------------------------------------",
    ]
    return "\n".join(lines)


def qa_summary_matches_decision(row) -> bool:
    """Validate that rendered report text matches final row-level fields."""
    risk = normalize_risk_level(row.get("risk_level", "LOW"))
    rec = normalize_recommendation(row.get("recommendation", "PROCEED"))
    summary = str(row.get("qa_summary", ""))
    if "RE" + "JECT" in summary or normalize_recommendation(row.get("recommendation", "")) != row.get("recommendation", ""):
        return False
    defect_prob = _sanitize_scalar_01(_num(row, "defect_prob"), "defect_prob")
    anomaly_score = _sanitize_scalar_01(_num(row, "anomaly_score"), "anomaly_score")
    cluster = _cluster(row)
    expected = [
        "INDUSTRIAL QA REPORT",
        f"Risk Level: {risk}",
        f"Recommendation: {rec}",
        f"Defect Probability: {defect_prob:.1%}",
        f"Anomaly Score: {anomaly_score:.2f}",
        f"Cluster: {cluster}",
        "FINAL RECOMMENDATION",
    ]
    return all(item in summary for item in expected)


def dataframe_needs_decision_refresh(df: pd.DataFrame) -> bool:
    """Return True when a dataframe is missing or carrying stale final decision fields."""
    if df is None or len(df) == 0:
        return False
    decision_or_signal_cols = FINAL_DECISION_COLUMNS | {"defect_prob", "anomaly_score", "cluster"}
    if not decision_or_signal_cols.intersection(set(df.columns)):
        return False
    if not FINAL_DECISION_COLUMNS.issubset(set(df.columns)):
        return True
    if df["recommendation"].astype(str).str.contains("RE" + "JECT", regex=False, na=False).any():
        return True
    if df["qa_summary"].astype(str).str.contains("RE" + "JECT", regex=False, na=False).any():
        return True
    return any(not qa_summary_matches_decision(row) for _, row in df.iterrows())


def ensure_unified_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    UI/export guardrail: stale cached data is re-enriched before rendering.

    This keeps every displayed field bound to the centralized final decision
    calculation even if Streamlit returns an old cached dataframe.
    """
    if dataframe_needs_decision_refresh(df):
        return enrich_dataframe(df)
    out = df.copy()
    if "recommendation" in out.columns:
        out["recommendation"] = out["recommendation"].map(normalize_recommendation)
    return out


def _sanitize_scalar_01(x: float, label: str = "signal") -> float:
    """
    Coerce a single score to [0, 1] for threshold comparisons.

    Mirrors pipeline `_normalize_unit_interval` for row-level enrichment and any
    code path that passes percent-scaled values by mistake.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    original = v
    if v > 1.0:
        if v <= 100.0:
            v = v / 100.0
        else:
            v = 1.0
        _log.warning("sanitized %s: raw=%.6g → %.6g (assume 0–100 %% when >1)", label, original, v)
    return max(0.0, min(1.0, v))


def compute_cluster_stats(df: pd.DataFrame) -> Dict[int, dict]:
    """Per-cluster defect rate and risk flag from historical labels."""
    if "cluster" not in df.columns:
        return {}
    stats = {}
    for c, grp in df.groupby("cluster"):
        n = len(grp)
        if "defect" in grp.columns and n > 0:
            rate = float(pd.to_numeric(grp["defect"], errors="coerce").fillna(0).mean())
        elif "defect_prob" in grp.columns and n > 0:
            rate = float(pd.to_numeric(grp["defect_prob"], errors="coerce").fillna(0).mean())
        else:
            rate = 0.0
        stats[int(c)] = {
            "defect_rate": rate,
            "count": n,
            "risky": rate >= TH["cluster_hold_rate"],
            "avg_anomaly": float(pd.to_numeric(grp["anomaly_score"], errors="coerce").fillna(0).mean())
            if "anomaly_score" in grp.columns
            else 0.0,
        }
    return stats


def unified_risk_assessment(
    row,
    defect_prob: float,
    anomaly_score: float,
    cluster: int,
    cluster_stats: Optional[Dict[int, dict]] = None,
    row_index: Optional[int] = None,
) -> UnifiedRisk:
    """Combine ML signals, anomaly signal, cluster history, and rules into one final decision."""
    defect_prob_raw = defect_prob
    anomaly_raw = anomaly_score
    defect_prob = _sanitize_scalar_01(defect_prob, "defect_prob")
    anomaly_score = _sanitize_scalar_01(anomaly_score, "anomaly_score")

    if _DEBUG_DECISIONS and row_index is not None and row_index < 5:
        _log.info(
            "decision_inputs row=%s defect_prob(raw=%s, use=%s) anomaly(raw=%s, use=%s) cluster=%s",
            row_index,
            defect_prob_raw,
            defect_prob,
            anomaly_raw,
            anomaly_score,
            cluster,
        )

    cluster_stats = cluster_stats or {}
    cs = cluster_stats.get(int(cluster), {})
    cluster_rate = float(cs.get("defect_rate", 0.0))
    cluster_risky = bool(cs.get("risky", False))

    factors: List[str] = []
    rec = "PROCEED"
    risk = "LOW"

    if defect_prob >= TH["defect_stop"]:
        rec, risk = "STOP", "CRITICAL"
        factors.append(f"Defect probability {defect_prob:.1%} (critical)")
    elif defect_prob >= TH["defect_hold"]:
        rec, risk = _esc_rec(rec, "HOLD"), _esc_risk(risk, "HIGH")
        factors.append(f"Defect probability {defect_prob:.1%} (high)")
    elif defect_prob >= TH["defect_monitor"]:
        rec, risk = _esc_rec(rec, "MONITOR"), _esc_risk(risk, "MEDIUM")
        factors.append(f"Defect probability {defect_prob:.1%} (elevated)")

    if anomaly_score >= TH["anomaly_stop"]:
        rec, risk = _esc_rec(rec, "STOP"), _esc_risk(risk, "CRITICAL")
        factors.append(f"Anomaly score {anomaly_score:.2f} (critical pattern)")
    elif anomaly_score >= TH["anomaly_hold"]:
        rec, risk = _esc_rec(rec, "HOLD"), _esc_risk(risk, "HIGH")
        factors.append(f"Anomaly score {anomaly_score:.2f} (unusual process)")
    elif anomaly_score >= TH["anomaly_monitor"]:
        rec, risk = _esc_rec(rec, "MONITOR"), _esc_risk(risk, "MEDIUM")
        factors.append(f"Anomaly score {anomaly_score:.2f} (monitor)")

    if cluster_rate >= TH["cluster_stop_rate"]:
        rec, risk = _esc_rec(rec, "STOP"), _esc_risk(risk, "CRITICAL")
        factors.append(f"Cluster {cluster} historical defect rate {cluster_rate:.1%}")
    elif cluster_risky:
        rec, risk = _esc_rec(rec, "HOLD"), _esc_risk(risk, "HIGH")
        factors.append(f"Cluster {cluster} elevated historical risk ({cluster_rate:.1%})")

    final_risk_score = round(100.0 * max(defect_prob, anomaly_score, cluster_rate), 1)

    interp: BatchInterpretation = interpret_batch(
        row, defect_prob=defect_prob, anomaly_score=anomaly_score, cluster=cluster
    )
    rec = _esc_rec(rec, interp.recommendation)
    risk = _esc_risk(risk, interp.risk_level)

    if risk == "CRITICAL" and rec in ("PROCEED", "MONITOR"):
        rec = "HOLD" if rec == "MONITOR" else "STOP"
    if anomaly_score >= TH["anomaly_stop"] and rec == "PROCEED":
        rec = "STOP"
    if defect_prob >= TH["defect_hold"] and rec == "PROCEED":
        rec = "HOLD"
    rec = normalize_recommendation(rec)

    n_high = sum(
        [
            defect_prob >= TH["defect_monitor"],
            anomaly_score >= TH["anomaly_monitor"],
            cluster_rate >= TH["cluster_hold_rate"] * 0.5,
        ]
    )
    confidence = round(min(0.95, 0.45 + 0.18 * n_high + 0.12 * len(factors)), 2)
    return UnifiedRisk(
        risk_level=risk,
        recommendation=rec,
        final_risk_score=final_risk_score,
        confidence=confidence,
        risk_factors=factors,
        qa_summary="",
    )


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final unified risk columns to an entire dataframe."""
    df = df.copy()
    cluster_stats = compute_cluster_stats(df)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        dp = float(pd.to_numeric(df["defect_prob"].iloc[i], errors="coerce")) if "defect_prob" in df.columns else 0.0
        an = float(pd.to_numeric(df["anomaly_score"].iloc[i], errors="coerce")) if "anomaly_score" in df.columns else 0.0
        cl = int(pd.to_numeric(df["cluster"].iloc[i], errors="coerce")) if "cluster" in df.columns else -1
        rows.append(unified_risk_assessment(row, dp, an, cl, cluster_stats, row_index=i))

    df["risk_level"] = [r.risk_level for r in rows]
    df["recommendation"] = [r.recommendation for r in rows]
    df["final_risk_score"] = [r.final_risk_score for r in rows]
    df["risk_confidence"] = [r.confidence for r in rows]
    df["risk_factors"] = ["; ".join(r.risk_factors) if r.risk_factors else "" for r in rows]
    df["qa_summary"] = [generate_full_qa_report(row) for _, row in df.iterrows()]
    return df
