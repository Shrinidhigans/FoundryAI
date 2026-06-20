"""
=============================================================================
INTERPRETATION RULES — INDUSTRIAL QA RULE ENGINE
=============================================================================
Purpose : Generate structured engineering explanations, warnings,
          recommendations, and risk decisions for each batch.
          Report text is generated only by dashboard.risk_scoring.
Usage   : Can be called stand-alone or imported by the Streamlit dashboard.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class BatchInterpretation:
    """Structured output of the rule engine for one batch."""
    risk_level:       str = "LOW"
    recommendation:   str = "PROCEED"
    warnings:         List[str] = field(default_factory=list)
    probable_causes:  List[str] = field(default_factory=list)
    recommendations:  List[str] = field(default_factory=list)
    qa_summary:       str = ""
    defect_prob:      float = 0.0
    anomaly_score:    float = 0.0
    cluster:          int   = -1


# ---------------------------------------------------------------------------
# THRESHOLD CONSTANTS  (tunable for specific foundry)
# ---------------------------------------------------------------------------
THRESH = {
    "s_high":          0.015,   # Sulfur — high risk above this
    "s_critical":      0.025,   # Sulfur — critical above this
    "mn_s_low":        2.0,     # Mn/S — dangerous below this
    "mn_s_warn":       3.0,     # Mn/S — warning below this
    "ce_hypo":         4.20,    # CE — hypoeutectic risk
    "ce_hyper":        4.30,    # CE — hypereutectic risk
    "mg_rec_low":      0.35,    # Mg recovery — low
    "mg_rec_warn":     0.45,    # Mg recovery — warning
    "temp_loss_high":  80,      # Temp loss °C — high
    "temp_loss_warn":  50,      # Temp loss °C — warning
    "pour_temp_low":   1290,    # Pouring temp — cold pour
    "pour_temp_high":  1480,    # Pouring temp — overheated
    "defect_prob_crit":0.75,    # Classifier probability — critical
    "defect_prob_high":0.50,    # Classifier probability — high
    "defect_prob_med": 0.30,    # Classifier probability — medium
    "anomaly_crit":    0.75,    # Anomaly score — critical
    "anomaly_high":    0.55,    # Anomaly score — high
}

# Recommendation levels (ordered from most to least severe)
RECOMMENDATIONS = ["STOP", "HOLD", "MONITOR", "PROCEED"]


# ---------------------------------------------------------------------------
# RULE EVALUATION HELPERS
# ---------------------------------------------------------------------------
def _get(row, col, default=None):
    """Safely extract a value from a dict-like or Series row."""
    if isinstance(row, dict):
        return row.get(col, default)
    try:
        val = row[col]
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


def _escalate_recommendation(current: str, new: str) -> str:
    """Return the more severe recommendation."""
    current = "STOP" if str(current).upper() == "RE" + "JECT" else current
    new = "STOP" if str(new).upper() == "RE" + "JECT" else new
    ci = RECOMMENDATIONS.index(current) if current in RECOMMENDATIONS else 3
    ni = RECOMMENDATIONS.index(new) if new in RECOMMENDATIONS else 3
    return RECOMMENDATIONS[min(ci, ni)]


def _escalate_risk(current: str, new: str) -> str:
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ci = order.index(current) if current in order else 0
    ni = order.index(new) if new in order else 0
    return order[max(ci, ni)]


# ---------------------------------------------------------------------------
# INDIVIDUAL RULE FUNCTIONS
# ---------------------------------------------------------------------------

def rule_sulfur(row, interp: BatchInterpretation):
    s = _get(row, "s_")
    mn = _get(row, "mn_")
    if s is None:
        return

    if s > THRESH["s_critical"]:
        interp.warnings.append(
            f"CRITICAL: Sulfur = {s:.4f}% (limit {THRESH['s_critical']}%). "
            "Severe nodularisation failure expected."
        )
        interp.probable_causes.append(
            "Excessive sulfur from contaminated scrap, CRCA, or inadequate desulfurisation. "
            "Mg treatment severely depleted by S."
        )
        interp.recommendations.append("STOP this heat. Investigate charge materials and desulfurisation.")
        interp.risk_level = _escalate_risk(interp.risk_level, "CRITICAL")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "STOP")

    elif s > THRESH["s_high"]:
        interp.warnings.append(
            f"HIGH: Sulfur = {s:.4f}% (warning >{THRESH['s_high']}%). "
            "Nodularisation efficiency compromised."
        )
        interp.probable_causes.append("Elevated sulfur from charge or poor bath control.")
        interp.recommendations.append("Increase FSM dosage or consider re-treatment. Monitor nodule count.")
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")

    # Mn/S ratio check
    if mn is not None and s > 0:
        mn_s = mn / s
        if mn_s < THRESH["mn_s_low"]:
            interp.warnings.append(
                f"Mn/S ratio = {mn_s:.1f} (dangerously low, require >3). "
                "Insufficient manganese to neutralise sulfur."
            )
            interp.probable_causes.append("Low Mn addition or excessive S build-up.")
            interp.recommendations.append("Add manganese alloy; target Mn/S ≥ 3.")
            interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        elif mn_s < THRESH["mn_s_warn"]:
            interp.warnings.append(f"Mn/S ratio = {mn_s:.1f} (below recommended 3.0).")
            interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")


def rule_carbon_equivalent(row, interp: BatchInterpretation):
    ce = _get(row, "feat_ce_calculated") or _get(row, "ce")
    if ce is None:
        return

    if ce < THRESH["ce_hypo"]:
        gap = THRESH["ce_hypo"] - ce
        interp.warnings.append(
            f"CE = {ce:.3f}  (hypoeutectic, {gap:.3f} below eutectic). "
            "Micro-shrinkage and hard spots likely."
        )
        interp.probable_causes.append("Insufficient carbon or silicon in charge.")
        interp.recommendations.append(
            "Increase carbon addition. Check graphite/carburiser additions. "
            f"Target CE ≥ {THRESH['ce_hypo']}."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")

    elif ce > THRESH["ce_hyper"]:
        gap = ce - THRESH["ce_hyper"]
        interp.warnings.append(
            f"CE = {ce:.3f}  (hypereutectic, {gap:.3f} above eutectic). "
            "Graphite flotation and surface defects possible."
        )
        interp.probable_causes.append("Excess carbon or silicon, possibly from high heel content.")
        interp.recommendations.append(f"Reduce carbon addition. Target CE ≤ {THRESH['ce_hyper']}.")
        interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")


def rule_mg_recovery(row, interp: BatchInterpretation):
    mg_rec = _get(row, "mg_recovery_")
    if mg_rec is None:
        return

    if mg_rec < THRESH["mg_rec_low"]:
        interp.warnings.append(
            f"CRITICAL: Mg recovery = {mg_rec:.2%}. "
            f"Severely below target {THRESH['mg_rec_low']:.0%}. "
            "Flake or vermicular graphite highly probable."
        )
        interp.probable_causes.append(
            "Excessive sulfur consumption of Mg, high pouring temperature, "
            "long treatment-to-pour time, or insufficient FSM dose."
        )
        interp.recommendations.append(
            "STOP or conduct microscopy. Investigate treatment protocol. "
            "Check FSM wire speed and wire chemistry."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "CRITICAL")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "STOP")

    elif mg_rec < THRESH["mg_rec_warn"]:
        interp.warnings.append(
            f"Mg recovery = {mg_rec:.2%} (below recommended {THRESH['mg_rec_warn']:.0%}). "
            "Reduced nodularity expected."
        )
        interp.recommendations.append("Increase FSM dosage. Verify treatment timing.")
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "MONITOR")


def rule_temperature(row, interp: BatchInterpretation):
    tap   = _get(row, "tapping_temp")
    pour  = _get(row, "pouring_temp")
    loss  = _get(row, "feat_temp_loss")

    if loss is not None:
        if loss > THRESH["temp_loss_high"]:
            interp.warnings.append(
                f"Temperature loss = {loss:.0f}°C (critical, >{THRESH['temp_loss_high']}°C). "
                "High risk of cold shut, mis-run, and ladle lining erosion."
            )
            interp.probable_causes.append(
                "Poor ladle insulation, long transfer time, or crane delay."
            )
            interp.recommendations.append(
                "Pre-heat ladle. Reduce transfer time. Check ladle cover usage."
            )
            interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
            interp.recommendation = _escalate_recommendation(interp.recommendation, "MONITOR")

        elif loss > THRESH["temp_loss_warn"]:
            interp.warnings.append(
                f"Temperature loss = {loss:.0f}°C (elevated, >{THRESH['temp_loss_warn']}°C)."
            )
            interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")

    if pour is not None:
        if pour < THRESH["pour_temp_low"]:
            interp.warnings.append(
                f"Pouring temperature = {pour:.0f}°C (cold pour, <{THRESH['pour_temp_low']}°C). "
                "Incomplete filling and cold shuts expected."
            )
            interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
            interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")

        elif pour > THRESH["pour_temp_high"]:
            interp.warnings.append(
                f"Pouring temperature = {pour:.0f}°C (overheated, >{THRESH['pour_temp_high']}°C). "
                "Oxidation, gas porosity, and coarse grain risk."
            )
            interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")
            interp.recommendations.append("Allow controlled cooling before pouring.")


def rule_shrinkage(row, interp: BatchInterpretation):
    sh = _get(row, "feat_shrinkage_risk_index")
    if sh is not None and sh > 2.5:
        interp.warnings.append(
            f"Shrinkage risk index = {sh:.2f}. "
            "Significant volumetric shrinkage expected during solidification."
        )
        interp.probable_causes.append(
            "Hypoeutectic CE, low Si, or excessive pouring temperature — "
            "all reduce graphite expansion that compensates shrinkage."
        )
        interp.recommendations.append(
            "Review risering design. Increase Si within limits. "
            "Adjust CE toward eutectic."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")


def rule_gas_porosity(row, interp: BatchInterpretation):
    gas = _get(row, "feat_gas_risk_index")
    if gas is not None and gas > 1.5:
        interp.warnings.append(
            f"Gas porosity risk index = {gas:.2f}. "
            "Hydrogen/nitrogen blowholes or Mg vapour pockets possible."
        )
        interp.probable_causes.append(
            "Moisture in charge (CRCA, SG Pig), elevated N%, or over-treatment with Mg."
        )
        interp.recommendations.append(
            "Dry all charge materials. Check bath N%. Verify Mg addition rate."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")


def rule_ai_prediction(defect_prob: float, interp: BatchInterpretation):
    """Translate the AI classifier's defect probability into a warning."""
    if defect_prob >= THRESH["defect_prob_crit"]:
        interp.warnings.append(
            f"AI PREDICTION: {defect_prob:.1%} defect probability (CRITICAL). "
            "Multiple process factors converging toward defect."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "CRITICAL")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "STOP")

    elif defect_prob >= THRESH["defect_prob_high"]:
        interp.warnings.append(
            f"AI PREDICTION: {defect_prob:.1%} defect probability (HIGH RISK)."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")

    elif defect_prob >= THRESH["defect_prob_med"]:
        interp.warnings.append(
            f"AI PREDICTION: {defect_prob:.1%} defect probability (MONITOR)."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "MEDIUM")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "MONITOR")


def rule_anomaly(anomaly_score: float, interp: BatchInterpretation):
    """Translate anomaly score into a warning."""
    if anomaly_score >= 0.80:
        interp.warnings.append(
            f"ANOMALY: Score = {anomaly_score:.2f} (CRITICAL). "
            "Process pattern is radically different from all historical normal batches."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "CRITICAL")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "STOP")

    elif anomaly_score >= THRESH["anomaly_crit"]:
        interp.warnings.append(
            f"ANOMALY: Score = {anomaly_score:.2f} (CRITICAL). "
            "Process pattern is radically different from all historical normal batches."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "CRITICAL")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")

    elif anomaly_score >= THRESH["anomaly_high"]:
        interp.warnings.append(
            f"ANOMALY: Score = {anomaly_score:.2f} (HIGH). "
            "Unusual process behaviour detected."
        )
        interp.risk_level = _escalate_risk(interp.risk_level, "HIGH")
        interp.recommendation = _escalate_recommendation(interp.recommendation, "HOLD")


# ---------------------------------------------------------------------------
# MAIN INTERPRETATION FUNCTION
# ---------------------------------------------------------------------------
def interpret_batch(
    row,
    defect_prob: float = 0.0,
    anomaly_score: float = 0.0,
    cluster: int = -1
) -> BatchInterpretation:
    """
    Run ALL rules against a single batch row (dict or pd.Series).
    Returns a BatchInterpretation with consolidated findings.

    Parameters
    ----------
    row           : dict or pd.Series — one row of the feature dataset
    defect_prob   : float — ML classifier's predicted defect probability [0,1]
    anomaly_score : float — normalised anomaly score [0,1]
    cluster       : int   — cluster label from KMeans

    Returns
    -------
    BatchInterpretation
    """
    interp = BatchInterpretation(
        defect_prob=defect_prob,
        anomaly_score=anomaly_score,
        cluster=cluster,
    )

    # Apply all rule functions
    rule_sulfur(row, interp)
    rule_carbon_equivalent(row, interp)
    rule_mg_recovery(row, interp)
    rule_temperature(row, interp)
    rule_shrinkage(row, interp)
    rule_gas_porosity(row, interp)
    rule_ai_prediction(defect_prob, interp)
    rule_anomaly(anomaly_score, interp)

    return interp


# ---------------------------------------------------------------------------
# BATCH PROCESSING
# ---------------------------------------------------------------------------
def interpret_dataframe(
    df: pd.DataFrame,
    defect_probs: np.ndarray = None,
    anomaly_scores: np.ndarray = None,
    clusters: np.ndarray = None,
) -> pd.DataFrame:
    """
    Apply interpretation engine to every row and append results to df.
    """
    n = len(df)
    defect_probs   = defect_probs   if defect_probs is not None   else np.zeros(n)
    anomaly_scores = anomaly_scores if anomaly_scores is not None else np.zeros(n)
    clusters       = clusters       if clusters is not None        else np.full(n, -1)

    risk_levels    = []
    recommendations= []

    for i, (_, row) in enumerate(df.iterrows()):
        interp = interpret_batch(
            row,
            defect_prob=float(defect_probs[i]),
            anomaly_score=float(anomaly_scores[i]),
            cluster=int(clusters[i]),
        )
        risk_levels.append(interp.risk_level)
        recommendations.append(interp.recommendation)

    df = df.copy()
    df["risk_level"]     = risk_levels
    df["recommendation"] = recommendations
    return df


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_row = {
        "s_":             0.022,
        "mn_":            0.04,
        "c_":             3.62,
        "si_":            2.05,
        "ce":             4.17,
        "mg_recovery_":   0.32,
        "tapping_temp":   1390,
        "pouring_temp":   1295,
        "feat_temp_loss": 95,
        "feat_ce_calculated": 4.17,
        "feat_shrinkage_risk_index": 2.9,
        "feat_gas_risk_index": 1.8,
    }

    result = interpret_batch(test_row, defect_prob=0.78, anomaly_score=0.82, cluster=2)
    print(f"\nRisk: {result.risk_level} | Recommendation: {result.recommendation}")
