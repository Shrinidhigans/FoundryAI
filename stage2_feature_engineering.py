"""
=============================================================================
STAGE 2 — METALLURGICAL FEATURE ENGINEERING
=============================================================================
Purpose : Add domain-specific, physics-aware features derived from raw
          chemistry and process parameters.  Every feature has an industrial
          justification.
Input   : outputs/melting_cleaned_stage1.csv
Output  : outputs/melting_features_stage2.csv
=============================================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_FILE  = os.path.join("outputs", "melting_cleaned_stage1.csv")
OUTPUT_FILE = os.path.join("outputs", "melting_features_stage2.csv")


# ---------------------------------------------------------------------------
# SAFE DIVISION — avoids divide-by-zero / NaN explosions
# ---------------------------------------------------------------------------
def safe_div(num, den, fill=0.0):
    """Element-wise safe division for pandas Series."""
    den_safe = den.copy().replace(0, np.nan)
    return (num / den_safe).fillna(fill)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING FUNCTIONS
# Each function is self-contained and documented.
# ---------------------------------------------------------------------------

def add_carbon_equivalent(df):
    """
    Carbon Equivalent (CE) — most fundamental cast iron quality metric.
    CE = C% + (Si% + P%) / 3
    A higher CE shifts solidification toward graphitic structures.
    CE > 4.3 → hypereutectic (shrinkage risk)
    CE < 4.2 → hypoeutectic (porosity risk)
    CE 4.2–4.3 → eutectic zone (optimal nodular iron)
    """
    if all(c in df for c in ["c_", "si_", "p__"]):
        df["feat_ce_calculated"] = df["c_"] + (df["si_"] + df["p__"]) / 3
    elif "ce" in df.columns:
        df["feat_ce_calculated"] = df["ce"]
    return df


def add_ce_zone_flags(df):
    """
    CE Zone Classification — critical for predicting shrinkage vs porosity.
    Hypereutectic: excess carbon → graphite flotation, surface defects.
    Hypoeutectic : insufficient carbon → micro-shrinkage, hard spots.
    """
    ce_col = "feat_ce_calculated" if "feat_ce_calculated" in df else "ce"
    if ce_col in df.columns:
        df["feat_ce_hypo_risk"]  = (df[ce_col] < 4.2).astype(int)
        df["feat_ce_hyper_risk"] = (df[ce_col] > 4.3).astype(int)
        df["feat_ce_optimal"]    = ((df[ce_col] >= 4.2) & (df[ce_col] <= 4.3)).astype(int)
    return df


def add_c_si_ratio(df):
    """
    C/Si Ratio — governs graphitisation tendency.
    In SG iron: C/Si ratio 1.6–1.8 is ideal for compact graphite.
    Below 1.4: risk of white iron zones (chill) → hard spots.
    Above 2.0: coarse graphite, lower tensile strength.
    """
    if "c_" in df and "si_" in df:
        df["feat_c_si_ratio"] = safe_div(df["c_"], df["si_"])
        df["feat_c_si_risk"]  = (
            (df["feat_c_si_ratio"] < 1.4) | (df["feat_c_si_ratio"] > 2.0)
        ).astype(int)
    return df


def add_mn_s_ratio(df):
    """
    Mn/S Ratio — critical neutralisation index.
    Sulphur causes hot-shortness and interferes with Mg nodularisation.
    Manganese ties up sulphur as MnS, which is less harmful.
    Rule: Mn/S ≥ 3 is required; Mn/S < 2 → high defect risk.
    """
    if "mn_" in df and "s_" in df:
        df["feat_mn_s_ratio"] = safe_div(df["mn_"], df["s_"])
        df["feat_mn_s_risk"]  = (df["feat_mn_s_ratio"] < 2.0).astype(int)
    return df


def add_sulfur_risk(df):
    """
    Sulfur Risk Index — elevated S degrades nodularity.
    S > 0.015%: nodularisation efficiency drops below 80%.
    S > 0.025%: severe risk — erratic Mg recovery, vermicular graphite.
    """
    if "s_" in df.columns:
        df["feat_sulfur_risk"] = np.select(
            [df["s_"] > 0.025, df["s_"] > 0.015],
            [2, 1],
            default=0
        )
    return df


def add_mg_recovery_features(df):
    """
    Mg Recovery % — measures treatment efficiency.
    Low Mg recovery (< 40%) → insufficient nodularisation → flake graphite.
    Erratic recovery (high variance across batches) → process instability.
    Formula: Mg Recovery = Mg_in_iron / Mg_added × 100
    """
    if "mg_recovery_" in df.columns:
        mg = df["mg_recovery_"]
        df["feat_mg_recovery_risk"]   = (mg < 0.40).astype(int)
        df["feat_mg_recovery_severe"] = (mg < 0.30).astype(int)
        # Deviation from median: proxy for batch-to-batch variability
        med_mg = mg.median()
        df["feat_mg_recovery_dev"] = np.abs(mg - med_mg)
    return df


def add_temperature_features(df):
    """
    Temperature Loss = Tapping Temp − Pouring Temp.
    High loss (> 80°C): ladle heat loss, poor insulation, long transfer time.
    Leads to: cold shuts, mis-runs, poor filling of thin sections.
    Low loss (< 20°C): possibly inaccurate measurement or very short ladle time.
    Thermal Stability: flag for batches near the edge of safe pouring range.
    """
    if "tapping_temp" in df and "pouring_temp" in df:
        df["feat_temp_loss"] = df["tapping_temp"] - df["pouring_temp"]
        df["feat_temp_loss_risk"]    = (df["feat_temp_loss"] > 80).astype(int)
        df["feat_temp_loss_low"]     = (df["feat_temp_loss"] < 20).astype(int)
    if "pouring_temp" in df.columns:
        # Pouring temp < 1300°C → cold pour; > 1500°C → excessive oxidation
        df["feat_pouring_stability"] = np.select(
            [df["pouring_temp"] < 1300, df["pouring_temp"] > 1500],
            [-1, 1],
            default=0
        )
        df["feat_pouring_risk"] = (df["feat_pouring_stability"] != 0).astype(int)
    return df


def add_oxidation_risk(df):
    """
    Oxidation Risk Index.
    High Al% + high O2 affinity elements → oxide inclusions.
    In SG iron: Al > 0.02% can cause sub-surface pinholes.
    Also: high pouring temp + long pouring time → oxide film risk.
    """
    score = pd.Series(0.0, index=df.index)
    if "al__" in df.columns:
        score += (df["al__"] > 0.02).astype(float)
    if "pouring_temp" in df.columns:
        score += (df["pouring_temp"] > 1450).astype(float) * 0.5
    df["feat_oxidation_risk"] = score
    return df


def add_graphitization_index(df):
    """
    Graphitisation Tendency Index.
    Silicon and Cerium promote graphite formation (graphitisers).
    Chromium and Manganese are carbide stabilisers (anti-graphitisers).
    Index > 0: graphitic tendency (good for SG iron)
    Index < 0: carbide tendency (white iron risk, hard spots)
    """
    pos, neg = pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
    if "si_"  in df.columns: pos += df["si_"] * 2.0
    if "ce_"  in df.columns: pos += df["ce_"] * 5.0
    if "cr_"  in df.columns: neg += df["cr_"] * 3.5
    if "mn_"  in df.columns: neg += df["mn_"] * 0.3
    if "mo_"  in df.columns: neg += df["mo_"] * 3.0
    df["feat_graphitization_index"] = pos - neg
    df["feat_white_iron_risk"]      = (df["feat_graphitization_index"] < 0).astype(int)
    return df


def add_shrinkage_risk(df):
    """
    Shrinkage Risk Index.
    Shrinkage occurs when metal contracts on solidification and insufficient
    liquid feed compensates the volume change.
    Risk factors:
      - CE below eutectic (hypoeutectic) → more shrinkage during solidification
      - High pouring temp → large solidification range
      - Low Si: reduced graphite expansion that compensates shrinkage
    """
    score = pd.Series(0.0, index=df.index)
    ce_col = "feat_ce_calculated" if "feat_ce_calculated" in df else "ce"
    if ce_col in df.columns:
        score += (df[ce_col] < 4.2).astype(float) * 1.5
    if "si_" in df.columns:
        score += (df["si_"] < 2.0).astype(float)
    if "pouring_temp" in df.columns:
        score += (df["pouring_temp"] > 1420).astype(float) * 0.5
    df["feat_shrinkage_risk_index"] = score
    return df


def add_gas_risk_index(df):
    """
    Gas Porosity Risk Index.
    Gas porosity (hydrogen / nitrogen pinholes) is driven by:
      - High moisture content in charge materials (CRCA, SG Pig, Pig Iron)
      - Elevated N% → nitrogen blowholes
      - High Mg% → Mg vapour pockets
    """
    score = pd.Series(0.0, index=df.index)
    if "n__"  in df.columns: score += (df["n__"]  > 0.008).astype(float) * 2
    if "mg_"  in df.columns: score += (df["mg_"]  > 0.055).astype(float)
    if "crca" in df.columns:
        crca_num = pd.to_numeric(df["crca"], errors="coerce").fillna(0)
        score += (crca_num > 50).astype(float) * 0.5
    df["feat_gas_risk_index"] = score
    return df


def add_chemistry_stability(df):
    """
    Chemistry Stability Score.
    A composite index measuring how far key elements are from their ideal range.
    Larger values → more deviation from target chemistry.
    Ideal ranges (SG iron):
      C  : 3.5–3.8 %
      Si : 2.0–2.8 %
      Mn : 0.1–0.4 %
      Mg : 0.03–0.055 %
    """
    score = pd.Series(0.0, index=df.index)
    ranges = {
        "c_":  (3.5, 3.8),
        "si_": (2.0, 2.8),
        "mn_": (0.1, 0.4),
        "mg_": (0.03, 0.055),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            dev = np.maximum(0, lo - df[col]) + np.maximum(0, df[col] - hi)
            score += dev
    df["feat_chemistry_instability"] = score
    df["feat_chemistry_ok"] = (score < 0.1).astype(int)
    return df


def add_fsm_efficiency(df):
    """
    FSM (Ferro Silicon Magnesium) Efficiency.
    FSM is the nodulariser wire/alloy. FSMaddition/MT is kg of FSM per metric
    tonne of melt. Normalising by bath sulphur gives a sulphur-corrected dose.
    Low FSM dose with high S → under-treatment → flake or vermicular graphite.
    """
    if "fsmaddition_mt" in df.columns and "s_" in df.columns:
        fsm = pd.to_numeric(df["fsmaddition_mt"], errors="coerce").fillna(0)
        df["feat_fsm_s_index"] = safe_div(fsm, df["s_"] * 100)
        df["feat_fsm_undertreat_risk"] = (
            (fsm < 8) & (df["s_"] > 0.012)
        ).astype(int)
    return df


def add_heel_charge_ratio(df):
    """
    Heel-to-Charge Ratio.
    Heel metal is the residual iron kept in the furnace between heats.
    A high heel ratio improves heat continuity but also carries over
    accumulated impurities. Ratio > 0.5 can increase S, N, and trace elements.
    """
    if "heel" in df.columns and "tapped_wt_" in df.columns:
        heel  = pd.to_numeric(df["heel"],       errors="coerce").fillna(0)
        tapped= pd.to_numeric(df["tapped_wt_"], errors="coerce").fillna(1)
        df["feat_heel_ratio"]      = safe_div(heel, tapped)
        df["feat_heel_risk"]       = (df["feat_heel_ratio"] > 0.5).astype(int)
    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_stage2(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE):
    log.info(f"Loading: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    log.info(f"Input shape: {df.shape}")

    original_cols = set(df.columns)

    # Apply all feature engineering functions
    feature_fns = [
        add_carbon_equivalent,
        add_ce_zone_flags,
        add_c_si_ratio,
        add_mn_s_ratio,
        add_sulfur_risk,
        add_mg_recovery_features,
        add_temperature_features,
        add_oxidation_risk,
        add_graphitization_index,
        add_shrinkage_risk,
        add_gas_risk_index,
        add_chemistry_stability,
        add_fsm_efficiency,
        add_heel_charge_ratio,
    ]

    for fn in feature_fns:
        df = fn(df)
        log.info(f"  Applied: {fn.__name__}")

    new_feat_cols = [c for c in df.columns if c.startswith("feat_")]
    log.info(f"New features added: {len(new_feat_cols)}")
    for c in new_feat_cols:
        log.info(f"  {c}")

    # Fill any NaNs introduced in feature columns
    df[new_feat_cols] = df[new_feat_cols].fillna(0)

    df.to_csv(output_file, index=False)
    log.info(f"Saved: {output_file}  Shape: {df.shape}")
    return df, new_feat_cols


if __name__ == "__main__":
    df, feat_cols = run_stage2()
    print(f"\nStage 2 complete. Shape: {df.shape}")
    print(f"New features ({len(feat_cols)}):\n  " + "\n  ".join(feat_cols))
