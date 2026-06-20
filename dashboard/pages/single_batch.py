"""Single batch QA inspection page."""

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.column_mapping import chemistry_columns
from dashboard.components import help_block, page_header, rec_badge, risk_panel, safe_render, section
from dashboard.exports import REPORTLAB_OK, render_page_exports, single_batch_qa_pdf
from dashboard.risk_scoring import ensure_unified_decisions
from dashboard.theme import RISK_COLORS
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting, safe_probability
from dashboard.utils.debug import log_df_info


def render(df: pd.DataFrame):
    df = safe_for_plotting(safe_dataframe(ensure_unified_decisions(df)))
    log_df_info("single_batch", df)
    if len(df) == 0:
        st.warning("No batch data available.")
        return
    _render_single_batch(df)


def _render_single_batch(df: pd.DataFrame):
    page_header("Single Batch Analysis", "Industrial QA inspection for one heat / batch")

    n = len(df)
    if n <= 1:
        idx = 0
        st.caption("Viewing batch index 0 (only row in dataset).")
    else:
        idx = st.slider("Batch index (row)", 0, n - 1, 0)
    row = df.iloc[idx]
    fleet_prob = safe_probability(pd.to_numeric(df["defect_prob"], errors="coerce").mean()) if "defect_prob" in df.columns else 0.0

    def_prob = safe_probability(row.get("defect_prob", 0))
    anom = safe_probability(row.get("anomaly_score", 0))
    cluster = int(pd.to_numeric(row.get("cluster", 0), errors="coerce") or 0)
    risk = str(row.get("risk_level", "LOW"))
    rec = str(row.get("recommendation", "PROCEED"))
    final_score = float(pd.to_numeric(row.get("final_risk_score", 0), errors="coerce") or 0)
    confidence = safe_probability(row.get("risk_confidence", 0.5))
    factors = str(row.get("risk_factors", ""))

    help_block(
        "Anomaly score",
        "0 = typical historical process; 1 = extremely unusual pattern. "
        "Scores above 0.6 trigger HOLD; above 0.8 trigger STOP.",
    )
    help_block(
        "Defect probability",
        "ML estimate that this batch will produce a defective casting (0–100%).",
    )

    risk_panel(risk, rec, final_score, confidence, factors)

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        if REPORTLAB_OK:
            try:
                st.download_button(
                    "QA inspection PDF",
                    single_batch_qa_pdf(row, idx),
                    f"batch_{idx}_qa.pdf",
                    "application/pdf",
                )
            except Exception as exc:
                st.caption(f"PDF export: {exc}")
    with exp2:
        st.metric("vs fleet avg defect prob", f"{def_prob:.1%}", f"{def_prob - fleet_prob:+.1%}")
    with exp3:
        st.metric("Confidence", f"{confidence:.0%}")

    try:
        render_page_exports(df.iloc[[idx]], f"batch_{idx}", f"Batch {idx} QA Report")
    except Exception as exc:
        st.caption(f"Exports: {exc}")

    left, right = st.columns([1, 1])
    with left:
        section("Defect probability")
        safe_render("Defect probability gauge", charts.fig_gauge, def_prob, "Defect Probability")
        st.markdown("**Cluster**")
        st.write(f"#{cluster}")

    with right:
        section("QA decision")
        color = RISK_COLORS.get(risk, "#888")
        st.markdown(
            f'<div class="ind-card"><span style="color:{color};font-size:1.4rem;font-weight:700;">{risk}</span> '
            f" &nbsp; <span>{rec}</span></div>",
            unsafe_allow_html=True,
        )
        rec_badge(rec)
        st.text_area("Engineering report", str(row.get("qa_summary", "")), height=420)

    section("Key process parameters")
    display_keys = [
        "tapping_temp", "pouring_temp", "feat_temp_loss", "s_", "c_", "si_", "mn_", "ce",
        "feat_ce_calculated", "mg_recovery_", "feat_mn_s_ratio", "feat_sulfur_risk",
        "feat_shrinkage_risk_index", "feat_gas_risk_index", "feat_chemistry_instability",
        "anomaly_score", "defect_prob", "c", "si", "s",
    ]
    avail = [(k, row[k]) for k in display_keys if k in row.index]
    if not avail:
        avail = [(k, row[k]) for k in row.index if isinstance(row[k], (int, float, np.floating))][:20]
    param_df = pd.DataFrame(avail, columns=["Parameter", "Value"])
    param_df["Value"] = param_df["Value"].apply(
        lambda x: f"{float(x):.4f}" if isinstance(x, (int, float, np.floating)) else str(x)
    )
    st.dataframe(param_df.set_index("Parameter"), width="stretch")

    section("Chemistry profile vs fleet")
    chem = chemistry_columns(df)
    if not chem:
        chem = list(df.select_dtypes(include=[np.number]).columns[:6])
    sub = df.iloc[[idx]]
    if chem:
        safe_render("Chemistry bars", charts.fig_chemistry_comparison, df, sub, chem[:6])
        try:
            vals_a = [float(pd.to_numeric(row[c], errors="coerce")) for c in chem[:6]]
            vals_b = [float(pd.to_numeric(df[c], errors="coerce").mean()) for c in chem[:6]]
            safe_render(
                "Chemistry radar",
                charts.fig_radar,
                [c.replace("_", " ").upper() for c in chem[:6]],
                vals_a,
                vals_b,
                f"Batch {idx}",
                "Fleet mean",
            )
        except Exception as exc:
            st.caption(f"Radar chart: {exc}")

    section("Why risky & recommendations")
    st.markdown(f"**Final recommendation:** {rec}")
    st.markdown(f"**Final risk level:** {risk}")
    if factors:
        for line in [p.strip() for p in factors.split(";") if p.strip()]:
            st.markdown(f"- {line}")
    else:
        st.caption("No additional centralized risk factors recorded for this batch.")
