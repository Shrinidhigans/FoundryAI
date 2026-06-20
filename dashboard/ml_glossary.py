"""Beginner-friendly definitions for the ML Performance page."""

from __future__ import annotations

import streamlit as st

ML_TERMS: dict[str, str] = {
    "Accuracy": "How often the AI gives the correct prediction overall — both healthy and defective batches combined.",
    "Precision": "When the AI says a batch is defective, how often it is actually right. High precision means fewer false alarms.",
    "Recall": "Of all truly defective batches, how many the AI successfully caught. High recall means fewer defects slip through.",
    "F1 Score": "A single score that balances precision and recall. Useful when you care about both missing defects and false alarms.",
    "ROC AUC": "Measures how well the model separates healthy vs defective batches across all probability thresholds. Closer to 1.0 is better.",
    "Confusion Matrix": "A table showing correct and incorrect predictions: true healthy, true defective, false alarms, and missed defects.",
    "Feature Importance": "Shows which input measurements the model relied on most when making decisions.",
    "Prediction Probability": "A number from 0 to 1 showing how confident the model is that a batch will be defective.",
    "Defect Probability": "The estimated chance that this casting batch may develop a quality defect.",
    "Anomaly Score": "How unusual this batch looks compared with normal historical production. Higher means more different from typical heats.",
    "Risk Score": "A combined quality risk number that blends defect probability, anomaly level, and process rules.",
    "Critical Risk": "A batch flagged at the highest concern level — usually requires hold or STOP before pouring.",
    "Cluster": "A group of similar batches discovered from process data. Batches in the same cluster tend to behave alike.",
    "PCA": "A method that compresses many process variables into two visual axes so similar casting batches appear near each other.",
    "Outlier": "A batch that is very different from most historical production — may need extra review.",
    "Correlation": "Shows whether two process values tend to move together across batches.",
    "Positive Correlation": "When one value goes up, the other tends to go up as well (e.g. tapping and pouring temperature).",
    "Negative Correlation": "When one value goes up, the other tends to go down.",
    "False Positive": "The AI predicted defective, but the batch was actually healthy — a false alarm.",
    "False Negative": "The batch was actually defective, but the AI predicted healthy — a missed defect.",
    "Classification": "Sorting each batch into categories such as healthy (0) or defective (1).",
    "Inference": "Using the trained model on new batch data to produce predictions — what happens when you upload a file.",
    "Training Data": "Historical batches with known defect labels used to teach the model what patterns to look for.",
    "Prediction Confidence": "How strongly the model favors one outcome over another; often shown as probability or risk confidence.",
}

FOUNDRY_TERMS: dict[str, str] = {
    "Pouring Temperature": "Metal temperature when poured into the mold. Too low can cause cold shuts; too high can increase oxidation and shrinkage risk.",
    "Tapping Temperature": "Temperature when metal is tapped from the furnace. It affects fluidity and heat available during transfer.",
    "Shrinkage": "Volume reduction during solidification that can cause internal or external defects if feeding is insufficient.",
    "Gas Defect": "Porosity or blowholes caused by trapped gas — often linked to chemistry, moisture, or pouring conditions.",
    "Mg Recovery": "How much magnesium remained effective after nodularizing treatment. Low recovery reduces nodularity.",
    "Silicon Addition": "Silicon added to promote graphitization and control carbon equivalent in cast iron.",
    "Chemistry Stability": "How steady key elements (carbon, silicon, sulfur, etc.) stay within target ranges across batches.",
    "Desulfurization": "Process step to reduce sulfur, which otherwise interferes with magnesium treatment and nodularity.",
    "Heat Loss": "Temperature drop between tapping and pouring — often from ladle transfer time or poor insulation.",
    "Ladle Transfer": "Moving molten metal from furnace to pouring station; long transfers increase heat loss.",
    "Process Stability": "How consistent temperatures, chemistry, and timing remain from batch to batch.",
    "Fleet Mean": "The average value across all batches currently loaded — used as a benchmark for one batch or cluster.",
    "Selected Batch": "The single heat you are inspecting in detail on the Single Batch Analysis page.",
    "Batch": "One melting/pouring cycle — one row of data in the system, often one heat number.",
    "Casting": "The final metal part produced from a pour; quality defects are what we try to predict and prevent.",
    "Heat": "Foundry term for one melt cycle — same idea as a batch in this dashboard.",
    "Defect Trend": "How defect rate or defect probability changes over time or batch order — helps spot worsening periods.",
}


def render_glossary_panel() -> None:
    """Full glossary in two expanders at bottom of ML Performance page."""
    with st.expander("📘 ML terms — plain English definitions", expanded=False):
        for term, definition in ML_TERMS.items():
            st.markdown(f"**{term}:** {definition}")
    with st.expander("🏭 Foundry & process terms — plain English definitions", expanded=False):
        for term, definition in FOUNDRY_TERMS.items():
            st.markdown(f"**{term}:** {definition}")
