# UI Component Guide

## Purpose

This guide explains the dashboard UI components and what each one means.

Main files:

| File | Purpose |
|---|---|
| `dashboard/components.py` | Reusable Streamlit components. |
| `dashboard/charts.py` | Chart builders. |
| `dashboard/theme.py` | Colors and Plotly configuration. |
| `dashboard/pages/main_analytics.py` | Main dashboard layout. |
| `dashboard/pages/ml_performance.py` | ML validation layout. |

## Component Map

| Component | Function | Meaning |
|---|---|---|
| Page header | `page_header` | Title and subtitle for a page. |
| Section title | `section` | Separates dashboard content areas. |
| Help block | `help_block` | Expandable explanation text. |
| KPI card | `render_kpi` | Compact metric display. |
| Recommendation badge | `rec_badge` | Color-coded recommendation. |
| Warning banner | `warning_banner` | Alerts for high/critical conditions. |
| Chart renderer | `show_chart` | Safe Plotly chart display. |
| Risk panel | `risk_panel` | Final score, recommendation, risk level, confidence. |
| Status indicator | `status_indicator` | Shows ready/unavailable state. |

## Cards

Cards are used for important metrics such as Fleet Quality, Process Stability, Critical Batches, and Recommendations.

Interpretation:

| Card Type | How to Read |
|---|---|
| KPI card | Fast summary of a fleet or model metric. |
| ML metric card | Model validation result with color status. |
| Risk panel metric | Selected batch decision output. |

## Graphs

| Graph | Meaning |
|---|---|
| Donut chart | Distribution by category, such as anomaly severity. |
| Horizontal bar | Counts or feature impacts. |
| Gauge | Single score on a 0-1 or 0-100 visual scale. |
| Radar chart | Selected casting vs healthy fleet profile. |
| Scatter plot | Relationship between two variables, such as anomaly and defect probability. |
| Heatmap | Correlation or confusion matrix intensity. |
| Histogram | Confidence distribution. |
| ROC curve | Model separation quality. |
| Precision-recall curve | Tradeoff between catching defects and avoiding false alarms. |

## Gauges

Gauges appear for fleet quality, process stability, and defect probability.

| Gauge | Interpretation |
|---|---|
| Fleet Quality | Higher means healthier overall dataset. |
| Process Stability | Higher means more stable process behavior. |
| Defect Probability | Higher means selected batch is more likely defective. |

## Accordions / Expanders

Expanders keep advanced details available without overwhelming the main screen.

Examples:

| Expander | Content |
|---|---|
| Template validation debug | Normalized and matched columns. |
| Anomaly Report | Severity distribution and top anomalies. |
| Risk Intelligence | Cluster insights and QA excerpt. |
| Single Batch Analysis | Detailed selected-batch risk explanation. |
| Classification report | Raw sklearn classification report. |
| Knowledge Center | ML and foundry term explanations. |

## Confidence Distribution

The confidence histogram on ML Performance shows whether predictions are near the decision boundary or clearly leaning healthy/defective.

| Confidence Band | Meaning |
|---|---|
| Low confidence | Near 50 percent probability; needs review. |
| Medium confidence | Some separation from boundary. |
| High confidence | Strong model decision. |

## Confusion Matrix

The confusion matrix is a 2x2 grid:

| Cell | Meaning |
|---|---|
| True Negative | Healthy accepted. |
| False Positive | Healthy falsely flagged. |
| False Negative | Defective missed. |
| True Positive | Defective caught. |

## ROC Curve

The ROC curve shows how the model performs across thresholds.

| Shape | Meaning |
|---|---|
| Close to diagonal | Weak model, near random. |
| Bends top-left | Strong separation. |
| Higher AUC | Better ranking of defective vs healthy. |

## Precision-Recall Chart

This chart is important when defects are less common than healthy batches.

| Metric | Foundry Interpretation |
|---|---|
| Precision | How many alarms are real. |
| Recall | How many real defects are caught. |

## Recommendation Badge Colors

The theme maps recommendation labels to colors. Users should treat the color as a quick triage indicator, but the text recommendation is the source of truth.

| Recommendation | Meaning |
|---|---|
| PROCEED | Continue normal process. |
| MONITOR | Watch carefully. |
| HOLD | Engineering review needed. |
| STOP | Critical intervention required. |

## Dashboard Reading Order

Recommended user workflow:

1. Upload file.
2. Confirm validation success.
3. Read Fleet Overview.
4. Check Critical Batches and Recommendation split.
5. Inspect Defect Drivers.
6. Review Anomaly Report.
7. Select a casting.
8. Read Single Batch Analysis and QA Summary.
9. Export report.
