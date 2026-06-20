# Top Defect-Driving Parameters

Source of ranking: saved classifier importances from `models/best_classifier.pkl` and `models/feature_columns.pkl`. The saved best classifier is `GradientBoostingClassifier`, selected in `stage3_supervised_model.py`. PCA and Isolation Forest inputs come from `models/pca_feature_names.json` and `models/isolation_forest_feature_names.json`. Rule-based QA decisions come from `interpretation_rules.py` and `dashboard/risk_scoring.py`.

SHAP note: no SHAP implementation or SHAP artifact was found in the codebase. Explainability currently uses model `feature_importances_`, absolute coefficients where applicable, permutation importance in `dashboard/ml_evaluation.py`, or correlation fallback logic in `dashboard/pipeline.py` / `dashboard/charts.py`.

| Rank | Column name | Human-readable name | Classifier importance | Where used in pipeline | Why it influences defects according to code |
|---:|---|---|---:|---|---|
| 1 | `sb_1` | Antimony chemistry 1 | 0.280921 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 2 | `barinoc` | Barinoc inoculant/additive | 0.200552 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance comes from trained classifier. |
| 3 | `pouring_wt` | Pouring weight | 0.042839 | Defect classifier, PCA clustering, Isolation Forest | Numeric process input. No explicit rule rationale found; importance comes from trained classifier. |
| 4 | `silicon_addition_1` | Silicon addition 1 | 0.028261 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. Silicon is also used by CE, C/Si, graphitization, shrinkage, and chemistry-stability feature logic. |
| 5 | `cr` | Chromium | 0.028106 | Defect classifier, PCA clustering, Isolation Forest | Used by graphitization feature logic as a carbide stabilizer/anti-graphitizer. |
| 6 | `ti` | Titanium | 0.025903 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 7 | `pouring_temp` | Pouring temperature | 0.023035 | Defect classifier, PCA clustering, Isolation Forest, `feat_temp_loss`, `feat_pouring_stability`, `feat_pouring_risk`, `feat_oxidation_risk`, `feat_shrinkage_risk_index`, QA temperature rules | Code flags cold pour, overheated pour, temperature loss, oxidation, shrinkage, cold shut, mis-run, gas porosity, and coarse grain risk. |
| 8 | `cu` | Copper | 0.019741 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 9 | `mn_1` | Manganese 1 | 0.017616 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. Bridged to Mn logic in upload aliases; Mn/S rules use manganese to neutralize sulfur. |
| 10 | `bath_s` | Bath sulfur | 0.016747 | Defect classifier, PCA clustering, Isolation Forest | Numeric sulfur-related input. Sulfur rules state high sulfur degrades nodularisation and consumes Mg treatment. |
| 11 | `p` | Phosphorus | 0.014498 | Defect classifier, PCA clustering, Isolation Forest | Used in CE calculation when bridged to `p__`; CE rules classify hypo/hypereutectic risk. |
| 12 | `s` | Sulfur | 0.014159 | Defect classifier, PCA clustering, Isolation Forest | Sulfur rules directly escalate risk and recommendations because sulfur compromises nodularisation and Mg treatment. |
| 13 | `ni` | Nickel | 0.012766 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 14 | `tapping_temp` | Tapping temperature | 0.012588 | Defect classifier, PCA clustering, Isolation Forest, `feat_temp_loss`, QA temperature rules | Used with pouring temperature to compute temperature loss; code links high loss to cold shuts, mis-runs, ladle heat loss, and transfer delays. |
| 15 | `sb` | Antimony | 0.012120 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 16 | `pouring_time_sec` | Pouring time | 0.011934 | Defect classifier, PCA clustering, Isolation Forest | Numeric process input. No explicit rule rationale found; importance comes from trained classifier. |
| 17 | `v` | Vanadium | 0.011845 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 18 | `ce_1` | Cerium / CE variant 1 | 0.011718 | Defect classifier, PCA clustering, Isolation Forest | Graphitization feature logic uses `ce_` as a graphite promoter when present. |
| 19 | `w` | Tungsten | 0.011665 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 20 | `al_1` | Aluminium 1 | 0.011381 | Defect classifier, PCA clustering, Isolation Forest | Oxidation feature logic flags Al above 0.02 as oxide-inclusion/subsurface-pinhole risk when bridged to `al__`. |
| 21 | `nb` | Niobium | 0.011009 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 22 | `si` | Silicon | 0.010752 | Defect classifier, PCA clustering, Isolation Forest | Silicon is used by CE, C/Si, graphitization, shrinkage, and chemistry-stability logic; low Si contributes to shrinkage index. |
| 23 | `c` | Carbon | 0.010402 | Defect classifier, PCA clustering, Isolation Forest | Carbon is used by CE, C/Si, and chemistry-stability logic; CE rules link off-target CE to shrinkage/hard spots or graphite flotation. |
| 24 | `sg_pigaddition_1` | SG pig addition 1 | 0.010399 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge/addition input. No explicit rule rationale found; importance comes from trained classifier. |
| 25 | `rr_1` | Return/revert ratio 1 | 0.010311 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge input. No explicit rule rationale found; importance comes from trained classifier. |
| 26 | `n` | Nitrogen | 0.010036 | Defect classifier, PCA clustering, Isolation Forest, `feat_gas_risk_index` when bridged to `n__` | Gas porosity feature logic flags elevated nitrogen as blowhole risk. |
| 27 | `zn` | Zinc | 0.009163 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 28 | `tapped_wt` | Tapped weight | 0.008271 | Defect classifier, PCA clustering, Isolation Forest | Numeric process input. No explicit rule rationale found; importance comes from trained classifier. |
| 29 | `fsm` | FSM additive | 0.008021 | Defect classifier, PCA clustering, Isolation Forest | Numeric nodulariser/additive input. FSM efficiency feature logic exists for `fsmaddition_mt`, but this raw `fsm` column is used directly by models. |
| 30 | `co` | Cobalt | 0.007936 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 31 | `rr` | Return/revert ratio | 0.007611 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge input. No explicit rule rationale found; importance comes from trained classifier. |
| 32 | `b` | Boron | 0.007394 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 33 | `sn` | Tin | 0.006995 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 34 | `lcrca_addition_1` | LCRCA addition 1 | 0.006161 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge/addition input. CRCA is used by gas-risk logic when column `crca` is present. |
| 35 | `zr` | Zirconium | 0.005732 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 36 | `sg_pig` | SG pig | 0.005544 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge input. Gas-risk comments identify SG Pig moisture as a gas-porosity driver, but formula uses `crca`, `n__`, and `mg_`. |
| 37 | `as` | Arsenic | 0.005537 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 38 | `ultraseed` | Ultraseed inoculant | 0.005134 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance comes from trained classifier. |
| 39 | `graphite` | Graphite addition | 0.004732 | Defect classifier, PCA clustering, Isolation Forest | Numeric carbon/addition input. CE rules recommend carbon/graphite adjustment when CE is low. |
| 40 | `low_re_fsm` | Low rare-earth FSM | 0.004286 | Defect classifier, PCA clustering, Isolation Forest | Numeric FSM/additive input. No explicit rule rationale found; importance comes from trained classifier. |
| 41 | `crca` | CRCA charge | 0.004231 | Defect classifier, PCA clustering, Isolation Forest, `feat_gas_risk_index` | Gas-risk logic treats CRCA above 50 as moisture/charge-related porosity risk. |
| 42 | `silicon_addition_2` | Silicon addition 2 | 0.003437 | Defect classifier, PCA clustering, Isolation Forest | Numeric silicon additive input. Silicon participates in CE/graphitization/shrinkage logic through chemistry columns. |
| 43 | `pb` | Lead | 0.003300 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 44 | `fe` | Iron | 0.003080 | Defect classifier, PCA clustering, Isolation Forest | Numeric chemistry input. No explicit rule rationale found; importance comes from trained classifier. |
| 45 | `preseed` | Preseed inoculant | 0.002754 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance comes from trained classifier. |
| 46 | `ce` | Carbon equivalent | 0.002581 | Defect classifier, PCA clustering, Isolation Forest, `feat_ce_calculated`, CE rules | CE rules link low CE to micro-shrinkage/hard spots and high CE to graphite flotation/surface defects. |
| 47 | `sorel_addition` | Sorel addition | 0.002175 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge/addition input. No explicit rule rationale found; importance comes from trained classifier. |
| 48 | `sorel_total` | Sorel total | 0.001773 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge/addition input. No explicit rule rationale found; importance comes from trained classifier. |
| 49 | `last_heel_metal` | Last heel metal | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric process/charge input. Heel ratio feature logic exists for `heel` and `tapped_wt_`; this raw column is used directly by models. |
| 50 | `sg_pigaddition_2` | SG pig addition 2 | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric charge/addition input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 51 | `heel` | Heel | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Heel ratio feature logic links high heel ratio to carry-over of S, N, and trace elements when `tapped_wt_` is available. |
| 52 | `cs` | CS addition | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 53 | `al` | Aluminium | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Oxidation feature logic flags Al above 0.02 when bridged to `al__`; this raw column has zero saved classifier importance. |
| 54 | `high_re_fsm` | High rare-earth FSM | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric FSM/additive input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 55 | `mn` | Manganese | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Manganese is used in Mn/S and graphitization logic when available as `mn_`; this raw column has zero saved classifier importance. |
| 56 | `zero_re_fsm` | Zero rare-earth FSM | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric FSM/additive input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 57 | `superseed` | Superseed inoculant | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 58 | `reseed` | Reseed inoculant | 0.000000 | Defect classifier, PCA clustering, Isolation Forest | Numeric additive input. No explicit rule rationale found; importance is zero in the saved classifier. |
| 59 | `fsmaddition_mt` | FSM addition per metric ton | N/A | PCA clustering, Isolation Forest, `feat_fsm_s_index`, `feat_fsm_undertreat_risk` when available | FSM efficiency logic normalizes FSM dose by sulfur; low FSM with high sulfur is under-treatment risk. Excluded from supervised classifier by `stage3_supervised_model.py`. |
| signal | `defect_prob` | Defect probability | N/A | Produced by classifier in `dashboard/pipeline.py`; consumed by `dashboard/risk_scoring.py` and `interpretation_rules.py` | Thresholds drive MONITOR, HOLD, and STOP decisions. |
| signal | `anomaly_score` | Anomaly score | N/A | Produced by Isolation Forest/LOF in `stage5_anomaly_detection.py` or dashboard inference; consumed by risk scoring and QA rules | Thresholds drive MEDIUM, HIGH, and CRITICAL risk escalation for unusual process patterns. |
| signal | `cluster` | Cluster label | N/A | Produced by KMeans in `stage4_pca_clustering.py`; consumed by `compute_cluster_stats()` in `dashboard/risk_scoring.py` | Cluster historical defect rate can escalate HOLD or STOP decisions. |

# Top Defect-Driving ML Features

The table below is the complete feature union actually used by saved ML artifacts. `Classifier` means `models/best_classifier.pkl`; `PCA` means `models/pca_model.pkl`; `Isolation Forest` means `models/isolation_forest.pkl`. Classifier importance is unavailable for `fsmaddition_mt` because the supervised training code excludes it.

| Feature name | Classifier importance | Classifier | PCA | Isolation Forest | Formula or calculation source | Business meaning | Expected effect on defect prediction |
|---|---:|:---:|:---:|:---:|---|---|---|
| `sb_1` | 0.280921 | Y | Y | Y | Raw numeric input selected in `stage3_supervised_model.py` | Antimony chemistry 1 | Learned by classifier; highest saved importance. Direction not encoded in tree importance. |
| `barinoc` | 0.200552 | Y | Y | Y | Raw numeric input | Barinoc inoculant/additive | Learned by classifier; second-highest saved importance. Direction not encoded in tree importance. |
| `pouring_wt` | 0.042839 | Y | Y | Y | Raw numeric input | Pouring weight | Learned by classifier; direction not encoded in tree importance. |
| `silicon_addition_1` | 0.028261 | Y | Y | Y | Raw numeric input | Silicon addition 1 | Learned by classifier; silicon also affects engineered CE/chemistry features when present as chemistry. |
| `cr` | 0.028106 | Y | Y | Y | Raw numeric input | Chromium | Learned by classifier; graphitization logic treats chromium as anti-graphitizer when bridged to `cr_`. |
| `ti` | 0.025903 | Y | Y | Y | Raw numeric input | Titanium | Learned by classifier; direction not encoded in tree importance. |
| `pouring_temp` | 0.023035 | Y | Y | Y | Raw input; also used in `feat_temp_loss`, `feat_pouring_stability`, `feat_pouring_risk`, `feat_oxidation_risk`, `feat_shrinkage_risk_index` | Pouring temperature | Off-range pouring temperature and high thermal loss increase rule-based risk; classifier also learned this raw input. |
| `cu` | 0.019741 | Y | Y | Y | Raw numeric input | Copper | Learned by classifier; direction not encoded in tree importance. |
| `mn_1` | 0.017616 | Y | Y | Y | Raw numeric input | Manganese 1 | Learned by classifier; manganese is related to Mn/S sulfur neutralization logic when bridged. |
| `bath_s` | 0.016747 | Y | Y | Y | Raw numeric input | Bath sulfur | Learned by classifier; sulfur-related rules link sulfur to nodularisation/Mg treatment risk. |
| `p` | 0.014498 | Y | Y | Y | Raw numeric input | Phosphorus | Learned by classifier; CE formula uses phosphorus when bridged to `p__`. |
| `s` | 0.014159 | Y | Y | Y | Raw numeric input | Sulfur | Learned by classifier; sulfur rules directly escalate risk and recommendations. |
| `ni` | 0.012766 | Y | Y | Y | Raw numeric input | Nickel | Learned by classifier; direction not encoded in tree importance. |
| `tapping_temp` | 0.012588 | Y | Y | Y | Raw input; used in `feat_temp_loss` | Tapping temperature | Drives temperature-loss feature and QA temperature rules. |
| `sb` | 0.012120 | Y | Y | Y | Raw numeric input | Antimony | Learned by classifier; direction not encoded in tree importance. |
| `pouring_time_sec` | 0.011934 | Y | Y | Y | Raw numeric input | Pouring time | Learned by classifier; direction not encoded in tree importance. |
| `v` | 0.011845 | Y | Y | Y | Raw numeric input | Vanadium | Learned by classifier; direction not encoded in tree importance. |
| `ce_1` | 0.011718 | Y | Y | Y | Raw numeric input | Cerium / CE variant 1 | Graphitization logic uses `ce_` as graphite promoter when present. |
| `w` | 0.011665 | Y | Y | Y | Raw numeric input | Tungsten | Learned by classifier; direction not encoded in tree importance. |
| `al_1` | 0.011381 | Y | Y | Y | Raw numeric input | Aluminium 1 | Oxidation logic uses aluminium when bridged to `al__`. |
| `nb` | 0.011009 | Y | Y | Y | Raw numeric input | Niobium | Learned by classifier; direction not encoded in tree importance. |
| `si` | 0.010752 | Y | Y | Y | Raw numeric input | Silicon | Affects CE, graphitization, shrinkage, and chemistry-stability features when bridged. |
| `c` | 0.010402 | Y | Y | Y | Raw numeric input | Carbon | Affects CE, C/Si ratio, and chemistry-stability features when bridged. |
| `sg_pigaddition_1` | 0.010399 | Y | Y | Y | Raw numeric input | SG pig addition 1 | Learned by classifier; direction not encoded in tree importance. |
| `rr_1` | 0.010311 | Y | Y | Y | Raw numeric input | Return/revert ratio 1 | Learned by classifier; direction not encoded in tree importance. |
| `n` | 0.010036 | Y | Y | Y | Raw numeric input | Nitrogen | Gas-risk logic uses nitrogen when bridged to `n__`; high N contributes to blowhole risk. |
| `feat_temp_loss` | 0.009802 | Y | Y | Y | `tapping_temp - pouring_temp`; `stage2_feature_engineering.py` | Temperature loss | Higher loss flags ladle heat loss/cold-pour risk in rules. |
| `zn` | 0.009163 | Y | Y | Y | Raw numeric input | Zinc | Learned by classifier; direction not encoded in tree importance. |
| `tapped_wt` | 0.008271 | Y | Y | Y | Raw numeric input | Tapped weight | Learned by classifier; direction not encoded in tree importance. |
| `fsm` | 0.008021 | Y | Y | Y | Raw numeric input | FSM additive | Learned by classifier; FSM dose logic exists for `fsmaddition_mt`. |
| `co` | 0.007936 | Y | Y | Y | Raw numeric input | Cobalt | Learned by classifier; direction not encoded in tree importance. |
| `rr` | 0.007611 | Y | Y | Y | Raw numeric input | Return/revert ratio | Learned by classifier; direction not encoded in tree importance. |
| `b` | 0.007394 | Y | Y | Y | Raw numeric input | Boron | Learned by classifier; direction not encoded in tree importance. |
| `sn` | 0.006995 | Y | Y | Y | Raw numeric input | Tin | Learned by classifier; direction not encoded in tree importance. |
| `lcrca_addition_1` | 0.006161 | Y | Y | Y | Raw numeric input | LCRCA addition 1 | Learned by classifier; CRCA charge is also used by gas-risk logic via `crca`. |
| `zr` | 0.005732 | Y | Y | Y | Raw numeric input | Zirconium | Learned by classifier; direction not encoded in tree importance. |
| `sg_pig` | 0.005544 | Y | Y | Y | Raw numeric input | SG pig | Learned by classifier; gas-risk comments mention SG Pig moisture as porosity driver. |
| `as` | 0.005537 | Y | Y | Y | Raw numeric input | Arsenic | Learned by classifier; direction not encoded in tree importance. |
| `ultraseed` | 0.005134 | Y | Y | Y | Raw numeric input | Ultraseed inoculant | Learned by classifier; direction not encoded in tree importance. |
| `graphite` | 0.004732 | Y | Y | Y | Raw numeric input | Graphite addition | Carbon/CE adjustment is referenced by CE rules; raw direction learned by classifier. |
| `low_re_fsm` | 0.004286 | Y | Y | Y | Raw numeric input | Low rare-earth FSM | Learned by classifier; direction not encoded in tree importance. |
| `crca` | 0.004231 | Y | Y | Y | Raw input; used in `feat_gas_risk_index` | CRCA charge | CRCA above 50 contributes to gas porosity risk in feature logic. |
| `silicon_addition_2` | 0.003437 | Y | Y | Y | Raw numeric input | Silicon addition 2 | Learned by classifier; silicon chemistry drives several engineered features when present. |
| `pb` | 0.003300 | Y | Y | Y | Raw numeric input | Lead | Learned by classifier; direction not encoded in tree importance. |
| `fe` | 0.003080 | Y | Y | Y | Raw numeric input | Iron | Learned by classifier; direction not encoded in tree importance. |
| `preseed` | 0.002754 | Y | Y | Y | Raw numeric input | Preseed inoculant | Learned by classifier; direction not encoded in tree importance. |
| `ce` | 0.002581 | Y | Y | Y | Raw CE input; copied to `feat_ce_calculated` if C/Si/P unavailable | Carbon equivalent | CE rules classify hypo/hypereutectic risk. |
| `feat_ce_calculated` | 0.002327 | Y | Y | Y | `c_ + (si_ + p__) / 3`, else `ce`; `stage2_feature_engineering.py` | Carbon equivalent feature | Off-target CE drives shrinkage/hard spots or graphite flotation/surface defects in rules. |
| `sorel_addition` | 0.002175 | Y | Y | Y | Raw numeric input | Sorel addition | Learned by classifier; direction not encoded in tree importance. |
| `sorel_total` | 0.001773 | Y | Y | Y | Raw numeric input | Sorel total | Learned by classifier; direction not encoded in tree importance. |
| `feat_ce_optimal` | 0.000416 | Y | Y | Y | `1` when CE is 4.2 to 4.3, else `0` | Eutectic CE zone flag | Optimal CE zone is expected by code comments to be favorable. |
| `feat_temp_loss_risk` | 0.000146 | Y | Y | Y | `1` when `feat_temp_loss > 80` | High temperature-loss flag | High loss is linked to cold shuts, mis-runs, and ladle heat loss. |
| `feat_ce_hypo_risk` | 0.000079 | Y | Y | Y | `1` when CE < 4.2 | Hypoeutectic CE risk | Code links low CE to micro-shrinkage and hard spots. |
| `feat_gas_risk_index` | 0.000050 | Y | Y | Y | `2*(n__ > 0.008) + 1*(mg_ > 0.055) + 0.5*(crca > 50)` | Gas porosity index | High N, high Mg, and CRCA moisture/charge risk contribute to blowholes or Mg vapor pockets. |
| `feat_shrinkage_risk_index` | 0.000021 | Y | Y | Y | `1.5*(CE < 4.2) + 1*(si_ < 2.0) + 0.5*(pouring_temp > 1420)` | Shrinkage risk index | Code links hypoeutectic CE, low Si, and high pouring temp to reduced graphite expansion/shrinkage risk. |
| `feat_ce_hyper_risk` | 0.000005 | Y | Y | Y | `1` when CE > 4.3 | Hypereutectic CE risk | Code links high CE to graphite flotation and surface defects. |
| `last_heel_metal` | 0.000000 | Y | Y | Y | Raw numeric input | Last heel metal | Learned importance is zero; no direct rule rationale found for this exact column. |
| `sg_pigaddition_2` | 0.000000 | Y | Y | Y | Raw numeric input | SG pig addition 2 | Learned importance is zero; no explicit rule rationale found. |
| `heel` | 0.000000 | Y | Y | Y | Raw numeric input | Heel | Heel-ratio logic says high heel can carry over S, N, and trace elements when ratio inputs are available. |
| `cs` | 0.000000 | Y | Y | Y | Raw numeric input | CS addition | Learned importance is zero; no explicit rule rationale found. |
| `al` | 0.000000 | Y | Y | Y | Raw numeric input | Aluminium | Oxidation logic uses aluminium when bridged to `al__`; this raw feature has zero classifier importance. |
| `high_re_fsm` | 0.000000 | Y | Y | Y | Raw numeric input | High rare-earth FSM | Learned importance is zero; no explicit rule rationale found. |
| `mn` | 0.000000 | Y | Y | Y | Raw numeric input | Manganese | Mn/S and graphitization rules use manganese when available as `mn_`; this raw feature has zero classifier importance. |
| `zero_re_fsm` | 0.000000 | Y | Y | Y | Raw numeric input | Zero rare-earth FSM | Learned importance is zero; no explicit rule rationale found. |
| `superseed` | 0.000000 | Y | Y | Y | Raw numeric input | Superseed inoculant | Learned importance is zero; no explicit rule rationale found. |
| `reseed` | 0.000000 | Y | Y | Y | Raw numeric input | Reseed inoculant | Learned importance is zero; no explicit rule rationale found. |
| `feat_pouring_stability` | 0.000000 | Y | Y | Y | `-1` if pouring temp < 1300; `1` if > 1500; else `0` | Pouring temperature stability | Off-range pouring temperature is coded as cold-pour or oxidation risk. |
| `feat_temp_loss_low` | 0.000000 | Y | Y | Y | `1` when `feat_temp_loss < 20` | Low temperature-loss flag | Code comments describe low loss as possible inaccurate measurement or very short ladle time. |
| `feat_pouring_risk` | 0.000000 | Y | Y | Y | `1` when `feat_pouring_stability != 0` | Pouring temperature out-of-range flag | Flags cold or overheated pouring conditions. |
| `feat_oxidation_risk` | 0.000000 | Y | Y | Y | `(al__ > 0.02) + 0.5*(pouring_temp > 1450)` | Oxidation risk index | Code links high Al and high pouring temp to oxide inclusions/subsurface pinholes. |
| `feat_white_iron_risk` | 0.000000 | Y | Y | Y | `1` when `feat_graphitization_index < 0` | White iron risk flag | Negative graphitization tendency indicates carbide/white-iron risk. |
| `feat_graphitization_index` | 0.000000 | Y | Y | Y | `2*si_ + 5*ce_ - 3.5*cr_ - 0.3*mn_ - 3*mo_` | Graphitization tendency | Higher is graphitic tendency; lower is carbide/white-iron tendency per code comments. |
| `feat_chemistry_instability` | 0.000000 | Y | Y | Y | Sum of deviations outside target ranges for C, Si, Mn, Mg | Chemistry instability | Larger value means farther from ideal chemistry range. |
| `feat_chemistry_ok` | 0.000000 | Y | Y | Y | `1` when `feat_chemistry_instability < 0.1` | Chemistry OK flag | Indicates chemistry is close to coded target ranges. |
| `fsmaddition_mt` | N/A |  | Y | Y | Raw numeric input; also used by optional `feat_fsm_s_index` and `feat_fsm_undertreat_risk` formulas | FSM addition per metric ton | Used by unsupervised PCA/anomaly models; supervised classifier explicitly excludes it. |

# Parameters Most Responsible For HOLD Decisions

These are rule-traced HOLD drivers, not inferred correlations.

| Driver | Source columns/features | Code threshold/path | Resulting decision behavior |
|---|---|---|---|
| High defect probability | `defect_prob` | `interpretation_rules.py`: `defect_prob >= 0.50`; `dashboard/risk_scoring.py`: `defect_prob >= 0.50` | Escalates to HIGH risk and HOLD. |
| Elevated anomaly score | `anomaly_score` | `interpretation_rules.py`: `anomaly_score >= 0.55`; `dashboard/risk_scoring.py`: `anomaly_score >= 0.60` | Escalates to HIGH risk and HOLD. |
| Hypoeutectic carbon equivalent | `feat_ce_calculated` or `ce` | `CE < 4.20` | Escalates to HIGH risk and HOLD; recommendation is to increase carbon/graphite/carburiser additions. |
| Cold pouring temperature | `pouring_temp` | `pouring_temp < 1290` | Escalates to HIGH risk and HOLD due to incomplete filling/cold shuts. |
| Elevated sulfur | `s_` | `s_ > 0.015` and `<= 0.025` | Escalates to HIGH risk and HOLD; recommendation is FSM/re-treatment monitoring. |
| Risky cluster history | `cluster`, `defect`, `defect_prob` | Cluster defect rate >= 0.22 in `dashboard/risk_scoring.py` | Escalates to HIGH risk and HOLD because the process group has elevated historical risk. |

# Parameters Most Responsible For STOP Decisions

| Driver | Source columns/features | Code threshold/path | Resulting decision behavior |
|---|---|---|---|
| Critical defect probability | `defect_prob` | `defect_prob >= 0.75` | STOP and CRITICAL risk. |
| Critical anomaly score | `anomaly_score` | `dashboard/risk_scoring.py`: `anomaly_score >= 0.80`; `interpretation_rules.py`: `anomaly_score >= 0.80` | STOP and CRITICAL risk for radically different process pattern. |
| Critical sulfur | `s_` | `s_ > 0.025` | STOP and CRITICAL risk due to severe nodularisation failure. |
| Severe low Mg recovery | `mg_recovery_` | `mg_recovery_ < 0.35` in rules | STOP and CRITICAL risk; code cites flake/vermicular graphite risk. |
| Critical cluster history | `cluster`, `defect`, `defect_prob` | Cluster defect rate >= 0.40 in `dashboard/risk_scoring.py` | STOP/CRITICAL escalation due to historical defect rate in process cluster. |

# Parameters Most Responsible For Critical Risk Classification

| Driver | Source columns/features | Critical threshold/path | Notes |
|---|---|---|---|
| Defect probability | `defect_prob` from classifier | >= 0.75 | Direct CRITICAL risk in both QA rules and unified risk scoring. |
| Anomaly score | `anomaly_score` from Isolation Forest/LOF or dashboard inference | >= 0.80 | Direct CRITICAL risk in unified scoring and rules. |
| Sulfur | `s_` | > 0.025 | Direct CRITICAL risk because sulfur depletes Mg treatment and degrades nodularisation. |
| Mg recovery | `mg_recovery_` | < 0.35 | Direct CRITICAL risk in QA rules. Note: current saved classifier feature schema contains `mg_recovery`, not `mg_recovery_`; upload bridging can create `mg_recovery_` for rules/features. |
| Cluster historical defect rate | `cluster` plus `defect` or `defect_prob` | >= 0.40 | Direct CRITICAL risk in unified scoring. |

# Executive Summary For Metallurgy Team

The strongest saved classifier drivers are `sb_1`, `barinoc`, `pouring_wt`, `silicon_addition_1`, `cr`, `ti`, and `pouring_temp`. These rankings come from the persisted Gradient Boosting model, not from manual interpretation.

The strongest rule-based HOLD/STOP drivers are different: `defect_prob`, `anomaly_score`, sulfur (`s_`), Mg recovery (`mg_recovery_`), carbon equivalent (`ce` / `feat_ce_calculated`), pouring temperature, and cluster historical defect rate. These directly change `risk_level` and `recommendation` through `interpretation_rules.py` and `dashboard/risk_scoring.py`.

The ML system uses three feature sets:

- Supervised defect classifier: 74 numeric features from `models/feature_names.json`.
- PCA/KMeans clustering: the same features plus `fsmaddition_mt`.
- Isolation Forest anomaly model: the same 75-feature set as PCA/KMeans.

The engineered metallurgical features actually present in the saved feature schemas are CE-zone flags, temperature-loss features, pouring-temperature risk flags, oxidation risk, graphitization/white-iron risk, shrinkage risk, gas risk, and chemistry-instability flags. Some stage-2 functions can create additional features (`feat_mg_recovery_risk`, `feat_mn_s_ratio`, `feat_sulfur_risk`, `feat_fsm_s_index`, etc.) when their canonical source columns exist, but those features are not in the current saved classifier/PCA/Isolation Forest feature schemas and therefore are not listed as current model inputs.
