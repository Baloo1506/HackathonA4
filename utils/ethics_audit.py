"""utils/ethics_audit.py — AIF360 Ethics & Fairness Audit"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    _AIF_AVAILABLE = True
except ImportError:
    _AIF_AVAILABLE = False
    print("[Ethics] AIF360 not installed. pip install aif360")

PROTECTED_ATTRS = {
    "Sex": {"privileged": [{"Sex": 1}], "unprivileged": [{"Sex": 0}]},
    "RaceDesc": {"privileged": [{"RaceDesc": 0}], "unprivileged": [{"RaceDesc": 1}]},
}
FAVORABLE_LABEL = 0
UNFAVORABLE_LABEL = 1

def _make_bld(df_combined, label_col, protected_col):
    return BinaryLabelDataset(
        df=df_combined[[protected_col, label_col]],
        label_names=[label_col],
        protected_attribute_names=[protected_col],
        favorable_label=FAVORABLE_LABEL,
        unfavorable_label=UNFAVORABLE_LABEL,
    )

def _verdict(di, diff):
    if di < 0.8 or abs(diff) > 0.1:
        return "Potential bias — consider debiasing"
    if di < 0.9 or abs(diff) > 0.05:
        return "Marginal — monitor carefully"
    return "Acceptable fairness"

def audit_dataset(X, y, df_sensitive):
    if not _AIF_AVAILABLE:
        return {"error": "AIF360 not installed"}
    results = {}
    for attr in PROTECTED_ATTRS:
        if attr not in df_sensitive.columns:
            continue
        df_combined = df_sensitive[[attr]].copy()
        df_combined["label"] = y.values
        try:
            bld = _make_bld(df_combined, "label", attr)
            metric = BinaryLabelDatasetMetric(bld, privileged_groups=PROTECTED_ATTRS[attr]["privileged"], unprivileged_groups=PROTECTED_ATTRS[attr]["unprivileged"])
            di = metric.disparate_impact()
            spd = metric.statistical_parity_difference()
            results[attr] = {"disparate_impact": round(di, 4), "statistical_parity_difference": round(spd, 4), "bias_detected": di < 0.8 or abs(spd) > 0.1, "verdict": _verdict(di, spd)}
            print(f"[Ethics] {attr}: DI={di:.3f}, SPD={spd:.3f} -> {results[attr]['verdict']}")
        except Exception as e:
            results[attr] = {"error": str(e)}
    return results

def apply_reweighing(X, y, df_sensitive, attr="Sex"):
    if not _AIF_AVAILABLE or attr not in df_sensitive.columns:
        return X, y, np.ones(len(y))
    df_combined = df_sensitive[[attr]].copy()
    df_combined["label"] = y.values
    bld = _make_bld(df_combined, "label", attr)
    rw = Reweighing(unprivileged_groups=PROTECTED_ATTRS[attr]["unprivileged"], privileged_groups=PROTECTED_ATTRS[attr]["privileged"])
    bld_rw = rw.fit_transform(bld)
    weights = bld_rw.instance_weights
    print(f"[Ethics] Reweighing applied for '{attr}'. Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    return X, y, weights

def audit_predictions(X, y_true, y_pred, df_sensitive):
    if not _AIF_AVAILABLE:
        return {"error": "AIF360 not installed"}
    results = {}
    for attr in PROTECTED_ATTRS:
        if attr not in df_sensitive.columns:
            continue
        df_comb_true = df_sensitive[[attr]].copy(); df_comb_true["label"] = y_true.values
        df_comb_pred = df_sensitive[[attr]].copy(); df_comb_pred["label"] = y_pred.astype(int)
        try:
            bld_true = _make_bld(df_comb_true, "label", attr)
            bld_pred = _make_bld(df_comb_pred, "label", attr)
            cm = ClassificationMetric(bld_true, bld_pred, unprivileged_groups=PROTECTED_ATTRS[attr]["unprivileged"], privileged_groups=PROTECTED_ATTRS[attr]["privileged"])
            eod = cm.equal_opportunity_difference(); di = cm.disparate_impact()
            results[attr] = {"equal_opportunity_difference": round(eod, 4), "disparate_impact": round(di, 4), "bias_detected": abs(eod) > 0.1 or di < 0.8, "verdict": _verdict(di, eod)}
        except Exception as e:
            results[attr] = {"error": str(e)}
    return results

def fairness_summary_df(audit_results):
    rows = []
    for attr, metrics in audit_results.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        rows.append({"Protected Attribute": attr, "Disparate Impact": metrics.get("disparate_impact","N/A"), "SPD/EOD": metrics.get("statistical_parity_difference", metrics.get("equal_opportunity_difference","N/A")), "Verdict": metrics.get("verdict","")})
    return pd.DataFrame(rows)
