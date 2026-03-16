"""
app.py  —  TalentGuard AI  —  Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(
    page_title="TalentGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0f1117;}
[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #2a2f3e;}
.metric-card{background:#1a1f2e;border:1px solid #2a2f3e;border-radius:12px;
  padding:20px 24px;text-align:center;}
.metric-value{font-size:2.2rem;font-weight:700;color:#e2e8f0;}
.metric-label{font-size:.85rem;color:#94a3b8;margin-top:4px;}
.risk-high{color:#f87171;font-weight:600;}
.risk-medium{color:#fbbf24;font-weight:600;}
.risk-low{color:#34d399;font-weight:600;}
.suggestion-box{background:#1e2a3a;border-left:3px solid #60a5fa;
  padding:12px 16px;border-radius:0 8px 8px 0;
  font-size:.9rem;color:#cbd5e1;margin-top:12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_artefacts():
    base = Path("models")
    art = {}
    if not (base / "saved_model.joblib").exists():
        return art
    import joblib
    art["model"]     = joblib.load(base / "saved_model.joblib")
    art["feat_cols"] = joblib.load(base / "feature_cols.joblib")
    art["emp_ids"]   = joblib.load(base / "employee_ids.joblib")
    art["X"]         = pd.read_parquet(base / "X_full.parquet")
    art["y"]         = pd.read_parquet(base / "y_full.parquet")["Termd"]
    if (base / "shap_values.joblib").exists():
        art["shap"] = joblib.load(base / "shap_values.joblib")
    if (base / "metrics.json").exists():
        art["metrics"] = json.load(open(base / "metrics.json"))
    if (base / "fairness_audit.json").exists():
        art["audit"] = json.load(open(base / "fairness_audit.json"))
    from models.attrition_model import predict_risk
    art["risk_df"] = predict_risk(art["model"], art["X"], pd.Series(art["emp_ids"]))
    return art


artefacts  = load_artefacts()
model_ready = "model" in artefacts

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ TalentGuard AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview", "👥 Risk List", "🔍 Employee",
        "⚖️ Ethics Audit", "📋 Model Card"
    ], label_visibility="collapsed")
    st.markdown("---")
    if model_ready:
        m = artefacts.get("metrics", {})
        st.markdown("**Model performance**")
        st.metric("AUC-ROC",  m.get("auc","—"))
        st.metric("Accuracy", m.get("accuracy","—"))
        st.metric("CV AUC",   m.get("cv_auc_mean","—"))
    else:
        st.warning("No model found.\nRun `python train.py` first.")

# ============================================================================
# PAGE 1 — Overview
# ============================================================================
if page == "📊 Overview":
    st.title("Attrition Risk Dashboard")
    st.caption("Explainable · Ethical · Frugal  |  TalentGuard AI v1.0")
    if not model_ready:
        st.info("Run `python train.py --data data/HRDataset_v14.csv` to generate predictions.")
        st.stop()

    risk_df = artefacts["risk_df"]
    high   = (risk_df["RiskLabel"]=="High").sum()
    medium = (risk_df["RiskLabel"]=="Medium").sum()
    low    = (risk_df["RiskLabel"]=="Low").sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(risk_df)}</div><div class="metric-label">Total Employees</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value risk-high">{high}</div><div class="metric-label">High Risk</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value risk-medium">{medium}</div><div class="metric-label">Medium Risk</div></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value risk-low">{low}</div><div class="metric-label">Low Risk</div></div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    import plotly.express as px
    with col_l:
        st.subheader("Risk distribution")
        fig = px.pie(names=["High","Medium","Low"], values=[high,medium,low],
            color=["High","Medium","Low"],
            color_discrete_map={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"},hole=0.55)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Risk score distribution")
        fig2 = px.histogram(risk_df, x="RiskScore", nbins=20, color_discrete_sequence=["#60a5fa"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)

    if "shap" in artefacts:
        st.subheader("Top attrition drivers  (global SHAP importance)")
        from models.attrition_model import top_shap_features
        top_f = top_shap_features(artefacts["shap"], artefacts["feat_cols"], n=10)
        fig3  = px.bar(top_f.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h", color_discrete_sequence=["#818cf8"])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8")
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# PAGE 2 — Risk List
# ============================================================================
elif page == "👥 Risk List":
    st.title("Employee Risk List")
    st.caption("Sorted by predicted attrition probability. Anonymized IDs only.")
    if not model_ready:
        st.info("Run the training pipeline first."); st.stop()

    risk_df = artefacts["risk_df"].copy()
    c1,c2 = st.columns([1,2])
    with c1:
        filter_risk = st.selectbox("Filter", ["All","High","Medium","Low"])
    with c2:
        search = st.text_input("Search employee ID")

    if filter_risk != "All":
        risk_df = risk_df[risk_df["RiskLabel"]==filter_risk]
    if search:
        risk_df = risk_df[risk_df["Employee_Name"].str.contains(search, case=False)]

    risk_df = risk_df.sort_values("RiskScore", ascending=False).reset_index(drop=True)
    risk_df["Risk %"] = (risk_df["RiskScore"]*100).round(1).astype(str)+"%"
    st.dataframe(
        risk_df[["Employee_Name","RiskLabel","Risk %"]].rename(
            columns={"Employee_Name":"Employee ID","RiskLabel":"Risk Level"}),
        use_container_width=True, height=520)

# ============================================================================
# PAGE 3 — Individual Employee
# ============================================================================
elif page == "🔍 Employee":
    st.title("Individual Employee Analysis")
    st.caption("SHAP explanations — every prediction is transparent and auditable.")
    if not model_ready:
        st.info("Run the training pipeline first."); st.stop()

    risk_df   = artefacts["risk_df"]
    shap_vals = artefacts.get("shap")
    feat_cols = artefacts["feat_cols"]
    selected  = st.selectbox("Select employee", risk_df["Employee_Name"].tolist())
    idx       = risk_df["Employee_Name"].tolist().index(selected)
    row       = risk_df.iloc[idx]

    c1,c2,c3 = st.columns(3)
    c1.metric("Risk Score", f"{row['RiskScore']*100:.1f}%")
    c2.metric("Risk Level", row["RiskLabel"])
    c3.metric("Prediction", "Will leave" if row["Prediction"]==1 else "Will stay")

    if shap_vals is not None:
        st.subheader("Why this prediction? (SHAP)")
        from models.attrition_model import employee_shap_explanation, generate_retention_suggestion
        import plotly.express as px
        exp_df = employee_shap_explanation(shap_vals, feat_cols, idx, n=8)
        exp_df["Direction"] = exp_df["SHAP"].apply(lambda v: "Increases risk" if v>0 else "Reduces risk")
        fig = px.bar(exp_df.sort_values("SHAP"), x="SHAP", y="Feature", orientation="h",
            color="Direction",
            color_discrete_map={"Increases risk":"#f87171","Reduces risk":"#34d399"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", xaxis_title="SHAP value (impact on attrition probability)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended HR actions")
        suggestion = generate_retention_suggestion(exp_df)
        st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
    else:
        st.warning("SHAP values not found — re-run training with `shap` installed.")

# ============================================================================
# PAGE 4 — Ethics Audit
# ============================================================================
elif page == "⚖️ Ethics Audit":
    st.title("Ethics & Fairness Audit")
    st.caption("IBM AIF360 — Disparate Impact across Sex and RaceDesc")
    audit = artefacts.get("audit")
    if not audit:
        st.info("Re-run `python train.py` with sensitive attribute columns present."); st.stop()

    for result in audit:
        bias = result.get("bias_detected")
        label = "⚠️ Bias detected" if bias else "✅ No bias detected"
        with st.expander(f"Audit: {result.get('sensitive_attribute','?')} — {label}", expanded=True):
            st.markdown(result.get("narrative",""))
            st.markdown("---")
            cols = st.columns(4)
            cols[0].metric("Disparate Impact",       result.get("disparate_impact","—"), help="Target ≥ 0.80")
            cols[1].metric("Stat. Parity Diff",      result.get("statistical_parity_diff","—"), help="Target |diff| ≤ 0.10")
            cols[2].metric("Equal Opportunity Diff", result.get("equal_opportunity_diff","—"))
            cols[3].metric("Avg. Odds Diff",         result.get("avg_odds_diff","—"))
            if bias:
                st.warning("Mitigation: call `reweigh_training_data()` in utils/fairness_audit.py and re-train.")

    st.info("Sensitive attributes are used **only** for this audit — excluded from prediction features. Compliant with EU AI Act Article 10.")

# ============================================================================
# PAGE 5 — Model Card
# ============================================================================
elif page == "📋 Model Card":
    st.title("Model Card & Data Card")
    st.caption("Transparency documentation — EU AI Act high-risk AI system")

    with st.expander("Model Card", expanded=True):
        st.markdown("""
| Field | Value |
|-------|-------|
| **Model name** | TalentGuard Attrition Predictor v1.0 |
| **Model type** | Random Forest Classifier (scikit-learn) |
| **Task** | Binary classification — predict employee attrition (Termd = 0/1) |
| **Features** | Salary, Engagement Survey, Satisfaction, Performance Score, Absences, Tenure, Sentiment scores, Department (OHE) |
| **Sensitive attributes** | Sex, RaceDesc — **excluded from features**, used for fairness audit only |
| **Explainability** | SHAP TreeExplainer — global + per-employee |
| **Fairness** | IBM AIF360 — Disparate Impact & Statistical Parity |
| **AI Act risk level** | High-risk (employment decisions) |
| **Intended use** | Support HR — human review mandatory before any action |
| **Out-of-scope** | Automated hiring/termination without oversight |
""")

    with st.expander("Data Card"):
        st.markdown("""
| Field | Value |
|-------|-------|
| **Source** | Huebner & Patalano HR Dataset (Kaggle, open source) |
| **Size** | ~400 synthetic employee records |
| **Anonymization** | Names/IDs SHA-256 hashed; email dropped; DOB → AgeBucket; hire date → TenureYears |
| **Sensitive data** | Sex, RaceDesc — retained only for fairness audit |
| **Text data** | Synthetic exit interviews + satisfaction survey text |
| **Known limitations** | Small dataset; synthetic text may not reflect real language |
| **GDPR** | No direct identifiers in any model artefact |
""")

    with st.expander("Responsible AI checklist"):
        st.markdown("""
- ✅ **Cybersecurity** — PII anonymized; AI Act risk classification performed
- ✅ **Ethics** — AIF360 fairness audit; sensitive attributes excluded from model
- ✅ **Frugality** — Lightweight Random Forest; no GPU required; CodeCarbon compatible
- ✅ **Explainability** — SHAP for every prediction; rule-based retention suggestions
- ✅ **Human oversight** — Dashboard informs HR; does not automate decisions
- ✅ **Transparency** — Model Card and Data Card documented and versioned
""")

    if artefacts.get("metrics"):
        with st.expander("Performance metrics"):
            st.json(artefacts["metrics"])
