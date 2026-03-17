import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same folder as this script, regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

RISK_REPORT_PATH = "risk_report_active (8).csv"
SHAP_PATH = "shap_active_df.csv"

FEATURE_LABELS = {
    "Salary":                    "Salary level",
    "EngagementSurvey":          "Engagement survey score",
    "EmpSatisfaction":           "Employee satisfaction",
    "SpecialProjectsCount":      "Number of special projects",
    "DaysLateLast30":            "Days late in last 30 days",
    "Absences":                  "Number of absences",
    "age":                       "Employee age",
    "tenure_years":              "Years at company",
    "salary_vs_dept_mean":       "Salary vs. dept. average",
    "engagement_x_satisfaction": "Engagement × satisfaction composite",
}

RISK_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


@st.cache_data
def load_data():
    report = pd.read_csv(RISK_REPORT_PATH)
    shap = pd.read_csv(SHAP_PATH).rename(columns={"Unnamed: 0": "anonymized_id"})
    return report, shap


def build_prompt(emp: dict, top_shap: list) -> str:
    drivers = "\n".join(
        f"  - {FEATURE_LABELS.get(feat, feat)}: impact {val:+.4f}"
        for feat, val in top_shap
    )
    advisory = emp["nlp_advisory"] if pd.notna(emp["nlp_advisory"]) and emp["nlp_advisory"] else "None"
    return f"""You are a senior HR advisor specialising in employee retention.
Based on the anonymised data below, write 3-4 concrete, empathetic, actionable recommendations
for the HR manager. Do NOT mention the risk score. Focus on proactive interventions.

Employee profile:
- Department: {emp["Department"]}
- Position: {emp["Position"]}
- Risk level: {emp["risk_level"]}
- Top model risk drivers (SHAP values):
{drivers}
- Additional signals from written feedback / requests: {advisory}

Format: short numbered list, one sentence each, practical and specific."""


def get_recommendation(emp: dict, top_shap: list) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(emp, top_shap)}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Retention Dashboard", layout="wide", page_icon="👥")

st.title("👥 HR Retention Risk Dashboard")
st.caption("Model: Random Forest + SHAP  |  Advisory: GPT-4o-mini  |  Data: anonymised")

report, shap_df = load_data()
FEATURE_COLS = [c for c in shap_df.columns if c != "anonymized_id" and not c.endswith("_enc")]

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    risk_filter = st.multiselect(
        "Risk level", ["High", "Medium", "Low"], default=["High", "Medium"]
    )
    dept_filter = st.multiselect(
        "Department",
        sorted(report["Department"].dropna().unique()),
        default=sorted(report["Department"].dropna().unique()),
    )

filtered = report[
    report["risk_level"].isin(risk_filter) & report["Department"].isin(dept_filter)
].sort_values("risk_score", ascending=False)

# ── Summary metrics ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("🔴 High risk",   int((filtered["risk_level"] == "High").sum()))
c2.metric("🟡 Medium risk", int((filtered["risk_level"] == "Medium").sum()))
c3.metric("🟢 Low risk",    int((filtered["risk_level"] == "Low").sum()))

# ── Employee table ─────────────────────────────────────────────────────────────
st.subheader(f"Employees ({len(filtered)} shown)")

display = filtered[
    ["anonymized_id", "Department", "Position", "risk_score", "risk_level", "nlp_advisory"]
].copy()
display["risk_score"] = display["risk_score"].map(lambda x: f"{x:.1%}")
display["risk_level"] = display["risk_level"].map(lambda x: f"{RISK_ICON.get(x, '')} {x}")

st.dataframe(display, use_container_width=True, hide_index=True)

# ── Employee detail ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Employee detail & AI recommendation")

selected_id = st.selectbox("Select employee", filtered["anonymized_id"].tolist())
emp = filtered[filtered["anonymized_id"] == selected_id].iloc[0].to_dict()
shap_row = shap_df[shap_df["anonymized_id"] == selected_id]

left, right = st.columns(2)

with left:
    st.markdown(f"**Department:** {emp['Department']}  \n**Position:** {emp['Position']}")
    st.metric("Risk score", f"{emp['risk_score']:.1%}")
    st.metric("Risk level", f"{RISK_ICON.get(emp['risk_level'], '')} {emp['risk_level']}")
    if emp["nlp_advisory"] and pd.notna(emp["nlp_advisory"]):
        st.warning(f"**NLP signals:** {emp['nlp_advisory']}")

with right:
    st.markdown("**Top SHAP risk drivers**")
    if not shap_row.empty:
        vals = shap_row[FEATURE_COLS].iloc[0].sort_values(ascending=False)
        top5 = vals.head(5)
        for feat, val in top5.items():
            label = FEATURE_LABELS.get(feat, feat)
            arrow = "▲" if val > 0 else "▼"
            color = "red" if val > 0 else "green"
            st.markdown(f":{color}[{arrow}] **{label}** — {val:+.4f}")

# ── LLM recommendation ─────────────────────────────────────────────────────────
if st.button("✨ Generate AI recommendation", type="primary"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Check your .env file.")
    elif shap_row.empty:
        st.error("SHAP data not found for this employee.")
    else:
        with st.spinner("Generating recommendation..."):
            vals = shap_row[FEATURE_COLS].iloc[0].sort_values(ascending=False)
            top3 = list(vals.head(3).items())
            rec = get_recommendation(emp, top3)
        st.success("Recommendation ready")
        st.markdown(rec)
