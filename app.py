import os
import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same folder as this script, regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# LLM backend: "ollama" (local, free) or "openai"
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

RISK_REPORT_PATH = Path(__file__).parent / "risk_report_active (8).csv"
SHAP_PATH        = Path(__file__).parent / "shap_active_df.csv"
MODEL_PATH       = Path(__file__).parent / "calibrated_model.joblib"
DEPT_MEANS_PATH  = Path(__file__).parent / "dept_salary_means.joblib"

FEATURE_COLS = [
    "Salary", "EngagementSurvey", "EmpSatisfaction", "SpecialProjectsCount",
    "DaysLateLast30", "Absences", "age", "tenure_years", "salary_vs_dept_mean",
    "engagement_x_satisfaction", "Department_enc", "Position_enc",
    "MaritalDesc_enc", "CitizenDesc_enc", "RecruitmentSource_enc", "PerformanceScore_enc",
]

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

# LabelEncoder assigns indices in sorted order — hardcoded from training data
ENCODINGS = {
    "Department": sorted([
        "Admin Offices", "Executive Office", "IT/IS",
        "Production", "Sales", "Software Engineering",
    ]),
    "Position": sorted(set([
        "Accountant I", "Administrative Assistant", "Area Sales Manager",
        "BI Developer", "BI Director", "CIO", "Data Analyst", "Data Architect",
        "Database Administrator", "Director of Operations", "Director of Sales",
        "Enterprise Architect", "IT Director", "IT Manager - DB",
        "IT Manager - Infra", "IT Manager - Support", "IT Support",
        "Network Engineer", "President & CEO", "Principal Data Architect",
        "Production Manager", "Production Technician I", "Production Technician II",
        "Sales Manager", "Senior BI Developer", "Shared Services Manager",
        "Software Engineer", "Software Engineering Manager",
        "Sr. Accountant", "Sr. DBA", "Sr. Network Engineer",
    ])),
    "MaritalDesc":       sorted(["Divorced", "Married", "Separated", "Single", "Widowed"]),
    "CitizenDesc":       sorted(["Eligible NonCitizen", "Non-Citizen", "US Citizen"]),
    "RecruitmentSource": sorted([
        "CareerBuilder", "Diversity Job Fair", "Employee Referral",
        "Google Search", "Indeed", "LinkedIn",
        "On-line Web application", "Other", "Website",
    ]),
    "PerformanceScore":  sorted(["Exceeds", "Fully Meets", "Needs Improvement", "PIP"]),
}

# Risk thresholds from notebook (85th / 60th percentile of training scores)
HIGH_THRESHOLD = 0.121
MED_THRESHOLD  = 0.039
RISK_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


# ── Data / model loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_data():
    report = pd.read_csv(RISK_REPORT_PATH)
    shap   = pd.read_csv(SHAP_PATH).rename(columns={"Unnamed: 0": "anonymized_id"})
    return report, shap


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not DEPT_MEANS_PATH.exists():
        return None, None
    model      = joblib.load(MODEL_PATH)
    dept_means = joblib.load(DEPT_MEANS_PATH)
    return model, dept_means


# ── LLM helpers ────────────────────────────────────────────────────────────────

def build_prompt_existing(emp: dict, top_shap: list) -> str:
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


def build_prompt_simulated(inputs: dict, risk_level: str) -> str:
    lines = "\n".join(f"  - {k}: {v}" for k, v in inputs.items())
    return f"""You are a senior HR advisor specialising in employee retention.
Based on the employee profile below, write 3-4 concrete, empathetic, actionable recommendations
for the HR manager to reduce flight risk. Do NOT mention the risk score. Focus on proactive interventions.

Employee profile:
- Department: {inputs["Department"]}
- Position: {inputs["Position"]}
- Risk level: {risk_level}
- Key metrics:
{lines}

Format: short numbered list, one sentence each, practical and specific."""


def get_llm_client():
    if LLM_BACKEND == "ollama":
        # Ollama exposes an OpenAI-compatible API — no key needed
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=180.0)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def get_model_name():
    return OLLAMA_MODEL if LLM_BACKEND == "ollama" else "gpt-4o-mini"


def get_recommendation_existing(emp: dict, top_shap: list) -> str:
    client = get_llm_client()
    if not client:
        return "⚠️ OPENAI_API_KEY not found. Check your .env file."
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": build_prompt_existing(emp, top_shap)}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content


def get_recommendation_simulated(inputs: dict, risk_level: str) -> str:
    client = get_llm_client()
    if not client:
        return "⚠️ OPENAI_API_KEY not found. Check your .env file."
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": build_prompt_simulated(inputs, risk_level)}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HR Retention Dashboard", layout="wide", page_icon="👥")
st.title("👥 HR Retention Risk Dashboard")
st.caption("Model: Random Forest + SHAP  |  Advisory: Gemma 3 4B (local)  |  Data: anonymised")

report, shap_df = load_data()
model, dept_means = load_model()
SHAP_FEATURE_COLS = [c for c in shap_df.columns if c != "anonymized_id" and not c.endswith("_enc")]

tab1, tab2 = st.tabs(["📋 Active employees", "🔮 Simulate employee"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — existing active employees dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Sidebar filters (only affect tab1)
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

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High risk",   int((filtered["risk_level"] == "High").sum()))
    c2.metric("🟡 Medium risk", int((filtered["risk_level"] == "Medium").sum()))
    c3.metric("🟢 Low risk",    int((filtered["risk_level"] == "Low").sum()))

    # Table
    st.subheader(f"Employees ({len(filtered)} shown)")
    display = filtered[
        ["anonymized_id", "Department", "Position", "risk_score", "risk_level", "nlp_advisory"]
    ].copy()
    display["risk_score"] = display["risk_score"].map(lambda x: f"{x:.1%}")
    display["risk_level"] = display["risk_level"].map(lambda x: f"{RISK_ICON.get(x, '')} {x}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Employee detail
    st.divider()
    st.subheader("Employee detail & AI recommendation")

    selected_id = st.selectbox("Select employee", filtered["anonymized_id"].tolist())
    emp      = filtered[filtered["anonymized_id"] == selected_id].iloc[0].to_dict()
    shap_row = shap_df[shap_df["anonymized_id"] == selected_id]

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Department:** {emp['Department']}  \n**Position:** {emp['Position']}")
        st.metric("Risk score", f"{emp['risk_score']:.1%}")
        st.metric("Risk level", f"{RISK_ICON.get(emp['risk_level'], '')} {emp['risk_level']}")
        if emp["nlp_advisory"] and pd.notna(emp["nlp_advisory"]):
            st.warning(f"**NLP signals:** {emp['nlp_advisory']}")

    with right:
        st.markdown("**Key risk factors**")
        for i, col in enumerate(["top_reason_1", "top_reason_2", "top_reason_3"], 1):
            reason = emp.get(col, "")
            if reason and pd.notna(reason):
                label = FEATURE_LABELS.get(reason, reason.replace("_", " ").title())
                st.markdown(f"{i}. {label}")
        if emp.get("recommended_action") and pd.notna(emp["recommended_action"]):
            st.markdown("**Suggested actions**")
            for action in str(emp["recommended_action"]).split("|"):
                st.markdown(f"- {action.strip()}")

    if st.button("✨ Generate AI recommendation", type="primary", key="btn_existing"):
        if shap_row.empty:
            st.error("SHAP data not found for this employee.")
        else:
            with st.spinner("Generating recommendation..."):
                vals  = shap_row[SHAP_FEATURE_COLS].iloc[0].sort_values(ascending=False)
                top3  = list(vals.head(3).items())
                rec   = get_recommendation_existing(emp, top3)
            st.success("Recommendation ready")
            st.markdown(rec)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — simulate a new employee
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if model is None:
        st.warning(
            "Model files not found. Add this cell to your Colab notebook, run it, "
            "download the two `.joblib` files, and place them next to `app.py`:\n\n"
            "```python\n"
            "import joblib\n"
            "joblib.dump(best_model_calibrated, 'calibrated_model.joblib')\n"
            "dept_salary_means = df[df['EmploymentStatus'] != 'Terminated for Cause']"
            ".groupby('Department')['Salary'].mean()\n"
            "joblib.dump(dept_salary_means.to_dict(), 'dept_salary_means.joblib')\n"
            "print('Done.')\n"
            "```"
        )
    else:
        st.subheader("Fill in employee details")
        st.caption("Enter values to get an instant attrition risk prediction.")

        col_l, col_r = st.columns(2)

        with col_l:
            salary        = st.number_input("Salary ($)", 45000, 250000, 65000, step=1000)
            engagement    = st.slider("Engagement survey (1–5)", 1.0, 5.0, 3.5, step=0.1)
            satisfaction  = st.slider("Employee satisfaction (1–5)", 1, 5, 3)
            special_proj  = st.number_input("Special projects count", 0, 20, 1)
            days_late     = st.number_input("Days late last 30 days", 0, 30, 0)
            absences      = st.number_input("Number of absences", 0, 50, 5)

        with col_r:
            age           = st.number_input("Age", 18, 70, 35)
            tenure        = st.number_input("Tenure (years)", 0.0, 20.0, 3.0, step=0.5)
            department    = st.selectbox("Department",    ENCODINGS["Department"])
            position      = st.selectbox("Position",      ENCODINGS["Position"])
            marital       = st.selectbox("Marital status", ENCODINGS["MaritalDesc"])
            citizen       = st.selectbox("Citizenship",   ENCODINGS["CitizenDesc"])
            recruit_src   = st.selectbox("Recruitment source", ENCODINGS["RecruitmentSource"])
            perf_score    = st.selectbox("Performance score",  ENCODINGS["PerformanceScore"])

        if st.button("🔍 Predict risk", type="primary", key="btn_simulate"):
            # Compute derived features
            dept_key = department.strip()
            dept_mean = dept_means.get(dept_key, dept_means.get(department, salary))
            sal_vs_dept = salary / dept_mean if dept_mean else 1.0
            eng_x_sat   = engagement * satisfaction

            # Encode categoricals
            def enc(col, val):
                opts = ENCODINGS[col]
                val_stripped = val.strip()
                # Try exact match first, then stripped match
                if val in opts:
                    return opts.index(val)
                for i, o in enumerate(opts):
                    if o.strip() == val_stripped:
                        return i
                return 0  # fallback

            X = pd.DataFrame([{
                "Salary":                    salary,
                "EngagementSurvey":          engagement,
                "EmpSatisfaction":           satisfaction,
                "SpecialProjectsCount":      special_proj,
                "DaysLateLast30":            days_late,
                "Absences":                  absences,
                "age":                       age,
                "tenure_years":              tenure,
                "salary_vs_dept_mean":       sal_vs_dept,
                "engagement_x_satisfaction": eng_x_sat,
                "Department_enc":            enc("Department", department),
                "Position_enc":              enc("Position", position),
                "MaritalDesc_enc":           enc("MaritalDesc", marital),
                "CitizenDesc_enc":           enc("CitizenDesc", citizen),
                "RecruitmentSource_enc":     enc("RecruitmentSource", recruit_src),
                "PerformanceScore_enc":      enc("PerformanceScore", perf_score),
            }])[FEATURE_COLS]

            risk_score = model.predict_proba(X)[0, 1]
            risk_level = (
                "High"   if risk_score >= HIGH_THRESHOLD else
                "Medium" if risk_score >= MED_THRESHOLD  else
                "Low"
            )

            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Risk score", f"{risk_score:.1%}")
            r2.metric("Risk level", f"{RISK_ICON.get(risk_level, '')} {risk_level}")

            # Store result in session state so the LLM button works
            st.session_state["sim_result"] = {
                "risk_level": risk_level,
                "inputs": {
                    "Department":           department,
                    "Position":             position,
                    "Salary":               f"${salary:,}",
                    "Engagement survey":    engagement,
                    "Satisfaction":         satisfaction,
                    "Special projects":     special_proj,
                    "Days late (30d)":      days_late,
                    "Absences":             absences,
                    "Age":                  age,
                    "Tenure (years)":       tenure,
                    "Performance score":    perf_score,
                    "Salary vs dept avg":   f"{sal_vs_dept:.2f}x",
                },
            }

        # Show recommendation button once a prediction has been made
        if "sim_result" in st.session_state:
            if st.button("✨ Generate AI recommendation", type="secondary", key="btn_sim_llm"):
                res = st.session_state["sim_result"]
                with st.spinner("Generating recommendation..."):
                    rec = get_recommendation_simulated(res["inputs"], res["risk_level"])
                st.success("Recommendation ready")
                st.markdown(rec)
