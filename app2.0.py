import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Synchronous liver-only metastasis risk calculator",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 1.6rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
    }

    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    h1 {
        color: #1f2937;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.35rem;
    }

    h2, h3 {
        color: #1f2937;
        font-weight: 800 !important;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.02rem;
        margin-bottom: 1.4rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 1.35rem 1.35rem 1.1rem 1.35rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    .result-card {
        background: linear-gradient(180deg, #fffafa 0%, #ffffff 100%);
        border: 1px solid #f2d3d3;
        border-radius: 22px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }

    .prob-label {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }

    .prob-value {
        font-size: 3rem;
        line-height: 1;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.7rem;
        letter-spacing: -0.8px;
    }

    .risk-high {
        display: inline-block;
        padding: 0.52rem 0.95rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #b91c1c;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .risk-mid {
        display: inline-block;
        padding: 0.52rem 0.95rem;
        border-radius: 999px;
        background: #fef3c7;
        color: #92400e;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .risk-low {
        display: inline-block;
        padding: 0.52rem 0.95rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .note-text {
        color: #6b7280;
        font-size: 0.98rem;
        line-height: 1.7;
    }

    .footer-box {
        margin-top: 1.4rem;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: #6b7280;
        font-size: 0.93rem;
        line-height: 1.7;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: #f9fafb !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 0.72rem 1.55rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 8px 18px rgba(220, 38, 38, 0.20);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white !important;
    }

    .stButton > button:focus:not(:active) {
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 18px rgba(220, 38, 38, 0.20) !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\linzj\Desktop\SEER_CRC_saved_models_and_preds\models\best_xgb_pipeline.pkl")

model = load_model()

# -------------------------
# Header
# -------------------------
st.title("Online calculator for synchronous liver-only metastasis risk")
st.markdown(
    '<div class="subtitle">A web-based calculator derived from the final XGBoost model for individualized estimation of synchronous liver-only metastasis risk.</div>',
    unsafe_allow_html=True
)

# -------------------------
# Main layout
# -------------------------
left_col, right_col = st.columns([1.06, 1], gap="large")

prob = None
summary_df = None

with left_col:
    st.subheader("Clinical inputs")

    age = st.slider("Age (years)", 18, 100, 65)

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    race = st.selectbox(
        "Race",
        ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"]
    )

    primary_site = st.selectbox(
        "Primary site",
        ["Colon", "Rectum"]
    )

    grade = st.selectbox(
        "Grade",
        ["Grade I", "Grade II", "Grade III", "Grade IV"]
    )

    t_stage = st.selectbox(
        "T stage",
        ["Tis-T1", "T2", "T3", "T4", "Tx"]
    )

    n_stage = st.selectbox(
        "N stage",
        ["N0", "N1", "N2", "Nx"]
    )

    cea = st.selectbox(
        "CEA",
        ["Negative", "Positive"]
    )

    predict_btn = st.button("Predict")

with right_col:
    st.subheader("Prediction result")

    if predict_btn:
        input_df = pd.DataFrame([{
            "Sex": sex,
            "Race recode (W, B, AI, API)": race,
            "Age": age,
            "Grade": grade,
            "T_stage": t_stage,
            "N_stage": n_stage,
            "Primary_Site": primary_site,
            "CEA": cea
        }])

        prob = model.predict_proba(input_df)[0, 1]

        st.markdown('<div class="prob-label">Predicted risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prob-value">{prob:.1%}</div>', unsafe_allow_html=True)

        if prob < 0.10:
            st.markdown('<div class="risk-low">Low-risk group</div>', unsafe_allow_html=True)
            interpret = (
                "The entered clinicopathological profile is associated with a low estimated probability "
                "of synchronous liver-only metastasis according to the final XGBoost model."
            )
        elif prob <= 0.30:
            st.markdown('<div class="risk-mid">Intermediate-risk group</div>', unsafe_allow_html=True)
            interpret = (
                "The entered clinicopathological profile is associated with an intermediate estimated probability "
                "of synchronous liver-only metastasis according to the final XGBoost model."
            )
        else:
            st.markdown('<div class="risk-high">High-risk group</div>', unsafe_allow_html=True)
            interpret = (
                "The entered clinicopathological profile is associated with a high estimated probability "
                "of synchronous liver-only metastasis according to the final XGBoost model."
            )

        st.markdown(f'<div class="note-text">{interpret}</div>', unsafe_allow_html=True)
        st.progress(float(prob))

        st.markdown("---")
        st.markdown("**Input summary**")

        summary_df = pd.DataFrame({
            "Variable": ["Age", "Sex", "Race", "Primary site", "Grade", "T stage", "N stage", "CEA"],
            "Value": [age, sex, race, primary_site, grade, t_stage, n_stage, cea]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            '<div class="note-text">Enter the clinicopathological variables on the left and click <b>Predict</b> to generate an individualized risk estimate.</div>',
            unsafe_allow_html=True
        )

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer-box">
<b>Risk stratification rule:</b> low risk, predicted probability &lt;10%; intermediate risk, 10%–30%; high risk, &gt;30%.<br><br>
<b>Note:</b> This calculator is intended for research demonstration and individualized risk estimation based on the final model developed in this study.
It should be interpreted in conjunction with comprehensive clinical assessment rather than used as a standalone basis for treatment decision-making.
</div>
""", unsafe_allow_html=True)