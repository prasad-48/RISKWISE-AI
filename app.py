import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RiskWise AI",
    page_icon="🏦",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("finance/models/xgb_model.pkl")

model = load_model()

# =========================
# CUSTOM CSS (PRO UI)
# =========================
st.markdown("""
    <style>

    .main {
        background-color: #0f172a;
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .metric-box {
        background: #1e293b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .risk-low {
        background-color: #16a34a;
        padding: 12px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }

    .risk-medium {
        background-color: #f59e0b;
        padding: 12px;
        border-radius: 10px;
        color: black;
        font-weight: bold;
        text-align: center;
    }

    .risk-high {
        background-color: #dc2626;
        padding: 12px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🏦 RiskWise AI")
st.subheader("AI Powered Credit Risk Intelligence System")

st.markdown("---")

# =========================
# SIDEBAR INFO
# =========================
with st.sidebar:
    st.title("📊 Navigation")
    st.info("Credit Risk Prediction System")
    st.write("Built using XGBoost + SHAP")

    st.markdown("---")
    st.write("🔹 Model: XGBoost")
    st.write("🔹 Dataset: UCI Credit Card")
    st.write("🔹 Type: Classification")

# =========================
# INPUT SECTION
# =========================
st.header("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input("Credit Limit", 10000, 1000000, 50000)
    age = st.number_input("Age", 18, 100, 35)
    education = st.selectbox("Education", [1,2,3,4])

with col2:
    marriage = st.selectbox("Marital Status", [1,2,3])
    sex = st.selectbox("Gender", [1,2])
    pay_0 = st.slider("PAY_0", -2, 8, 0)
    pay_2 = st.slider("PAY_2", -2, 8, 0)

with col3:
    pay_3 = st.slider("PAY_3", -2, 8, 0)
    pay_4 = st.slider("PAY_4", -2, 8, 0)
    pay_5 = st.slider("PAY_5", -2, 8, 0)
    pay_6 = st.slider("PAY_6", -2, 8, 0)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Credit Risk", use_container_width=True):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": 10000,
        "BILL_AMT2": 10000,
        "BILL_AMT3": 10000,
        "BILL_AMT4": 10000,
        "BILL_AMT5": 10000,
        "BILL_AMT6": 10000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000
    }])

    prob = model.predict_proba(input_data)[0][1]

    # =========================
    # RISK CLASSIFICATION
    # =========================
    if prob < 0.30:
        risk = "LOW"
        css_class = "risk-low"
        recommendation = "Approve Loan"

    elif prob < 0.60:
        risk = "MEDIUM"
        css_class = "risk-medium"
        recommendation = "Manual Review Required"

    else:
        risk = "HIGH"
        css_class = "risk-high"
        recommendation = "Reject Loan"

    # =========================
    # OUTPUT UI
    # =========================
    st.markdown("## 📊 Prediction Results")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Default Probability</h3>
            <h2>{prob*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="{css_class}">
            Risk Level: {risk}
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Decision</h3>
            <h3>{recommendation}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.success("Prediction Completed Successfully")