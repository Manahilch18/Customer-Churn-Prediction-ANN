# .streamlit/config.toml will be added separately; this file contains UI enhancements.

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import plotly.express as px

from src import config
from src.utils import load_all_artifacts, get_top_k_risk_customers
from src.predict import predict_single, predict_batch

# ── Page config must be the very first Streamlit command ──
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Global CSS – one scoped <style> block + background decoration layers
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root { color-scheme: light; }

        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > div {
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
        }

        /* ───────── Background: white → light-blue gradient ───────── */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 40%, #DBEAFE 100%) !important;
            position: relative;
            min-height: 100vh;
        }

        /* ───────── Radial glows in corners ───────── */
        [data-testid="stAppViewContainer"]::before {
            content: '';
            position: fixed;
            top: -120px;
            left: -80px;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(196, 181, 253, 0.10) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        [data-testid="stAppViewContainer"]::after {
            content: '';
            position: fixed;
            bottom: -100px;
            right: -60px;
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, rgba(96, 165, 250, 0.09) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .main .block-container {
            position: relative;
            z-index: 2;
            padding-top: 1rem;
            padding-bottom: 2rem;
            background: transparent !important;
        }

        /* ───────── Thin grid pattern overlay ───────── */
        .grid-overlay {
            position: fixed;
            inset: 0;
            background:
                repeating-linear-gradient(
                    0deg,
                    rgba(37, 99, 235, 0.025) 0px,
                    transparent 1px,
                    transparent 64px
                ),
                repeating-linear-gradient(
                    90deg,
                    rgba(37, 99, 235, 0.025) 0px,
                    transparent 1px,
                    transparent 64px
                );
            pointer-events: none;
            z-index: 0;
        }

        /* ───────── Tiny glowing network-dot accents ───────── */
        .network-dots {
            position: fixed;
            inset: 0;
            background-image:
                radial-gradient(circle, rgba(124, 58, 237, 0.10) 1.2px, transparent 1.2px),
                radial-gradient(circle, rgba(37, 99, 235, 0.07) 1px, transparent 1px);
            background-size: 140px 140px, 90px 90px;
            background-position: 0 0, 45px 45px;
            pointer-events: none;
            z-index: 0;
        }

        /* ───────── Glass card (standalone HTML) ───────── */
        .glass-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(37, 99, 235, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        /* ───────── Expanders re-skinned as glass cards ───────── */
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.88) !important;
            border: 1px solid rgba(255, 255, 255, 0.65) !important;
            border-radius: 24px !important;
            box-shadow: 0 8px 32px rgba(37, 99, 235, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            overflow: hidden;
            margin-bottom: 1rem;
        }

        [data-testid="stExpander"] details {
            border: none !important;
            background: transparent !important;
        }

        [data-testid="stExpander"] details summary {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E293B;
            padding: 1rem 1.25rem;
        }

        [data-testid="stExpander"] details summary:hover {
            color: #2563EB;
        }

        [data-testid="stExpander"] .streamlit-expanderContent {
            padding: 0 1.25rem 1rem 1.25rem;
        }

        /* ───────── Gradient divider ───────── */
        .gradient-divider {
            height: 2px;
            border: 0;
            margin: 1.5rem 0;
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.25), #7C3AED, transparent);
            border-radius: 2px;
        }

        /* ───────── Badge pill ───────── */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            color: #FFFFFF;
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.18);
        }

        /* ───────── Result card ───────── */
        .result-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(59, 130, 246, 0.16);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(37, 99, 235, 0.08);
            backdrop-filter: blur(12px);
            padding: 1.5rem;
            margin-top: 1.5rem;
        }
        .result-card.low  { box-shadow: 0 8px 40px rgba(34, 197, 94, 0.14); border-color: rgba(34, 197, 94, 0.25); }
        .result-card.high { box-shadow: 0 8px 40px rgba(231, 76, 60, 0.14); border-color: rgba(231, 76, 60, 0.25); }

        /* ───────── KPI / generic card ───────── */
        .kpi-card, .card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(37, 99, 235, 0.08);
            backdrop-filter: blur(12px);
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        /* ───────── Buttons ───────── */
        .stButton > button {
            border-radius: 999px !important;
            padding: 0.75rem 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.18);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 8px 28px rgba(37, 99, 235, 0.30);
            transform: translateY(-2px);
        }

        .stDownloadButton > button {
            border-radius: 999px !important;
            font-weight: 600;
        }

        /* ───────── Typography ───────── */
        h1, h2, h3, h4, h5 {
            letter-spacing: -0.03em;
            color: #0F172A !important;
        }

        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #475569;
        }

        /* ───────── Tab bar ───────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 14px 14px 0 0;
            padding: 0.65rem 1.3rem;
            font-weight: 600;
        }

        /* ───────── Metrics ───────── */
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #1E40AF !important;
        }

        /* ───────── Sidebar ───────── */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.72) !important;
            backdrop-filter: blur(14px);
        }
        [data-testid="stSidebar"] .block-container {
            background: transparent !important;
        }
    </style>

    <!-- Background decoration layers -->
    <div class="grid-overlay"></div>
    <div class="network-dots"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_all_artifacts()

artifacts = get_artifacts()

# ---------------------------------------------------------------------------
# Header – title on left, threshold badge on right
# ---------------------------------------------------------------------------
col_title, col_threshold = st.columns([3, 1])
with col_title:
    st.title("Bank Customer Churn Prediction Dashboard")
    st.caption("Predict churn risk and prioritize interventions")
with col_threshold:
    threshold = artifacts.get('best_threshold', config.DEFAULT_THRESHOLD)
    st.markdown(
        f"**Operating Threshold:** <span class='badge'>{threshold:.3f}</span>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Prediction", "Batch Risk Ranking", "Model Insights", "About This Project"]
)

# ===================== Tab 1 · Single Prediction =====================
with tab1:
    st.header("Single Customer Prediction")
    st.write("Enter customer details to predict churn risk.")

    # ── Demographics (glass-card via expander) ──
    with st.expander("🧑  Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.slider(
                "CreditScore", min_value=300, max_value=850, value=650,
                key="input_credit_score",
            )
            age = st.slider(
                "Age", min_value=18, max_value=100, value=35,
                key="input_age",
            )
            geography_germany = st.selectbox(
                "Geography_Germany", [0, 1], key="input_geography_germany",
            )
        with col2:
            gender = st.selectbox(
                "Gender", [0, 1], help="0: Female, 1: Male",
                key="input_gender",
            )
            geography_spain = st.selectbox(
                "Geography_Spain", [0, 1], key="input_geography_spain",
            )

    # ── Account Details (glass-card via expander) ──
    with st.expander("🏦  Account Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            has_cr_card = st.selectbox(
                "HasCrCard", [0, 1], key="input_has_cr_card",
            )
            tenure = st.slider(
                "Tenure", min_value=0, max_value=10, value=5,
                key="input_tenure",
            )
        with col2:
            is_active_member = st.selectbox(
                "IsActiveMember", [0, 1], key="input_is_active_member",
            )
            num_of_products = st.number_input(
                "NumOfProducts", min_value=1, max_value=4, value=1,
                key="input_num_of_products",
            )

    # ── Financials (glass-card via expander) ──
    with st.expander("💰  Financials", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input(
                "Balance", min_value=0.0, value=0.0, key="input_balance",
            )
        with col2:
            estimated_salary = st.number_input(
                "EstimatedSalary", min_value=0.0, value=50000.0,
                key="input_estimated_salary",
            )

    # Validation and predict action
    error_msg = None
    if age < 18 or age > 100:
        error_msg = "Age must be between 18 and 100."

    if error_msg:
        st.error(error_msg)
    else:
        if st.button("Predict Churn", key="input_predict_churn"):
            try:
                customer_dict = {
                    "CreditScore": credit_score,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": num_of_products,
                    "HasCrCard": has_cr_card,
                    "IsActiveMember": is_active_member,
                    "EstimatedSalary": estimated_salary,
                    "Geography_Germany": geography_germany,
                    "Geography_Spain": geography_spain,
                }
                result = predict_single(customer_dict)
                prob = result["churn_probability"]
                pred = result["predicted_churn"]
                risk_class = "low" if pred == 0 else "high"
                st.markdown(
                    f"""
                    <div class='result-card {risk_class}'>
                        <h4>Prediction Result</h4>
                        {"<span style='color:#e74c3c;font-weight:600;font-size:1.15rem;'>⚠ High Risk</span>" if pred == 1 else "<span style='color:#2ecc71;font-weight:600;font-size:1.15rem;'>✅ Low Risk</span>"}
                        <br/>
                        <div style='font-size:1.2rem; margin-top:0.5rem;'>
                            Churn Probability: <strong>{prob:.2%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Risk Factors (static example)"):
                    st.write(
                        "- Low credit score\n"
                        "- High balance\n"
                        "- Inactive member"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ===================== Tab 2 · Batch Risk Ranking =====================
with tab2:
    st.header("Batch Risk Ranking")
    st.write("Upload a CSV of customer data to rank them by churn risk.")

    with st.expander("📤  Upload & Configure", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], key="input_upload_csv",
        )
        capacity = st.slider(
            "Intervention Capacity (Top K)",
            min_value=10, max_value=2000,
            value=int(artifacts.get('capacity', config.DEFAULT_CAPACITY)),
            step=10, key="input_capacity",
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            feature_columns = artifacts.get('feature_columns', [])
            missing_cols = [col for col in feature_columns if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in feature_columns]

            st.markdown(
                f"""
                <div class='glass-card'>
                    <strong>📄 File:</strong> {uploaded_file.name}<br/>
                    <strong>📊 Rows:</strong> {len(df)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
                if extra_cols:
                    st.warning(
                        f"Extra columns found (ignored during prediction): {extra_cols}"
                    )
            else:
                predictions_df = predict_batch(df)
                top_k_df = get_top_k_risk_customers(predictions_df, capacity)

                # KPI cards
                col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                with col_kpi1:
                    st.metric(label="Total Customers", value=len(df))
                with col_kpi2:
                    flagged = (predictions_df["predicted_churn"] == 1).sum()
                    st.metric(label="Flagged (Risk)", value=flagged)
                with col_kpi3:
                    est_churners = (
                        predictions_df["churn_probability"] > threshold
                    ).sum()
                    st.metric(
                        label="Est. Churners (>threshold)", value=est_churners,
                    )

                st.markdown(
                    '<div class="gradient-divider"></div>',
                    unsafe_allow_html=True,
                )

                st.subheader(f"Top {capacity} Customers at Risk")
                st.dataframe(
                    top_k_df.style.format({"churn_probability": "{:.2%}"}),
                )

                csv = top_k_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Top K as CSV",
                    data=csv,
                    file_name="top_k_risk_customers.csv",
                    mime="text/csv",
                    key="input_download_topk",
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ===================== Tab 3 · Model Insights =====================
with tab3:
    st.header("Model Insights")

    metrics_path = config.METRICS_CSV_PATH
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        with st.expander("📊  Performance Metrics", expanded=True):
            st.dataframe(metrics_df)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # Confusion Matrix
    cm_path = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        img = Image.open(cm_path)
        fig_cm = px.imshow(np.array(img))
        fig_cm.update_xaxes(showticklabels=False).update_yaxes(
            showticklabels=False,
        )
        fig_cm.update_layout(
            margin=dict(l=0, r=0, t=30, b=0), title="Confusion Matrix",
        )
        with col_left:
            st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    roc_path = os.path.join(config.FIGURES_DIR, "roc_curve.png")
    if os.path.exists(roc_path):
        img = Image.open(roc_path)
        fig_roc = px.imshow(np.array(img))
        fig_roc.update_xaxes(showticklabels=False).update_yaxes(
            showticklabels=False,
        )
        fig_roc.update_layout(
            margin=dict(l=0, r=0, t=30, b=0), title="ROC Curve",
        )
        with col_right:
            st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall Curve
    pr_path = os.path.join(config.FIGURES_DIR, "precision_recall_curve.png")
    if os.path.exists(pr_path):
        img = Image.open(pr_path)
        fig_pr = px.imshow(np.array(img))
        fig_pr.update_xaxes(showticklabels=False).update_yaxes(
            showticklabels=False,
        )
        fig_pr.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            title="Precision\u2011Recall Curve",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

# ===================== Tab 4 · About This Project =====================
with tab4:
    st.header("About This Project")

    with st.expander("💰  Business Assumptions", expanded=True):
        st.markdown(
            f"""
            <div style="display:flex; gap:2rem; flex-wrap:wrap;">
                <div>
                    <strong>Cost of False Negative</strong> (Missed Churner)<br/>
                    <span class='badge' style="margin-top:0.5rem; display:inline-block;">
                        ${config.COST_FN}
                    </span>
                </div>
                <div>
                    <strong>Cost of False Positive</strong> (Unnecessary Intervention)<br/>
                    <span class='badge' style="margin-top:0.5rem; display:inline-block;">
                        ${config.COST_FP}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔧  Project Features", expanded=True):
        st.write(
            "- Handles class imbalance with class\u2011weights\n"
            "- Cost\u2011sensitive threshold optimisation\n"
            "- Capacity\u2011based risk ranking for targeted interventions\n"
            "- Fully reproducible training pipeline"
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class='glass-card'>
            <h4 style="margin-top:0;">📖 About</h4>
            <ul style="margin-bottom:0;">
                <li>This project is an Artificial Neural Network (ANN) built to
                    predict customer churn in the banking sector.</li>
                <li>A Streamlit dashboard provides single\u2011prediction, batch
                    ranking, and model insight views.</li>
                <li>All configuration values (threshold, capacity, costs, etc.)
                    are pulled from <code>src/config.py</code> and saved
                    artifacts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar – static navigation / info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Customer Churn")
    st.caption("Predict & act on churn risk")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.write(
        "[GitHub Repo](https://github.com/Manahilch18/Customer-Churn-Prediction-ANN)"
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.info(
        "Use the tabs above to explore predictions, rankings, and model diagnostics."
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
