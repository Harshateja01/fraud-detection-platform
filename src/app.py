import os
import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Investigator",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# API CONFIG
# ============================================================

API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{API_BASE_URL}/score-transaction"
HEALTH_URL = f"{API_BASE_URL}/health"


# ============================================================
# HEADER
# ============================================================

st.title("🚨 Fraud Detection Investigator")

st.caption(
    "Real-time transaction fraud scoring through "
    "a FastAPI-powered XGBoost scoring service."
)


# ============================================================
# API HEALTH CHECK
# ============================================================

api_available = False

try:
    health_response = requests.get(
        HEALTH_URL,
        timeout=3,
    )

    if health_response.status_code == 200:
        api_available = True

except requests.RequestException:
    api_available = False


if api_available:

    st.success(
        "Scoring API is online.",
        icon="✅",
    )

else:

    st.error(
        "FastAPI scoring service is not available. "
        "Start it with: "
        "`python -m uvicorn src.api:app --reload`"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Transaction Input")

col1, col2, col3 = st.columns(3)


with col1:

    tx_amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=164.75,
        step=1.0,
    )

    during_weekend = st.selectbox(
        "Weekend Transaction",
        options=[0, 1],
        format_func=lambda x: (
            "Yes" if x == 1 else "No"
        ),
    )

    during_night = st.selectbox(
        "Night Transaction",
        options=[0, 1],
        format_func=lambda x: (
            "Yes" if x == 1 else "No"
        ),
    )


with col2:

    customer_tx_count_1h = st.number_input(
        "Customer Transactions — 1h",
        min_value=0,
        value=0,
        step=1,
    )

    customer_tx_count_6h = st.number_input(
        "Customer Transactions — 6h",
        min_value=0,
        value=0,
        step=1,
    )

    customer_tx_count_24h = st.number_input(
        "Customer Transactions — 24h",
        min_value=0,
        value=1,
        step=1,
    )

    terminal_tx_count_24h = st.number_input(
        "Terminal Transactions — 24h",
        min_value=0,
        value=2,
        step=1,
    )


with col3:

    customer_avg_amount_24h = st.number_input(
        "Customer Avg Amount — 24h",
        min_value=0.0,
        value=35.49,
        step=1.0,
    )

    customer_avg_amount_7d = st.number_input(
        "Customer Avg Amount — 7d",
        min_value=0.0,
        value=60.72,
        step=1.0,
    )

    terminal_fraud_rate_7d = st.number_input(
        "Terminal Fraud Rate — 7d",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )

    terminal_fraud_rate_30d = st.number_input(
        "Terminal Fraud Rate — 30d",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

customer_amount_deviation = (
    tx_amount
    - customer_avg_amount_7d
)

terminal_history_available = 1


# ============================================================
# REQUEST PAYLOAD
# ============================================================

payload = {

    "tx_amount":
        tx_amount,

    "during_weekend":
        during_weekend,

    "during_night":
        during_night,

    "customer_tx_count_1h":
        customer_tx_count_1h,

    "customer_tx_count_6h":
        customer_tx_count_6h,

    "customer_tx_count_24h":
        customer_tx_count_24h,

    "customer_avg_amount_24h":
        customer_avg_amount_24h,

    "customer_avg_amount_7d":
        customer_avg_amount_7d,

    "customer_amount_deviation":
        customer_amount_deviation,

    "terminal_tx_count_24h":
        terminal_tx_count_24h,

    "terminal_fraud_rate_7d":
        terminal_fraud_rate_7d,

    "terminal_fraud_rate_30d":
        terminal_fraud_rate_30d,

    "terminal_history_available":
        terminal_history_available,
}


# ============================================================
# SCORE BUTTON
# ============================================================

score_button = st.button(
    "Score Transaction",
    type="primary",
    width="stretch",
    disabled=not api_available,
)


# ============================================================
# SCORE VIA FASTAPI
# ============================================================

if score_button:

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=15,
        )


        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if response.status_code != 200:

            st.error(
                f"Scoring API returned "
                f"HTTP {response.status_code}"
            )

            st.code(
                response.text
            )

            st.stop()


        result = response.json()


        # ====================================================
        # RESPONSE VALUES
        # ====================================================

        probability = result[
            "fraud_probability"
        ]

        probability_pct = result[
            "fraud_probability_pct"
        ]

        threshold = result[
            "decision_threshold"
        ]

        risk_level = result[
            "risk_level"
        ]

        alert_generated = result[
            "alert_generated"
        ]

        recommended_action = result[
            "recommended_action"
        ]

        model_name = result[
            "model_name"
        ]


        # ====================================================
        # RISK ASSESSMENT
        # ====================================================

        st.divider()

        st.subheader(
            "Risk Assessment"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        metric1.metric(
            "Fraud Probability",
            f"{probability_pct:.2f}%",
        )


        metric2.metric(
            "Risk Level",
            risk_level,
        )


        metric3.metric(
            "Alert Threshold",
            f"{threshold:.0%}",
        )


        metric4.metric(
            "Alert Generated",
            (
                "YES"
                if alert_generated
                else "NO"
            ),
        )


        # ====================================================
        # RECOMMENDED ACTION
        # ====================================================

        if alert_generated:

            st.warning(
                f"Recommended action: "
                f"**{recommended_action}**"
            )

        else:

            st.success(
                f"Recommended action: "
                f"**{recommended_action}**"
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.subheader(
            "Fraud Risk Probability"
        )

        st.progress(
            min(
                probability,
                1.0,
            )
        )

        st.caption(
            f"Predicted fraud probability: "
            f"{probability_pct:.2f}% | "
            f"Decision threshold: "
            f"{threshold:.0%}"
        )


        # ====================================================
        # TRANSACTION INFORMATION
        # ====================================================

        st.subheader(
            "Transaction Information"
        )

        info1, info2, info3, info4 = (
            st.columns(4)
        )


        info1.metric(
            "Transaction Amount",
            f"${tx_amount:,.2f}",
        )


        info2.metric(
            "7-Day Customer Average",
            f"${customer_avg_amount_7d:,.2f}",
        )


        info3.metric(
            "Amount Deviation",
            f"${customer_amount_deviation:,.2f}",
        )


        info4.metric(
            "Terminal 7-Day Fraud Rate",
            f"{terminal_fraud_rate_7d:.2%}",
        )


        # ====================================================
        # SHAP EXPLAINABILITY FROM API
        # ====================================================

        st.subheader(
            "Model Explainability"
        )

        st.caption(
            "SHAP explanations are generated "
            "by the FastAPI scoring service."
        )


        increasing = result[
            "factors_increasing_risk"
        ]

        reducing = result[
            "factors_reducing_risk"
        ]


        increasing_df = pd.DataFrame(
            increasing
        )

        reducing_df = pd.DataFrame(
            reducing
        )


        # Friendly column names

        if not increasing_df.empty:

            increasing_df = (
                increasing_df.rename(
                    columns={
                        "feature":
                            "Feature",

                        "value":
                            "Value",

                        "shap_value":
                            "SHAP Value",
                    }
                )
            )


        if not reducing_df.empty:

            reducing_df = (
                reducing_df.rename(
                    columns={
                        "feature":
                            "Feature",

                        "value":
                            "Value",

                        "shap_value":
                            "SHAP Value",
                    }
                )
            )


        explain1, explain2 = (
            st.columns(2)
        )


        with explain1:

            st.markdown(
                "### 🔴 Factors Increasing Risk"
            )

            if increasing_df.empty:

                st.write(
                    "No major positive "
                    "risk factors."
                )

            else:

                st.dataframe(
                    increasing_df,
                    width="stretch",
                    hide_index=True,
                )


        with explain2:

            st.markdown(
                "### 🟢 Factors Reducing Risk"
            )

            if reducing_df.empty:

                st.write(
                    "No major "
                    "risk-reducing factors."
                )

            else:

                st.dataframe(
                    reducing_df,
                    width="stretch",
                    hide_index=True,
                )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.subheader(
            "Model Information"
        )

        m1, m2, m3, m4 = (
            st.columns(4)
        )


        m1.metric(
            "Model",
            model_name,
        )


        m2.metric(
            "ROC-AUC",
            "0.8866",
        )


        m3.metric(
            "PR-AUC",
            "0.6613",
        )


        m4.metric(
            "Test Recall",
            "75.50%",
        )


    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except requests.RequestException as error:

        st.error(
            "Unable to reach the FastAPI "
            "scoring service."
        )

        st.exception(
            error
        )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.divider()

st.subheader(
    "Scoring Architecture"
)

st.code(
    """
Streamlit Investigator Dashboard
              |
              | HTTP POST
              v
      FastAPI Scoring Service
              |
      -------------------
      |        |        |
   Imputer  XGBoost   SHAP
      |        |        |
      -------------------
              |
              v
Probability + Alert + Explanation
""",
    language="text",
)


# ============================================================
# GOVERNANCE
# ============================================================

st.subheader(
    "Model Governance"
)

st.info(
    """
**Research / Portfolio Fraud Detection Model**

This application is intended for demonstration and
analytical purposes.

The model was developed on simulated transaction data.
The operating threshold was selected using validation-period
business-cost assumptions, and the final model was evaluated
on a chronologically later test period.

The model should not be used directly for production fraud
decisions without additional validation, calibration,
monitoring, fairness analysis, security controls, operational
review, and governance approval.
"""
)