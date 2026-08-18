from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = project_root / "models"

model_path = (
    models_dir
    / "fraud_xgboost_model.joblib"
)

imputer_path = (
    models_dir
    / "fraud_imputer.joblib"
)

metadata_path = (
    models_dir
    / "fraud_model_metadata.joblib"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(
    model_path
)

imputer = joblib.load(
    imputer_path
)

metadata = joblib.load(
    metadata_path
)


FEATURE_COLUMNS = metadata[
    "feature_columns"
]

DECISION_THRESHOLD = metadata[
    "decision_threshold"
]


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fraud Detection Scoring API",
    description=(
        "Real-time transaction fraud scoring "
        "using the champion XGBoost fraud model "
        "with SHAP explainability."
    ),
    version="1.1.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TransactionRequest(BaseModel):

    tx_amount: float = Field(
        ge=0,
        description="Transaction amount",
    )

    during_weekend: int = Field(
        ge=0,
        le=1,
        description="1 if transaction occurs on weekend",
    )

    during_night: int = Field(
        ge=0,
        le=1,
        description="1 if transaction occurs during night",
    )

    customer_tx_count_1h: int = Field(
        ge=0
    )

    customer_tx_count_6h: int = Field(
        ge=0
    )

    customer_tx_count_24h: int = Field(
        ge=0
    )

    customer_avg_amount_24h: float | None = None

    customer_avg_amount_7d: float | None = None

    customer_amount_deviation: float

    terminal_tx_count_24h: int = Field(
        ge=0
    )

    terminal_fraud_rate_7d: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    terminal_fraud_rate_30d: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    terminal_history_available: int = Field(
        ge=0,
        le=1,
    )


# ============================================================
# SHAP RISK FACTOR MODEL
# ============================================================

class RiskFactor(BaseModel):

    feature: str

    value: float | None

    shap_value: float


# ============================================================
# RESPONSE MODEL
# ============================================================

class FraudScoreResponse(BaseModel):

    fraud_probability: float

    fraud_probability_pct: float

    decision_threshold: float

    alert_generated: bool

    risk_level: str

    recommended_action: str

    model_name: str

    factors_increasing_risk: list[RiskFactor]

    factors_reducing_risk: list[RiskFactor]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "Fraud Detection Scoring API",

        "status":
            "running",

        "model":
            metadata["model_name"],

        "version":
            "1.1.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model_loaded":
            True,

        "imputer_loaded":
            True,

        "shap_explainer_loaded":
            True,

        "decision_threshold":
            DECISION_THRESHOLD,

        "feature_count":
            len(FEATURE_COLUMNS),
    }


# ============================================================
# HELPER FUNCTION
# ============================================================

def build_risk_factors(
    dataframe
):

    factors = []

    for _, row in dataframe.iterrows():

        value = row["value"]

        if pd.isna(value):
            value = None

        else:
            value = float(value)

        factors.append(
            RiskFactor(

                feature=
                    str(row["feature"]),

                value=
                    value,

                shap_value=
                    round(
                        float(
                            row["shap_value"]
                        ),
                        6,
                    ),
            )
        )

    return factors


# ============================================================
# FRAUD SCORING ENDPOINT
# ============================================================

@app.post(
    "/score-transaction",
    response_model=FraudScoreResponse,
)
def score_transaction(
    transaction: TransactionRequest
):

    # --------------------------------------------------------
    # REQUEST → DATAFRAME
    # --------------------------------------------------------

    transaction_data = pd.DataFrame(
        [
            transaction.model_dump()
        ],
        columns=FEATURE_COLUMNS,
    )


    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------

    transaction_imputed = (
        imputer.transform(
            transaction_data
        )
    )


    # --------------------------------------------------------
    # MODEL SCORE
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(
            transaction_imputed
        )[0, 1]
    )


    # --------------------------------------------------------
    # ALERT DECISION
    # --------------------------------------------------------

    alert_generated = (
        probability
        >= DECISION_THRESHOLD
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.90:

        risk_level = "CRITICAL"

    elif probability >= 0.70:

        risk_level = "HIGH"

    elif probability >= DECISION_THRESHOLD:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    if risk_level == "CRITICAL":

        recommended_action = (
            "IMMEDIATE FRAUD REVIEW"
        )

    elif risk_level == "HIGH":

        recommended_action = (
            "PRIORITY INVESTIGATION"
        )

    elif risk_level == "MEDIUM":

        recommended_action = (
            "QUEUE FOR REVIEW"
        )

    else:

        recommended_action = (
            "NO FRAUD ALERT"
        )


    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        transaction_imputed
    )

    shap_values = np.asarray(
        shap_values
    )

    shap_values = np.squeeze(
        shap_values
    )


    explanation_df = pd.DataFrame({
        "feature":
            FEATURE_COLUMNS,

        "value":
            transaction_data
            .iloc[0]
            .values,

        "shap_value":
            shap_values,
    })


    explanation_df[
        "abs_shap"
    ] = explanation_df[
        "shap_value"
    ].abs()


    # --------------------------------------------------------
    # FACTORS INCREASING RISK
    # --------------------------------------------------------

    increasing_df = (
        explanation_df[
            explanation_df[
                "shap_value"
            ] > 0
        ]
        .sort_values(
            "abs_shap",
            ascending=False,
        )
        .head(5)
    )


    # --------------------------------------------------------
    # FACTORS REDUCING RISK
    # --------------------------------------------------------

    reducing_df = (
        explanation_df[
            explanation_df[
                "shap_value"
            ] < 0
        ]
        .sort_values(
            "abs_shap",
            ascending=False,
        )
        .head(5)
    )


    increasing_factors = (
        build_risk_factors(
            increasing_df
        )
    )

    reducing_factors = (
        build_risk_factors(
            reducing_df
        )
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return FraudScoreResponse(

        fraud_probability=
            round(
                probability,
                6,
            ),

        fraud_probability_pct=
            round(
                probability * 100,
                2,
            ),

        decision_threshold=
            DECISION_THRESHOLD,

        alert_generated=
            alert_generated,

        risk_level=
            risk_level,

        recommended_action=
            recommended_action,

        model_name=
            metadata["model_name"],

        factors_increasing_risk=
            increasing_factors,

        factors_reducing_risk=
            reducing_factors,
    )