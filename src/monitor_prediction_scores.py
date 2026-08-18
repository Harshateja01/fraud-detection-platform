import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)
DB_PORT = os.getenv(
    "DB_PORT",
    "5432",
)
DB_NAME = os.getenv(
    "DB_NAME",
    "fraud_detection",
)

engine = create_engine(
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# LOAD MODELING DATA
# ============================================================

query = """
SELECT
    t.transaction_id,
    t.tx_datetime,
    t.tx_amount,
    t.tx_fraud,

    f.during_weekend,
    f.during_night,
    f.customer_tx_count_1h,
    f.customer_tx_count_6h,
    f.customer_tx_count_24h,
    f.customer_avg_amount_24h,
    f.customer_avg_amount_7d,
    f.customer_amount_deviation,
    f.terminal_tx_count_24h,
    f.terminal_fraud_rate_7d,
    f.terminal_fraud_rate_30d,
    f.terminal_history_available

FROM transactions t

JOIN transaction_features f
    ON t.transaction_id = f.transaction_id

ORDER BY t.tx_datetime, t.transaction_id;
"""


print("=" * 80)
print("FRAUD MODEL — PREDICTION SCORE MONITORING")
print("=" * 80)

print("\nLoading model-scoring data...")

with engine.connect() as connection:

    df = pd.read_sql(
        text(query),
        connection,
    )

print(
    f"Rows loaded: "
    f"{len(df):,}"
)


# ============================================================
# REFERENCE / CURRENT PERIOD
# ============================================================

reference_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

current_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nMONITORING PERIODS")

print(
    f"Reference rows: "
    f"{len(reference_df):,}"
)

print(
    f"Current rows:   "
    f"{len(current_df):,}"
)


# ============================================================
# SCORE FUNCTION
# ============================================================

def score_dataframe(
    dataframe
):

    X = dataframe[
        FEATURE_COLUMNS
    ]

    X_imputed = (
        imputer.transform(
            X
        )
    )

    probabilities = (
        model.predict_proba(
            X_imputed
        )[:, 1]
    )

    return probabilities


# ============================================================
# GENERATE MODEL SCORES
# ============================================================

print(
    "\nGenerating reference-period scores..."
)

reference_scores = score_dataframe(
    reference_df
)

print(
    "Generating current-period scores..."
)

current_scores = score_dataframe(
    current_df
)


# ============================================================
# SCORE SUMMARY
# ============================================================

def summarize_scores(
    scores,
):

    return {
        "mean":
            float(
                np.mean(scores)
            ),

        "median":
            float(
                np.median(scores)
            ),

        "p90":
            float(
                np.quantile(
                    scores,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    scores,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    scores,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(scores)
            ),

        "alert_rate":
            float(
                np.mean(
                    scores
                    >= DECISION_THRESHOLD
                )
            ),
    }


reference_summary = summarize_scores(
    reference_scores
)

current_summary = summarize_scores(
    current_scores
)


summary_df = pd.DataFrame(
    [
        {
            "period":
                "Reference",

            **reference_summary,
        },

        {
            "period":
                "Current",

            **current_summary,
        },
    ]
)


display_summary = (
    summary_df.copy()
)

display_summary[
    [
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "max",
        "alert_rate",
    ]
] = display_summary[
    [
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "max",
        "alert_rate",
    ]
].round(4)


print("\n" + "=" * 90)
print("PREDICTION SCORE SUMMARY")
print("=" * 90)

print(
    display_summary.to_string(
        index=False
    )
)


# ============================================================
# PSI FUNCTION FOR MODEL SCORES
# ============================================================

def calculate_score_psi(
    reference_scores,
    current_scores,
    bins=10,
):

    reference = pd.Series(
        reference_scores
    )

    current = pd.Series(
        current_scores
    )


    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    bin_edges = np.unique(
        reference.quantile(
            quantiles
        ).values
    )


    if len(bin_edges) < 3:

        minimum = min(
            reference.min(),
            current.min(),
        )

        maximum = max(
            reference.max(),
            current.max(),
        )

        if minimum == maximum:
            return 0.0

        bin_edges = np.linspace(
            minimum,
            maximum,
            bins + 1,
        )


    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf


    reference_bins = pd.cut(
        reference,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    current_bins = pd.cut(
        current,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )


    reference_pct = (
        reference_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    current_pct = (
        current_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )


    all_bins = (
        reference_pct.index.union(
            current_pct.index
        )
    )

    reference_pct = (
        reference_pct.reindex(
            all_bins,
            fill_value=0,
        )
    )

    current_pct = (
        current_pct.reindex(
            all_bins,
            fill_value=0,
        )
    )


    epsilon = 0.000001

    reference_pct = (
        reference_pct.clip(
            lower=epsilon
        )
    )

    current_pct = (
        current_pct.clip(
            lower=epsilon
        )
    )


    psi_values = (
        current_pct
        - reference_pct
    ) * np.log(
        current_pct
        / reference_pct
    )


    return float(
        psi_values.sum()
    )


score_psi = calculate_score_psi(
    reference_scores,
    current_scores,
    bins=10,
)


# ============================================================
# SCORE DRIFT CLASSIFICATION
# ============================================================

def classify_score_drift(
    psi
):

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "MODERATE DRIFT"

    return "SIGNIFICANT DRIFT"


score_status = classify_score_drift(
    score_psi
)


# ============================================================
# ALERT-RATE CHANGE
# ============================================================

reference_alert_rate = (
    reference_summary[
        "alert_rate"
    ]
)

current_alert_rate = (
    current_summary[
        "alert_rate"
    ]
)

alert_rate_change_pp = (
    current_alert_rate
    - reference_alert_rate
) * 100


# ============================================================
# FRAUD SCORE SEPARATION
# ============================================================

current_analysis_df = (
    current_df[
        [
            "tx_fraud"
        ]
    ].copy()
)

current_analysis_df[
    "fraud_probability"
] = current_scores


fraud_score_summary = (
    current_analysis_df
    .groupby(
        "tx_fraud"
    )[
        "fraud_probability"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "max",
        ]
    )
    .reset_index()
)


fraud_score_summary[
    [
        "mean",
        "median",
        "max",
    ]
] = fraud_score_summary[
    [
        "mean",
        "median",
        "max",
    ]
].round(4)


print("\n" + "=" * 80)
print("SCORE DRIFT RESULT")
print("=" * 80)

print(
    f"\nPrediction-score PSI: "
    f"{score_psi:.4f}"
)

print(
    f"Score drift status: "
    f"{score_status}"
)

print(
    f"Reference alert rate: "
    f"{reference_alert_rate:.2%}"
)

print(
    f"Current alert rate:   "
    f"{current_alert_rate:.2%}"
)

print(
    f"Alert-rate change:    "
    f"{alert_rate_change_pp:+.2f} "
    f"percentage points"
)


print("\n" + "=" * 80)
print("CURRENT PERIOD SCORE BY ACTUAL FRAUD STATUS")
print("=" * 80)

print(
    fraud_score_summary.to_string(
        index=False
    )
)


# ============================================================
# OVERALL MONITORING STATUS
# ============================================================

if score_status == "SIGNIFICANT DRIFT":

    overall_status = (
        "REVIEW REQUIRED"
    )

elif score_status == "MODERATE DRIFT":

    overall_status = (
        "MONITOR"
    )

else:

    overall_status = (
        "STABLE"
    )


# Extra operational signal:
# large alert-rate changes deserve attention even if PSI
# remains below the significant-drift threshold.

if abs(
    alert_rate_change_pp
) >= 2.0:

    if overall_status == "STABLE":

        overall_status = (
            "MONITOR — ALERT RATE SHIFT"
        )


print("\n" + "=" * 80)
print("OVERALL SCORE MONITORING STATUS")
print("=" * 80)

print(
    f"\nPrediction-score monitoring status: "
    f"{overall_status}"
)


# ============================================================
# MONITORING POLICY
# ============================================================

print("\n" + "=" * 80)
print("MONITORING POLICY")
print("=" * 80)

print(
    "\nScore PSI < 0.10       → STABLE"
)

print(
    "0.10 ≤ Score PSI < 0.25 → MODERATE DRIFT"
)

print(
    "Score PSI ≥ 0.25       → SIGNIFICANT DRIFT"
)

print(
    "\nAbsolute alert-rate change ≥ "
    "2 percentage points triggers "
    "additional operational monitoring."
)


print("\n" + "=" * 80)
print("PREDICTION SCORE MONITORING COMPLETE")
print("=" * 80)