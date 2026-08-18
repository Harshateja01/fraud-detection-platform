import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODELS_DIR
    / "fraud_xgboost_model.joblib"
)

IMPUTER_PATH = (
    MODELS_DIR
    / "fraud_imputer.joblib"
)

METADATA_PATH = (
    MODELS_DIR
    / "fraud_model_metadata.joblib"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_PATH
)

imputer = joblib.load(
    IMPUTER_PATH
)

metadata = joblib.load(
    METADATA_PATH
)

FEATURE_COLUMNS = metadata[
    "feature_columns"
]

DECISION_THRESHOLD = metadata[
    "decision_threshold"
]


# ============================================================
# EXPECTED MATURITY FEATURES
# ============================================================

EXPECTED_MATURITY_FEATURES = {
    "terminal_history_available",
}


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
# LOAD DATA
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


print("=" * 90)
print("FRAUD DETECTION PLATFORM — UNIFIED MODEL HEALTH REPORT")
print("=" * 90)

print("\nLoading monitoring data...")

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
# PERIODS
# ============================================================

reference_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

current_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nMONITORING PERIODS")

print(
    f"Reference: "
    f"{len(reference_df):,} transactions"
)

print(
    f"Current:   "
    f"{len(current_df):,} transactions"
)


# ============================================================
# GENERIC PSI
# ============================================================

def calculate_psi(
    reference_series,
    current_series,
    bins=10,
):

    reference = pd.to_numeric(
        reference_series,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current_series,
        errors="coerce",
    ).dropna()


    if len(reference) == 0 or len(current) == 0:
        return np.nan


    # --------------------------------------------------------
    # CONSTANT REFERENCE FEATURE
    # --------------------------------------------------------

    if reference.nunique() <= 1:

        reference_value = (
            reference.iloc[0]
        )

        current_same = (
            current
            == reference_value
        ).mean()

        if current_same == 1:
            return 0.0

        return 1.0


    # --------------------------------------------------------
    # QUANTILE BINS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LOW-CARDINALITY FALLBACK
    # --------------------------------------------------------

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


    all_bins = reference_pct.index.union(
        current_pct.index
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


# ============================================================
# PSI STATUS
# ============================================================

def classify_psi(
    psi
):

    if pd.isna(psi):
        return "UNKNOWN"

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "MODERATE DRIFT"

    return "SIGNIFICANT DRIFT"


# ============================================================
# MODEL SCORE FUNCTION
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
# 1. FEATURE DRIFT MONITORING
# ============================================================

feature_results = []


for feature in FEATURE_COLUMNS:

    psi = calculate_psi(
        reference_df[feature],
        current_df[feature],
    )

    feature_results.append({
        "feature":
            feature,

        "psi":
            psi,

        "status":
            classify_psi(
                psi
            ),
    })


feature_results_df = pd.DataFrame(
    feature_results
)

feature_results_df = (
    feature_results_df
    .sort_values(
        "psi",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


unexpected_feature_alerts = (
    feature_results_df[
        (
            feature_results_df[
                "status"
            ] != "STABLE"
        )
        &
        (
            ~feature_results_df[
                "feature"
            ].isin(
                EXPECTED_MATURITY_FEATURES
            )
        )
    ]
)


maturity_feature_alerts = (
    feature_results_df[
        (
            feature_results_df[
                "status"
            ] != "STABLE"
        )
        &
        (
            feature_results_df[
                "feature"
            ].isin(
                EXPECTED_MATURITY_FEATURES
            )
        )
    ]
)


significant_unexpected_features = (
    unexpected_feature_alerts[
        unexpected_feature_alerts[
            "status"
        ] == "SIGNIFICANT DRIFT"
    ]
)


if not significant_unexpected_features.empty:

    feature_status = (
        "REVIEW REQUIRED"
    )

elif not unexpected_feature_alerts.empty:

    feature_status = (
        "MONITOR"
    )

elif not maturity_feature_alerts.empty:

    feature_status = (
        "STABLE — EXPECTED FEATURE MATURITY"
    )

else:

    feature_status = (
        "STABLE"
    )


# ============================================================
# 2. PREDICTION SCORE MONITORING
# ============================================================

print(
    "\nGenerating model scores..."
)

reference_scores = score_dataframe(
    reference_df
)

current_scores = score_dataframe(
    current_df
)


score_psi = calculate_psi(
    pd.Series(
        reference_scores
    ),
    pd.Series(
        current_scores
    ),
)


score_psi_status = classify_psi(
    score_psi
)


reference_alert_rate = float(
    np.mean(
        reference_scores
        >= DECISION_THRESHOLD
    )
)

current_alert_rate = float(
    np.mean(
        current_scores
        >= DECISION_THRESHOLD
    )
)

alert_rate_change_pp = (
    current_alert_rate
    - reference_alert_rate
) * 100


if score_psi_status == "SIGNIFICANT DRIFT":

    score_status = (
        "REVIEW REQUIRED"
    )

elif score_psi_status == "MODERATE DRIFT":

    score_status = (
        "MONITOR"
    )

elif abs(
    alert_rate_change_pp
) >= 2:

    score_status = (
        "MONITOR — ALERT RATE SHIFT"
    )

else:

    score_status = (
        "STABLE"
    )


# ============================================================
# 3. DELAYED LABEL PERFORMANCE MONITORING
# ============================================================

reference_probabilities = (
    reference_scores
)

current_probabilities = (
    current_scores
)


reference_y = (
    reference_df[
        "tx_fraud"
    ]
    .astype(int)
    .to_numpy()
)

current_y = (
    current_df[
        "tx_fraud"
    ]
    .astype(int)
    .to_numpy()
)


reference_predictions = (
    reference_probabilities
    >= DECISION_THRESHOLD
).astype(int)

current_predictions = (
    current_probabilities
    >= DECISION_THRESHOLD
).astype(int)


reference_pr_auc = (
    average_precision_score(
        reference_y,
        reference_probabilities,
    )
)

current_pr_auc = (
    average_precision_score(
        current_y,
        current_probabilities,
    )
)


reference_precision = (
    precision_score(
        reference_y,
        reference_predictions,
        zero_division=0,
    )
)

current_precision = (
    precision_score(
        current_y,
        current_predictions,
        zero_division=0,
    )
)


reference_recall = (
    recall_score(
        reference_y,
        reference_predictions,
        zero_division=0,
    )
)

current_recall = (
    recall_score(
        current_y,
        current_predictions,
        zero_division=0,
    )
)


pr_auc_change = (
    current_pr_auc
    - reference_pr_auc
)

precision_change = (
    current_precision
    - reference_precision
)

recall_change = (
    current_recall
    - reference_recall
)


performance_alerts = []


if pr_auc_change <= -0.10:

    performance_alerts.append(
        "SIGNIFICANT PR-AUC DEGRADATION"
    )

elif pr_auc_change <= -0.05:

    performance_alerts.append(
        "MODERATE PR-AUC DEGRADATION"
    )


if recall_change <= -0.10:

    performance_alerts.append(
        "SIGNIFICANT RECALL DEGRADATION"
    )

elif recall_change <= -0.05:

    performance_alerts.append(
        "MODERATE RECALL DEGRADATION"
    )


if precision_change <= -0.10:

    performance_alerts.append(
        "SIGNIFICANT PRECISION DEGRADATION"
    )

elif precision_change <= -0.05:

    performance_alerts.append(
        "MODERATE PRECISION DEGRADATION"
    )


significant_performance_alerts = [
    alert
    for alert in performance_alerts
    if "SIGNIFICANT" in alert
]


if significant_performance_alerts:

    performance_status = (
        "REVIEW REQUIRED"
    )

elif performance_alerts:

    performance_status = (
        "MONITOR"
    )

else:

    performance_status = (
        "STABLE"
    )


# ============================================================
# CONFUSION MATRIX — CURRENT PERIOD
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        current_y,
        current_predictions,
    )
    .ravel()
)


# ============================================================
# OVERALL MODEL HEALTH
# ============================================================

statuses = [
    feature_status,
    score_status,
    performance_status,
]


if any(
    "REVIEW REQUIRED" in status
    for status in statuses
):

    overall_health = (
        "REVIEW REQUIRED"
    )

elif any(
    "MONITOR" in status
    for status in statuses
):

    overall_health = (
        "MONITOR"
    )

else:

    overall_health = (
        "HEALTHY"
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "monitor":
                "Feature Drift",

            "status":
                feature_status,

            "primary_metric":
                (
                    feature_results_df[
                        "psi"
                    ].max()
                ),
        },

        {
            "monitor":
                "Prediction Score Drift",

            "status":
                score_status,

            "primary_metric":
                score_psi,
        },

        {
            "monitor":
                "Delayed Performance",

            "status":
                performance_status,

            "primary_metric":
                current_pr_auc,
        },
    ]
)


summary_df[
    "primary_metric"
] = summary_df[
    "primary_metric"
].round(4)


# ============================================================
# DISPLAY HEALTH SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("MODEL HEALTH SUMMARY")
print("=" * 90)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE DRIFT DETAILS
# ============================================================

print("\n" + "=" * 90)
print("FEATURE DRIFT DETAILS")
print("=" * 90)

feature_display = (
    feature_results_df.copy()
)

feature_display[
    "psi"
] = feature_display[
    "psi"
].round(4)

print(
    feature_display.to_string(
        index=False
    )
)


# ============================================================
# PREDICTION SCORE DETAILS
# ============================================================

print("\n" + "=" * 90)
print("PREDICTION SCORE MONITORING")
print("=" * 90)

print(
    f"\nScore PSI: "
    f"{score_psi:.4f}"
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


# ============================================================
# PERFORMANCE DETAILS
# ============================================================

print("\n" + "=" * 90)
print("DELAYED PERFORMANCE MONITORING")
print("=" * 90)

print(
    f"\nReference PR-AUC: "
    f"{reference_pr_auc:.4f}"
)

print(
    f"Current PR-AUC:   "
    f"{current_pr_auc:.4f}"
)

print(
    f"Change:           "
    f"{pr_auc_change:+.4f}"
)


print(
    f"\nReference precision: "
    f"{reference_precision:.4f}"
)

print(
    f"Current precision:   "
    f"{current_precision:.4f}"
)

print(
    f"Change:              "
    f"{precision_change:+.4f}"
)


print(
    f"\nReference recall: "
    f"{reference_recall:.4f}"
)

print(
    f"Current recall:   "
    f"{current_recall:.4f}"
)

print(
    f"Change:           "
    f"{recall_change:+.4f}"
)


print(
    f"\nCurrent confusion matrix:"
)

print(
    f"TP={tp:,}  "
    f"FP={fp:,}  "
    f"FN={fn:,}  "
    f"TN={tn:,}"
)


# ============================================================
# MONITORING ALERTS
# ============================================================

print("\n" + "=" * 90)
print("ACTIVE MONITORING ALERTS")
print("=" * 90)


active_alerts = []


if not unexpected_feature_alerts.empty:

    for _, row in (
        unexpected_feature_alerts
        .iterrows()
    ):

        active_alerts.append(
            f"FEATURE DRIFT: "
            f"{row['feature']} "
            f"(PSI={row['psi']:.4f})"
        )


if score_status != "STABLE":

    active_alerts.append(
        f"SCORE MONITORING: "
        f"{score_status}"
    )


for alert in performance_alerts:

    active_alerts.append(
        f"PERFORMANCE: "
        f"{alert}"
    )


if not active_alerts:

    print(
        "\nNo unexpected active monitoring alerts."
    )

else:

    for alert in active_alerts:

        print(
            f"- {alert}"
        )


# ============================================================
# EXPECTED MATURITY NOTES
# ============================================================

if not maturity_feature_alerts.empty:

    print("\n" + "=" * 90)
    print("EXPECTED FEATURE MATURITY")
    print("=" * 90)

    for _, row in (
        maturity_feature_alerts
        .iterrows()
    ):

        print(
            f"\n{row['feature']}: "
            f"PSI={row['psi']:.4f}"
        )

    print(
        "\nThese features are tracked separately "
        "because their distributions naturally "
        "change as historical observations accumulate."
    )


# ============================================================
# FINAL MODEL HEALTH
# ============================================================

print("\n" + "=" * 90)
print("OVERALL MODEL HEALTH")
print("=" * 90)

print(
    f"\nFeature Drift:          "
    f"{feature_status}"
)

print(
    f"Prediction Score Drift: "
    f"{score_status}"
)

print(
    f"Model Performance:      "
    f"{performance_status}"
)

print(
    "\n----------------------------------------"
)

print(
    f"Overall Model Health:   "
    f"{overall_health}"
)

print(
    "----------------------------------------"
)


# ============================================================
# STRUCTURED HEALTH OBJECT
# ============================================================

health_report = {
    "feature_drift_status":
        feature_status,

    "prediction_score_status":
        score_status,

    "performance_status":
        performance_status,

    "overall_health":
        overall_health,

    "score_psi":
        round(
            score_psi,
            4,
        ),

    "current_pr_auc":
        round(
            current_pr_auc,
            4,
        ),

    "current_precision":
        round(
            current_precision,
            4,
        ),

    "current_recall":
        round(
            current_recall,
            4,
        ),

    "current_alert_rate":
        round(
            current_alert_rate,
            4,
        ),

    "active_alert_count":
        len(
            active_alerts
        ),
}


print("\n" + "=" * 90)
print("STRUCTURED HEALTH REPORT")
print("=" * 90)

for key, value in (
    health_report.items()
):

    print(
        f"{key}: {value}"
    )


print("\n" + "=" * 90)
print("UNIFIED MODEL HEALTH REPORT COMPLETE")
print("=" * 90)