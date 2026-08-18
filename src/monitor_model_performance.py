import os
from pathlib import Path

import joblib
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
# LOAD LABELED DATA
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


print("=" * 85)
print("FRAUD MODEL — DELAYED LABEL PERFORMANCE MONITORING")
print("=" * 85)

print("\nLoading labeled transaction data...")

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
# MONTHLY PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(
    dataframe,
):

    y_true = (
        dataframe[
            "tx_fraud"
        ]
        .astype(int)
        .to_numpy()
    )

    probabilities = score_dataframe(
        dataframe
    )

    predictions = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(int)


    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
        )
        .ravel()
    )


    fraud_rate = (
        y_true.mean()
    )

    alert_rate = (
        predictions.mean()
    )


    return {
        "transactions":
            len(dataframe),

        "fraud_rate":
            fraud_rate,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "alert_rate":
            alert_rate,

        "tp":
            int(tp),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tn":
            int(tn),
    }


# ============================================================
# CREATE MONTH COLUMN
# ============================================================

df[
    "month"
] = (
    df[
        "tx_datetime"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


# ============================================================
# MONTHLY MONITORING
# ============================================================

monitoring_rows = []


for month in sorted(
    df["month"].unique()
):

    month_df = df[
        df["month"] == month
    ].copy()


    # Need both classes for ROC-AUC / PR-AUC monitoring

    if (
        month_df[
            "tx_fraud"
        ]
        .nunique()
        < 2
    ):

        print(
            f"\nSkipping {month}: "
            "only one target class present."
        )

        continue


    metrics = calculate_metrics(
        month_df
    )

    metrics[
        "month"
    ] = month

    monitoring_rows.append(
        metrics
    )


results_df = pd.DataFrame(
    monitoring_rows
)


# ============================================================
# DISPLAY MONTHLY RESULTS
# ============================================================

display_df = results_df.copy()

display_df[
    [
        "fraud_rate",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "alert_rate",
    ]
] = display_df[
    [
        "fraud_rate",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "alert_rate",
    ]
].round(4)


display_columns = [
    "month",
    "transactions",
    "fraud_rate",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "alert_rate",
    "tp",
    "fp",
    "fn",
    "tn",
]


print("\n" + "=" * 125)
print("MONTHLY MODEL PERFORMANCE")
print("=" * 125)

print(
    display_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# REFERENCE PERFORMANCE
# ============================================================

reference_df = results_df[
    results_df[
        "month"
    ].isin(
        [
            "2018-04",
            "2018-05",
            "2018-06",
            "2018-07",
        ]
    )
].copy()


current_df = results_df[
    results_df[
        "month"
    ] == "2018-09"
].copy()


if reference_df.empty:

    raise RuntimeError(
        "No reference monitoring months found."
    )


if current_df.empty:

    raise RuntimeError(
        "September monitoring period not found."
    )


# ============================================================
# REFERENCE BASELINES
# ============================================================

reference_pr_auc = (
    reference_df[
        "pr_auc"
    ].mean()
)

reference_recall = (
    reference_df[
        "recall"
    ].mean()
)

reference_precision = (
    reference_df[
        "precision"
    ].mean()
)

reference_alert_rate = (
    reference_df[
        "alert_rate"
    ].mean()
)

reference_fraud_rate = (
    reference_df[
        "fraud_rate"
    ].mean()
)


current_row = (
    current_df.iloc[0]
)


# ============================================================
# PERFORMANCE CHANGES
# ============================================================

pr_auc_change = (
    current_row[
        "pr_auc"
    ]
    - reference_pr_auc
)

recall_change = (
    current_row[
        "recall"
    ]
    - reference_recall
)

precision_change = (
    current_row[
        "precision"
    ]
    - reference_precision
)

alert_rate_change = (
    current_row[
        "alert_rate"
    ]
    - reference_alert_rate
)

fraud_rate_change = (
    current_row[
        "fraud_rate"
    ]
    - reference_fraud_rate
)


# ============================================================
# MONITORING RULES
# ============================================================

alerts = []


# PR-AUC degradation

if pr_auc_change <= -0.10:

    alerts.append(
        "SIGNIFICANT PR-AUC DEGRADATION"
    )

elif pr_auc_change <= -0.05:

    alerts.append(
        "MODERATE PR-AUC DEGRADATION"
    )


# Recall degradation

if recall_change <= -0.10:

    alerts.append(
        "SIGNIFICANT RECALL DEGRADATION"
    )

elif recall_change <= -0.05:

    alerts.append(
        "MODERATE RECALL DEGRADATION"
    )


# Precision degradation

if precision_change <= -0.10:

    alerts.append(
        "SIGNIFICANT PRECISION DEGRADATION"
    )

elif precision_change <= -0.05:

    alerts.append(
        "MODERATE PRECISION DEGRADATION"
    )


# Alert-volume change

if abs(
    alert_rate_change
) >= 0.02:

    alerts.append(
        "MATERIAL ALERT-RATE SHIFT"
    )


# Fraud prevalence change

if abs(
    fraud_rate_change
) >= 0.005:

    alerts.append(
        "MATERIAL FRAUD-RATE SHIFT"
    )


# ============================================================
# STATUS
# ============================================================

significant_alerts = [
    alert
    for alert in alerts
    if "SIGNIFICANT" in alert
]


if significant_alerts:

    overall_status = (
        "REVIEW REQUIRED"
    )

elif alerts:

    overall_status = (
        "MONITOR"
    )

else:

    overall_status = (
        "STABLE"
    )


# ============================================================
# DISPLAY REFERENCE VS CURRENT
# ============================================================

print("\n" + "=" * 85)
print("REFERENCE VS CURRENT PERFORMANCE")
print("=" * 85)

print(
    f"\nReference average PR-AUC: "
    f"{reference_pr_auc:.4f}"
)

print(
    f"September PR-AUC:         "
    f"{current_row['pr_auc']:.4f}"
)

print(
    f"PR-AUC change:            "
    f"{pr_auc_change:+.4f}"
)


print(
    f"\nReference average recall: "
    f"{reference_recall:.4f}"
)

print(
    f"September recall:          "
    f"{current_row['recall']:.4f}"
)

print(
    f"Recall change:             "
    f"{recall_change:+.4f}"
)


print(
    f"\nReference average precision: "
    f"{reference_precision:.4f}"
)

print(
    f"September precision:         "
    f"{current_row['precision']:.4f}"
)

print(
    f"Precision change:             "
    f"{precision_change:+.4f}"
)


print(
    f"\nReference alert rate: "
    f"{reference_alert_rate:.2%}"
)

print(
    f"September alert rate: "
    f"{current_row['alert_rate']:.2%}"
)

print(
    f"Alert-rate change:    "
    f"{alert_rate_change * 100:+.2f} "
    f"percentage points"
)


print(
    f"\nReference fraud rate: "
    f"{reference_fraud_rate:.2%}"
)

print(
    f"September fraud rate: "
    f"{current_row['fraud_rate']:.2%}"
)

print(
    f"Fraud-rate change:     "
    f"{fraud_rate_change * 100:+.2f} "
    f"percentage points"
)


# ============================================================
# MONITORING ALERTS
# ============================================================

print("\n" + "=" * 85)
print("PERFORMANCE MONITORING ALERTS")
print("=" * 85)

if not alerts:

    print(
        "\nNo material model-performance "
        "degradation detected."
    )

else:

    for alert in alerts:

        print(
            f"- {alert}"
        )


# ============================================================
# OVERALL STATUS
# ============================================================

print("\n" + "=" * 85)
print("OVERALL PERFORMANCE MONITORING STATUS")
print("=" * 85)

print(
    f"\nModel performance monitoring status: "
    f"{overall_status}"
)


# ============================================================
# MONITORING POLICY
# ============================================================

print("\n" + "=" * 85)
print("MONITORING POLICY")
print("=" * 85)

print(
    "\nPR-AUC / Recall / Precision:"
)

print(
    "Drop of 5–10 percentage points → MONITOR"
)

print(
    "Drop of 10+ percentage points   → REVIEW REQUIRED"
)

print(
    "\nAbsolute alert-rate change "
    ">= 2 percentage points "
    "triggers monitoring."
)

print(
    "Absolute fraud-rate change "
    ">= 0.5 percentage points "
    "triggers monitoring."
)

print(
    "\nPerformance monitoring should only "
    "run after fraud labels have matured."
)


print("\n" + "=" * 85)
print("DELAYED LABEL PERFORMANCE MONITORING COMPLETE")
print("=" * 85)