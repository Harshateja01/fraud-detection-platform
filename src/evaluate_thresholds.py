import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ============================================================
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_detection")

connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "tx_amount",
    "during_weekend",
    "during_night",
    "customer_tx_count_1h",
    "customer_tx_count_6h",
    "customer_tx_count_24h",
    "customer_avg_amount_24h",
    "customer_avg_amount_7d",
    "customer_amount_deviation",
    "terminal_tx_count_24h",
    "terminal_fraud_rate_7d",
    "terminal_fraud_rate_30d",
    "terminal_history_available",
]

TARGET_COLUMN = "tx_fraud"


# ============================================================
# LOAD DATA
# ============================================================

query = """
SELECT
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

WHERE t.tx_datetime < '2018-09-01'

ORDER BY t.tx_datetime;
"""


print("=" * 70)
print("FRAUD THRESHOLD ANALYSIS")
print("=" * 70)

print("\nLoading train and validation data...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# TIME SPLIT
# ============================================================

train_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

validation_df = df[
    (df["tx_datetime"] >= "2018-08-01")
    & (df["tx_datetime"] < "2018-09-01")
].copy()


print(
    f"\nTrain rows:      "
    f"{len(train_df):,}"
)

print(
    f"Validation rows: "
    f"{len(validation_df):,}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_validation = imputer.transform(
    X_validation
)


# ============================================================
# CLASS WEIGHT
# ============================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative_count
    / positive_count
)


# ============================================================
# XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    reg_lambda=1.0,
    reg_alpha=0.0,

    tree_method="hist",

    random_state=42,
    n_jobs=-1,
)


print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    verbose=False,
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_validation
)[:, 1]


print("\nVALIDATION RANKING METRICS")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_validation, probabilities):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_validation, probabilities):.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.05,
    1.00,
    0.05
)

results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )


    true_positives = int(
        (
            (predictions == 1)
            & (y_validation == 1)
        ).sum()
    )

    false_positives = int(
        (
            (predictions == 1)
            & (y_validation == 0)
        ).sum()
    )

    false_negatives = int(
        (
            (predictions == 0)
            & (y_validation == 1)
        ).sum()
    )

    alerts = int(
        predictions.sum()
    )

    alert_rate = (
        alerts
        / len(predictions)
        * 100
    )


    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "alerts": alerts,
        "alert_rate_pct": alert_rate,
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

display_df = results_df.copy()

display_df[
    [
        "threshold",
        "precision",
        "recall",
        "f1",
        "alert_rate_pct",
    ]
] = display_df[
    [
        "threshold",
        "precision",
        "recall",
        "f1",
        "alert_rate_pct",
    ]
].round(4)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold:       "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"Precision:       "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"Recall:          "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"F1:              "
    f"{best_f1_row['f1']:.4f}"
)

print(
    f"Alerts:          "
    f"{int(best_f1_row['alerts']):,}"
)

print(
    f"Alert rate:      "
    f"{best_f1_row['alert_rate_pct']:.2f}%"
)

print(
    f"True positives:  "
    f"{int(best_f1_row['true_positives']):,}"
)

print(
    f"False positives: "
    f"{int(best_f1_row['false_positives']):,}"
)

print(
    f"False negatives: "
    f"{int(best_f1_row['false_negatives']):,}"
)


print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)