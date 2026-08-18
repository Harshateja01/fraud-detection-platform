import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_detection")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


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


print("=" * 75)
print("ISOLATION FOREST — FRAUD ANOMALY DETECTION")
print("=" * 75)

print("\nLoading modeling data...")

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

test_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print(f"\nTrain rows: {len(train_df):,}")
print(f"Test rows:  {len(test_df):,}")


# ============================================================
# UNSUPERVISED TRAINING DATA
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

y_test = test_df["tx_fraud"]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


# ============================================================
# ISOLATION FOREST
# ============================================================

# contamination is only an estimate of anomaly prevalence.
# It is NOT using fraud labels.

model = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)


print("\nTraining Isolation Forest...")

model.fit(
    X_train_imputed
)


# ============================================================
# ANOMALY SCORES
# ============================================================

# score_samples:
# larger values = more normal
#
# We multiply by -1 so:
# larger anomaly_score = more anomalous

anomaly_score = (
    -model.score_samples(
        X_test_imputed
    )
)


# ============================================================
# RANKING METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    anomaly_score
)

pr_auc = average_precision_score(
    y_test,
    anomaly_score
)


print("\n" + "=" * 75)
print("ANOMALY RANKING PERFORMANCE")
print("=" * 75)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC:  "
    f"{pr_auc:.4f}"
)


# ============================================================
# TOP-K ANALYSIS
# ============================================================

ranked = pd.DataFrame({
    "transaction_id":
        test_df["transaction_id"].to_numpy(),

    "anomaly_score":
        anomaly_score,

    "tx_fraud":
        y_test.to_numpy(),
})

ranked = ranked.sort_values(
    "anomaly_score",
    ascending=False
).reset_index(drop=True)


total_fraud = int(
    ranked["tx_fraud"].sum()
)

fraud_rate = (
    ranked["tx_fraud"].mean()
)


K_VALUES = [
    500,
    1000,
    2500,
    5000,
    10000,
]

results = []


for k in K_VALUES:

    top_k = ranked.head(k)

    fraud_found = int(
        top_k["tx_fraud"].sum()
    )

    precision_at_k = (
        fraud_found / k
    )

    recall_at_k = (
        fraud_found / total_fraud
    )

    lift = (
        precision_at_k
        / fraud_rate
    )

    alert_rate = (
        k
        / len(ranked)
        * 100
    )

    results.append({
        "k": k,
        "fraud_found": fraud_found,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "alert_rate_pct": alert_rate,
        "lift_vs_random": lift,
    })


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 75)
print("TOP-K ANOMALY PERFORMANCE")
print("=" * 75)

display_df = results_df.copy()

display_df[
    [
        "precision_at_k",
        "recall_at_k",
        "alert_rate_pct",
        "lift_vs_random",
    ]
] = display_df[
    [
        "precision_at_k",
        "recall_at_k",
        "alert_rate_pct",
        "lift_vs_random",
    ]
].round(4)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("ANOMALY MODEL SUMMARY")
print("=" * 75)

print(
    f"Test fraud prevalence: "
    f"{fraud_rate:.4%}"
)

print(
    "\nIsolation Forest is trained "
    "without using tx_fraud labels."
)

print(
    "Fraud labels are used only "
    "for evaluation."
)

print("\n" + "=" * 75)
print("ANOMALY MODEL COMPLETE")
print("=" * 75)