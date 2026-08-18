import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
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
print("FRAUD MODEL — PRECISION@K / RECALL@K")
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
# X / Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["tx_fraud"]

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
# CLASS IMBALANCE
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
# MODEL
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
    X_train_imputed,
    y_train,
    verbose=False,
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test_imputed
)[:, 1]


print("\nTEST RANKING METRICS")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, probabilities):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_test, probabilities):.4f}"
)


# ============================================================
# BUILD RANKED TEST DATA
# ============================================================

ranked = pd.DataFrame({
    "transaction_id": test_df["transaction_id"].to_numpy(),
    "fraud_probability": probabilities,
    "tx_fraud": y_test.to_numpy(),
})

ranked = ranked.sort_values(
    "fraud_probability",
    ascending=False,
).reset_index(drop=True)


total_fraud = int(
    ranked["tx_fraud"].sum()
)

print(
    f"\nTotal fraud transactions in test set: "
    f"{total_fraud:,}"
)


# ============================================================
# PRECISION@K / RECALL@K
# ============================================================

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

    alert_rate = (
        k / len(ranked) * 100
    )

    lift = (
        precision_at_k
        / ranked["tx_fraud"].mean()
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


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("TOP-K INVESTIGATION PERFORMANCE")
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
print("INVESTIGATION CAPACITY SUMMARY")
print("=" * 75)

for _, row in results_df.iterrows():

    print(
        f"\nTop {int(row['k']):,} alerts:"
    )

    print(
        f"  Fraud found: "
        f"{int(row['fraud_found']):,}"
    )

    print(
        f"  Precision@K: "
        f"{row['precision_at_k']:.2%}"
    )

    print(
        f"  Recall@K: "
        f"{row['recall_at_k']:.2%}"
    )

    print(
        f"  Alert rate: "
        f"{row['alert_rate_pct']:.2f}%"
    )

    print(
        f"  Lift vs random: "
        f"{row['lift_vs_random']:.1f}x"
    )


print("\n" + "=" * 75)
print("TOP-K ANALYSIS COMPLETE")
print("=" * 75)