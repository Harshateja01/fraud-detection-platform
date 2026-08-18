import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================


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
# LOAD TRAIN + VALIDATION ONLY
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
print("FRAUD DETECTION — BUSINESS COST OPTIMIZATION")
print("=" * 70)



print("\nLoading data...")

with engine.connect() as connection:
    df = pd.read_sql(text(query), connection)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

validation_df = df[
    (df["tx_datetime"] >= "2018-08-01")
    & (df["tx_datetime"] < "2018-09-01")
].copy()

print(f"Train rows:      {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")


X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["tx_fraud"]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df["tx_fraud"]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_validation = imputer.transform(X_validation)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count


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
# ============================================================
# COST SENSITIVITY ANALYSIS
# ============================================================

COST_SCENARIOS = [
    {
        "name": "Low fraud cost",
        "fp_cost": 5,
        "fn_cost": 100,
    },
    {
        "name": "Moderate",
        "fp_cost": 5,
        "fn_cost": 250,
    },
    {
        "name": "Base",
        "fp_cost": 5,
        "fn_cost": 500,
    },
    {
        "name": "High fraud cost",
        "fp_cost": 5,
        "fn_cost": 1000,
    },
    {
        "name": "High review cost",
        "fp_cost": 20,
        "fn_cost": 500,
    },
]

thresholds = np.arange(
    0.05,
    1.00,
    0.01,
)

y_array = y_validation.to_numpy()

scenario_results = []

for scenario in COST_SCENARIOS:

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = int(
            ((predictions == 1) & (y_array == 1)).sum()
        )

        fp = int(
            ((predictions == 1) & (y_array == 0)).sum()
        )

        fn = int(
            ((predictions == 0) & (y_array == 1)).sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        alerts = tp + fp

        alert_rate = (
            alerts / len(y_array) * 100
        )

        total_cost = (
            fp * scenario["fp_cost"]
            + fn * scenario["fn_cost"]
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "alerts": alerts,
            "alert_rate_pct": alert_rate,
            "business_cost": total_cost,
        })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    best = threshold_df.loc[
        threshold_df["business_cost"].idxmin()
    ]

    scenario_results.append({
        "scenario": scenario["name"],
        "fp_cost": scenario["fp_cost"],
        "fn_cost": scenario["fn_cost"],
        "optimal_threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "false_positives": int(best["fp"]),
        "false_negatives": int(best["fn"]),
        "alerts": int(best["alerts"]),
        "alert_rate_pct": best["alert_rate_pct"],
        "minimum_cost": best["business_cost"],
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    scenario_results
)

print("\n" + "=" * 100)
print("COST SENSITIVITY RESULTS")
print("=" * 100)

display_df = results_df.copy()

display_df[
    [
        "optimal_threshold",
        "precision",
        "recall",
        "alert_rate_pct",
    ]
] = display_df[
    [
        "optimal_threshold",
        "precision",
        "recall",
        "alert_rate_pct",
    ]
].round(4)

print(
    display_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)

for _, row in results_df.iterrows():

    print(
        f"\n{row['scenario']}:"
    )

    print(
        f"  FP cost / FN cost: "
        f"${row['fp_cost']:.0f} / "
        f"${row['fn_cost']:.0f}"
    )

    print(
        f"  Optimal threshold: "
        f"{row['optimal_threshold']:.2f}"
    )

    print(
        f"  Recall: "
        f"{row['recall']:.2%}"
    )

    print(
        f"  Precision: "
        f"{row['precision']:.2%}"
    )

    print(
        f"  Alert rate: "
        f"{row['alert_rate_pct']:.2f}%"
    )

    print(
        f"  Estimated cost: "
        f"${row['minimum_cost']:,.0f}"
    )


print("\n" + "=" * 100)
print("COST SENSITIVITY ANALYSIS COMPLETE")
print("=" * 100)

