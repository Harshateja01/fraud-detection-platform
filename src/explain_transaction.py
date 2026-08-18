import os

import numpy as np
import pandas as pd
import shap

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

FINAL_THRESHOLD = 0.46


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
# LOAD MODELING DATA
# ============================================================

query = """
SELECT
    t.transaction_id,
    t.tx_datetime,
    t.customer_id,
    t.terminal_id,
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
print("INDIVIDUAL FRAUD TRANSACTION EXPLANATION")
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


# ============================================================
# TRAINING DATA
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["tx_fraud"]

X_test = test_df[FEATURE_COLUMNS]


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
# TRAIN CHAMPION MODEL
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


print("\nTraining champion XGBoost model...")

model.fit(
    X_train_imputed,
    y_train,
    verbose=False,
)


# ============================================================
# USER INPUT
# ============================================================

min_transaction = int(
    test_df["transaction_id"].min()
)

max_transaction = int(
    test_df["transaction_id"].max()
)

print(
    f"\nSeptember test transaction IDs range from "
    f"{min_transaction:,} to {max_transaction:,}"
)

transaction_id = int(
    input(
        "\nEnter transaction ID to explain: "
    )
)


# ============================================================
# FIND TRANSACTION
# ============================================================

matching_rows = test_df[
    test_df["transaction_id"] == transaction_id
]

if matching_rows.empty:

    print(
        "\nTransaction not found in the "
        "September test period."
    )

    raise SystemExit


transaction_row = matching_rows.iloc[0]

test_position = (
    test_df
    .index
    .get_loc(
        matching_rows.index[0]
    )
)

transaction_features = (
    X_test_imputed[
        [test_position]
    ]
)


# ============================================================
# PREDICTION
# ============================================================

fraud_probability = (
    model.predict_proba(
        transaction_features
    )[0, 1]
)

alert = (
    fraud_probability
    >= FINAL_THRESHOLD
)


if fraud_probability >= 0.90:
    risk_level = "CRITICAL"

elif fraud_probability >= 0.70:
    risk_level = "HIGH"

elif fraud_probability >= FINAL_THRESHOLD:
    risk_level = "MEDIUM"

else:
    risk_level = "LOW"


# ============================================================
# SHAP
# ============================================================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    transaction_features
)

shap_values = np.asarray(
    shap_values
)

shap_values = np.squeeze(
    shap_values
)


# ============================================================
# BUILD EXPLANATION TABLE
# ============================================================

original_values = matching_rows[
    FEATURE_COLUMNS
].iloc[0]


shap_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "value": original_values.values,
    "shap_value": shap_values,
})

shap_df["abs_shap"] = (
    shap_df["shap_value"].abs()
)

shap_df = shap_df.sort_values(
    "abs_shap",
    ascending=False,
)


# ============================================================
# DISPLAY TRANSACTION
# ============================================================

print("\n" + "=" * 75)
print("TRANSACTION RISK ASSESSMENT")
print("=" * 75)

print(
    f"\nTransaction ID: "
    f"{transaction_id:,}"
)

print(
    f"Transaction time: "
    f"{transaction_row['tx_datetime']}"
)

print(
    f"Customer ID: "
    f"{int(transaction_row['customer_id'])}"
)

print(
    f"Terminal ID: "
    f"{int(transaction_row['terminal_id'])}"
)

print(
    f"Transaction amount: "
    f"${transaction_row['tx_amount']:,.2f}"
)

print(
    f"\nFraud probability: "
    f"{fraud_probability:.2%}"
)

print(
    f"Risk level: "
    f"{risk_level}"
)

print(
    f"Alert threshold: "
    f"{FINAL_THRESHOLD:.0%}"
)

print(
    f"Alert generated: "
    f"{'YES' if alert else 'NO'}"
)

print(
    f"Actual fraud label: "
    f"{int(transaction_row['tx_fraud'])}"
)


# ============================================================
# FACTORS INCREASING RISK
# ============================================================

increasing = (
    shap_df[
        shap_df["shap_value"] > 0
    ]
    .head(5)
)


print("\n" + "=" * 75)
print("TOP FACTORS INCREASING FRAUD RISK")
print("=" * 75)

if increasing.empty:

    print(
        "\nNo major positive SHAP contributions."
    )

else:

    print(
        increasing[
            [
                "feature",
                "value",
                "shap_value",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FACTORS REDUCING RISK
# ============================================================

reducing = (
    shap_df[
        shap_df["shap_value"] < 0
    ]
    .head(5)
)


print("\n" + "=" * 75)
print("TOP FACTORS REDUCING FRAUD RISK")
print("=" * 75)

if reducing.empty:

    print(
        "\nNo major negative SHAP contributions."
    )

else:

    print(
        reducing[
            [
                "feature",
                "value",
                "shap_value",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SIMPLE INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("INTERPRETATION")
print("=" * 75)

for _, row in (
    shap_df
    .head(5)
    .iterrows()
):

    direction = (
        "increases"
        if row["shap_value"] > 0
        else "reduces"
    )

    value = row["value"]

    if pd.isna(value):
        value_text = "missing"

    elif isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        value_text = f"{value:.4f}"

    else:
        value_text = str(value)

    print(
        f"- {row['feature']} = "
        f"{value_text} "
        f"{direction} predicted fraud risk"
    )


print("\n" + "=" * 75)
print("TRANSACTION EXPLANATION COMPLETE")
print("=" * 75)