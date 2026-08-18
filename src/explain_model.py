import os

import numpy as np
import pandas as pd
import shap

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
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
# LOAD TRAIN + TEST DATA
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
print("XGBOOST FRAUD MODEL — GLOBAL SHAP EXPLAINABILITY")
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
# SHAP SAMPLE
# ============================================================

# We do not need all 287k test rows for global SHAP.
# A representative sample is enough and keeps runtime manageable.

SHAP_SAMPLE_SIZE = 10000

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X_test_imputed)
)

rng = np.random.default_rng(
    42
)

sample_indices = rng.choice(
    len(X_test_imputed),
    size=sample_size,
    replace=False,
)

X_shap = X_test_imputed[
    sample_indices
]


print(
    f"\nCalculating SHAP values for "
    f"{sample_size:,} test transactions..."
)


# ============================================================
# TREE EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)

shap_values = np.asarray(
    shap_values
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0,
)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "mean_abs_shap": mean_abs_shap,
})

importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 75)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# TOP FIVE
# ============================================================

print("\n" + "=" * 75)
print("TOP FIVE GLOBAL MODEL DRIVERS")
print("=" * 75)

for rank, row in (
    importance_df
    .head(5)
    .iterrows()
):

    print(
        f"{rank + 1}. "
        f"{row['feature']} "
        f"(mean |SHAP| = "
        f"{row['mean_abs_shap']:.6f})"
    )


print("\n" + "=" * 75)
print("GLOBAL SHAP ANALYSIS COMPLETE")
print("=" * 75)