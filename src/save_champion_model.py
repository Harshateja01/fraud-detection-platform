import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

FINAL_THRESHOLD = 0.46

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
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = (
    project_root
    / "models"
)

models_dir.mkdir(
    parents=True,
    exist_ok=True
)

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
# LOAD TRAINING DATA
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

WHERE t.tx_datetime < '2018-08-01'

ORDER BY t.tx_datetime;
"""


print("=" * 75)
print("SAVE CHAMPION FRAUD MODEL")
print("=" * 75)

print("\nLoading training data...")

with engine.connect() as connection:
    train_df = pd.read_sql(
        text(query),
        connection,
    )

print(
    f"Training rows loaded: "
    f"{len(train_df):,}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    "tx_fraud"
]


# ============================================================
# IMPUTER
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
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


print("\nTraining class balance")

print(
    f"Legitimate: "
    f"{negative_count:,}"
)

print(
    f"Fraud:      "
    f"{positive_count:,}"
)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# CHAMPION MODEL
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
# METADATA
# ============================================================

metadata = {
    "model_name":
        "XGBoost Fraud Detection Champion",

    "model_type":
        "XGBClassifier",

    "feature_columns":
        FEATURE_COLUMNS,

    "decision_threshold":
        FINAL_THRESHOLD,

    "training_end_date":
        "2018-07-31",

    "validation_period":
        "2018-08-01 to 2018-08-31",

    "test_period":
        "2018-09-01 to 2018-09-30",

    "label_delay_days":
        7,

    "test_roc_auc":
        0.8866,

    "test_pr_auc":
        0.6613,

    "test_precision":
        0.1873,

    "test_recall":
        0.7550,

    "test_f1":
        0.3002,

    "test_alert_rate_pct":
        3.57,
}


# ============================================================
# SAVE ARTIFACTS
# ============================================================

joblib.dump(
    model,
    model_path
)

joblib.dump(
    imputer,
    imputer_path
)

joblib.dump(
    metadata,
    metadata_path
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("MODEL ARTIFACTS SAVED")
print("=" * 75)

print(
    f"\nModel:\n{model_path}"
)

print(
    f"\nImputer:\n{imputer_path}"
)

print(
    f"\nMetadata:\n{metadata_path}"
)

print(
    f"\nDecision threshold: "
    f"{FINAL_THRESHOLD:.2f}"
)

print(
    f"Number of features: "
    f"{len(FEATURE_COLUMNS)}"
)

print("\n" + "=" * 75)
print("MODEL PERSISTENCE COMPLETE")
print("=" * 75)