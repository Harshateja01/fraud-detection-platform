import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
# MODEL FEATURES
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

print("=" * 70)
print("LOGISTIC REGRESSION FRAUD BASELINE")
print("=" * 70)

print("\nLoading modeling dataset...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# TIME-BASED SPLIT
# ============================================================

train_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

validation_df = df[
    (df["tx_datetime"] >= "2018-08-01")
    & (df["tx_datetime"] < "2018-09-01")
].copy()

test_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nTIME SPLIT")

print(
    f"Train:      {len(train_df):,} rows "
    f"({train_df['tx_datetime'].min()} "
    f"to {train_df['tx_datetime'].max()})"
)

print(
    f"Validation: {len(validation_df):,} rows "
    f"({validation_df['tx_datetime'].min()} "
    f"to {validation_df['tx_datetime'].max()})"
)

print(
    f"Test:       {len(test_df):,} rows "
    f"({test_df['tx_datetime'].min()} "
    f"to {test_df['tx_datetime'].max()})"
)


# ============================================================
# FRAUD RATES
# ============================================================

print("\nFRAUD RATES")

for name, split_df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:
    rate = (
        split_df[TARGET_COLUMN].mean()
        * 100
    )

    print(
        f"{name}: {rate:.4f}%"
    )


# ============================================================
# X / Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[
    FEATURE_COLUMNS
]
y_validation = validation_df[
    TARGET_COLUMN
]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = FEATURE_COLUMNS

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            ),
            numeric_features,
        )
    ],
    remainder="drop",
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    split_name,
    X,
    y,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)


    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )


    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

validation_metrics = evaluate_model(
    "VALIDATION PERFORMANCE",
    X_validation,
    y_validation,
)


# ============================================================
# TEST PERFORMANCE
# ============================================================

test_metrics = evaluate_model(
    "TEST PERFORMANCE",
    X_test,
    y_test,
)


print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)