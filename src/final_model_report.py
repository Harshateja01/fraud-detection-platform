import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ============================================================
# FINAL MODEL POLICY
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
# LOAD FULL MODELING DATA
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
print("FINAL FRAUD DETECTION MODEL REPORT")
print("=" * 75)

print("\nLoading modeling dataset...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# FINAL CHRONOLOGICAL SPLIT
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


print("\nFINAL DATA SPLIT")

print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(validation_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)

print(
    f"\nFrozen decision threshold: "
    f"{FINAL_THRESHOLD:.2f}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["tx_fraud"]

X_validation = validation_df[
    FEATURE_COLUMNS
]
y_validation = validation_df[
    "tx_fraud"
]

X_test = test_df[
    FEATURE_COLUMNS
]
y_test = test_df[
    "tx_fraud"
]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_validation_imputed = imputer.transform(
    X_validation
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
# FINAL XGBOOST MODEL
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


print("\nTraining final XGBoost model...")

model.fit(
    X_train_imputed,
    y_train,
    verbose=False,
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_split(
    split_name,
    X,
    y,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= FINAL_THRESHOLD
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


    cm = confusion_matrix(
        y,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    alerts = (
        tp + fp
    )

    alert_rate = (
        alerts
        / len(y)
        * 100
    )


    print("\n" + "=" * 75)
    print(split_name)
    print("=" * 75)

    print(
        f"ROC-AUC:       "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC:        "
        f"{pr_auc:.4f}"
    )

    print(
        f"Precision:     "
        f"{precision:.4f}"
    )

    print(
        f"Recall:        "
        f"{recall:.4f}"
    )

    print(
        f"F1:            "
        f"{f1:.4f}"
    )

    print(
        f"Alert rate:    "
        f"{alert_rate:.2f}%"
    )

    print(
        f"True positives:  "
        f"{tp:,}"
    )

    print(
        f"False positives: "
        f"{fp:,}"
    )

    print(
        f"False negatives: "
        f"{fn:,}"
    )

    print(
        f"True negatives:  "
        f"{tn:,}"
    )


    print("\nConfusion Matrix:")

    print(cm)


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
        "alert_rate": alert_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# VALIDATION CHECK
# ============================================================

validation_metrics = evaluate_split(
    "VALIDATION — FROZEN THRESHOLD",
    X_validation_imputed,
    y_validation,
)


# ============================================================
# FINAL UNTOUCHED TEST
# ============================================================

test_metrics = evaluate_split(
    "FINAL SEPTEMBER TEST PERFORMANCE",
    X_test_imputed,
    y_test,
)


# ============================================================
# BUSINESS COST ON TEST SET
# ============================================================

FALSE_POSITIVE_COST = 5
FALSE_NEGATIVE_COST = 500

test_business_cost = (
    test_metrics["fp"]
    * FALSE_POSITIVE_COST
    +
    test_metrics["fn"]
    * FALSE_NEGATIVE_COST
)


print("\n" + "=" * 75)
print("TEST BUSINESS IMPACT")
print("=" * 75)

print(
    f"Assumed FP cost: "
    f"${FALSE_POSITIVE_COST}"
)

print(
    f"Assumed FN cost: "
    f"${FALSE_NEGATIVE_COST}"
)

print(
    f"Estimated test cost: "
    f"${test_business_cost:,.0f}"
)


# ============================================================
# MODEL FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
)


print("\n" + "=" * 75)
print("TOP MODEL FEATURES")
print("=" * 75)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SUMMARY")
print("=" * 75)

print(
    "\nChampion model: XGBoost"
)

print(
    f"Operating threshold: "
    f"{FINAL_THRESHOLD:.2f}"
)

print(
    f"Test ROC-AUC: "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"Test PR-AUC: "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Test precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test recall: "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test F1: "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Test alert rate: "
    f"{test_metrics['alert_rate']:.2f}%"
)

print("\n" + "=" * 75)
print("FINAL MODEL EVALUATION COMPLETE")
print("=" * 75)