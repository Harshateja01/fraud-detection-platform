import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
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

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# FEATURE SETS
# ============================================================

BEHAVIORAL_FEATURES = [
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
]


DELAYED_RISK_FEATURES = [
    "terminal_fraud_rate_7d",
    "terminal_fraud_rate_30d",
    "terminal_history_available",
]


FULL_FEATURES = (
    BEHAVIORAL_FEATURES
    + DELAYED_RISK_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

query = """
SELECT
    t.transaction_id,
    t.tx_datetime,
    t.tx_amount,
    t.tx_fraud,
    t.tx_fraud_scenario,

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


print("=" * 80)
print("FRAUD MODEL — FEATURE LEAKAGE / DEPENDENCY AUDIT")
print("=" * 80)

print("\nLoading modeling data...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# CONFIRM LEAKAGE COLUMN IS NOT A MODEL FEATURE
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE SAFETY CHECK")
print("=" * 80)

print(
    "tx_fraud_scenario included in model features:",
    "tx_fraud_scenario" in FULL_FEATURES,
)

print(
    "tx_fraud included in model features:",
    "tx_fraud" in FULL_FEATURES,
)

print(
    "transaction_id included in model features:",
    "transaction_id" in FULL_FEATURES,
)

print(
    "customer_id included in model features:",
    "customer_id" in FULL_FEATURES,
)

print(
    "terminal_id included in model features:",
    "terminal_id" in FULL_FEATURES,
)


# ============================================================
# TIME SPLIT
# ============================================================

train_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

test_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nTIME SPLIT")

print(
    f"Train rows: {len(train_df):,}"
)

print(
    f"Test rows:  {len(test_df):,}"
)

print(
    f"Test fraud rate: "
    f"{test_df['tx_fraud'].mean():.4%}"
)


# ============================================================
# COMMON TARGETS
# ============================================================

y_train = train_df["tx_fraud"]
y_test = test_df["tx_fraud"]


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
# MODEL EVALUATION FUNCTION
# ============================================================

def train_and_evaluate(
    model_name,
    feature_columns,
):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print("\nFeatures:")

    for feature in feature_columns:
        print(f"- {feature}")


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]


    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",
        eval_metric="aucpr",

        scale_pos_weight=
            scale_pos_weight,

        reg_lambda=1.0,
        reg_alpha=0.0,

        tree_method="hist",

        random_state=42,
        n_jobs=-1,
    )


    print("\nTraining...")

    model.fit(
        X_train_imputed,
        y_train,
        verbose=False,
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test_imputed
        )[:, 1]
    )


    # --------------------------------------------------------
    # RANKING METRICS
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )


    print(
        f"\nROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  "
        f"{pr_auc:.4f}"
    )


    # --------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------

    ranked = pd.DataFrame({
        "probability":
            probabilities,

        "tx_fraud":
            y_test.to_numpy(),
    })

    ranked = ranked.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)


    total_fraud = int(
        ranked["tx_fraud"].sum()
    )

    base_fraud_rate = (
        ranked["tx_fraud"].mean()
    )


    top_k_results = {}

    for k in [
        500,
        1000,
        2500,
        5000,
        10000,
    ]:

        top_k = ranked.head(k)

        fraud_found = int(
            top_k["tx_fraud"].sum()
        )

        precision_at_k = (
            fraud_found / k
        )

        recall_at_k = (
            fraud_found
            / total_fraud
        )

        lift = (
            precision_at_k
            / base_fraud_rate
        )


        top_k_results[k] = {
            "fraud_found":
                fraud_found,

            "precision_at_k":
                precision_at_k,

            "recall_at_k":
                recall_at_k,

            "lift":
                lift,
        }


        print(
            f"\nTop {k:,}:"
        )

        print(
            f"  Fraud found: "
            f"{fraud_found:,}"
        )

        print(
            f"  Precision@K: "
            f"{precision_at_k:.2%}"
        )

        print(
            f"  Recall@K: "
            f"{recall_at_k:.2%}"
        )

        print(
            f"  Lift: "
            f"{lift:.1f}x"
        )


    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "top_k": top_k_results,
    }


# ============================================================
# MODEL A
# ============================================================

behavioral_results = (
    train_and_evaluate(
        "MODEL A — BEHAVIORAL FEATURES ONLY",
        BEHAVIORAL_FEATURES,
    )
)


# ============================================================
# MODEL B
# ============================================================

full_results = (
    train_and_evaluate(
        "MODEL B — BEHAVIORAL + DELAYED TERMINAL RISK",
        FULL_FEATURES,
    )
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPARISON")
print("=" * 80)

comparison = pd.DataFrame([
    {
        "model":
            behavioral_results["model"],

        "roc_auc":
            behavioral_results["roc_auc"],

        "pr_auc":
            behavioral_results["pr_auc"],

        "precision_at_1000":
            behavioral_results[
                "top_k"
            ][1000][
                "precision_at_k"
            ],

        "recall_at_1000":
            behavioral_results[
                "top_k"
            ][1000][
                "recall_at_k"
            ],

        "precision_at_2500":
            behavioral_results[
                "top_k"
            ][2500][
                "precision_at_k"
            ],

        "recall_at_2500":
            behavioral_results[
                "top_k"
            ][2500][
                "recall_at_k"
            ],
    },

    {
        "model":
            full_results["model"],

        "roc_auc":
            full_results["roc_auc"],

        "pr_auc":
            full_results["pr_auc"],

        "precision_at_1000":
            full_results[
                "top_k"
            ][1000][
                "precision_at_k"
            ],

        "recall_at_1000":
            full_results[
                "top_k"
            ][1000][
                "recall_at_k"
            ],

        "precision_at_2500":
            full_results[
                "top_k"
            ][2500][
                "precision_at_k"
            ],

        "recall_at_2500":
            full_results[
                "top_k"
            ][2500][
                "recall_at_k"
            ],
    },
])


numeric_columns = [
    "roc_auc",
    "pr_auc",
    "precision_at_1000",
    "recall_at_1000",
    "precision_at_2500",
    "recall_at_2500",
]

comparison[
    numeric_columns
] = comparison[
    numeric_columns
].round(4)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

pr_improvement = (
    full_results["pr_auc"]
    - behavioral_results["pr_auc"]
)


print("\n" + "=" * 80)
print("AUDIT INTERPRETATION")
print("=" * 80)

print(
    "\nPR-AUC improvement from delayed "
    "terminal-risk features:"
)

print(
    f"{pr_improvement:+.4f}"
)

print(
    "\nImportant:"
)

print(
    "- tx_fraud_scenario was NOT used."
)

print(
    "- Current transaction fraud label "
    "was NOT used as a feature."
)

print(
    "- Terminal fraud rates use a "
    "7-day label-availability delay."
)

print(
    "- Performance gains from terminal "
    "risk should therefore be described "
    "as historical-risk signal, not "
    "future-label leakage."
)

print(
    "\nHowever, this is a synthetic dataset. "
    "Persistent fraudulent-terminal patterns "
    "may be stronger than they would be in "
    "a real production portfolio."
)


print("\n" + "=" * 80)
print("FEATURE AUDIT COMPLETE")
print("=" * 80)