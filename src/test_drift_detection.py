import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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
# FEATURES TO MONITOR
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


EXPECTED_MATURITY_FEATURES = {
    "terminal_history_available",
}


# ============================================================
# LOAD DATA
# ============================================================

query = """
SELECT
    t.tx_datetime,
    t.tx_amount,

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

ORDER BY t.tx_datetime;
"""


print("=" * 80)
print("FRAUD MODEL — DRIFT STRESS TEST")
print("=" * 80)

print("\nLoading feature data...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# REFERENCE + BASE CURRENT PERIOD
# ============================================================

reference_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

base_current_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nBASE DATA")

print(
    f"Reference rows: "
    f"{len(reference_df):,}"
)

print(
    f"September rows: "
    f"{len(base_current_df):,}"
)


# ============================================================
# CREATE ARTIFICIALLY DRIFTED BATCH
# ============================================================

drifted_df = (
    base_current_df.copy()
)


# ------------------------------------------------------------
# 1. TRANSACTION AMOUNT SHIFT
#
# Simulate materially larger transaction values.
# ------------------------------------------------------------

drifted_df[
    "tx_amount"
] = (
    drifted_df[
        "tx_amount"
    ] * 2.5
)


# ------------------------------------------------------------
# 2. CUSTOMER VELOCITY SHIFT
#
# Simulate much more frequent customer activity.
# ------------------------------------------------------------

drifted_df[
    "customer_tx_count_1h"
] = (
    drifted_df[
        "customer_tx_count_1h"
    ] * 3
    + 2
)

drifted_df[
    "customer_tx_count_6h"
] = (
    drifted_df[
        "customer_tx_count_6h"
    ] * 2
    + 3
)

drifted_df[
    "customer_tx_count_24h"
] = (
    drifted_df[
        "customer_tx_count_24h"
    ] * 2
    + 5
)


# ------------------------------------------------------------
# 3. CUSTOMER SPENDING BASELINE SHIFT
# ------------------------------------------------------------

drifted_df[
    "customer_avg_amount_24h"
] = (
    drifted_df[
        "customer_avg_amount_24h"
    ] * 1.8
)

drifted_df[
    "customer_avg_amount_7d"
] = (
    drifted_df[
        "customer_avg_amount_7d"
    ] * 1.8
)


# ------------------------------------------------------------
# 4. AMOUNT DEVIATION SHIFT
# ------------------------------------------------------------

drifted_df[
    "customer_amount_deviation"
] = (
    drifted_df[
        "customer_amount_deviation"
    ] * 2.5
    + 75
)


# ------------------------------------------------------------
# 5. TERMINAL VELOCITY SHIFT
# ------------------------------------------------------------

drifted_df[
    "terminal_tx_count_24h"
] = (
    drifted_df[
        "terminal_tx_count_24h"
    ] * 2
    + 3
)


# ------------------------------------------------------------
# 6. TERMINAL RISK SHIFT
#
# Increase historical terminal risk while keeping
# values inside [0, 1].
# ------------------------------------------------------------

drifted_df[
    "terminal_fraud_rate_7d"
] = (
    drifted_df[
        "terminal_fraud_rate_7d"
    ]
    .fillna(0)
    * 2
    + 0.08
).clip(
    lower=0,
    upper=1,
)

drifted_df[
    "terminal_fraud_rate_30d"
] = (
    drifted_df[
        "terminal_fraud_rate_30d"
    ]
    .fillna(0)
    * 2
    + 0.05
).clip(
    lower=0,
    upper=1,
)


print(
    "\nArtificial drift injected into "
    "transaction amount, customer velocity, "
    "customer spending behavior, terminal velocity, "
    "and terminal risk."
)


# ============================================================
# PSI FUNCTION
# ============================================================

def calculate_psi(
    reference_series,
    current_series,
    bins=10,
):

    reference = pd.to_numeric(
        reference_series,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current_series,
        errors="coerce",
    ).dropna()


    if len(reference) == 0 or len(current) == 0:
        return np.nan


    # --------------------------------------------------------
    # CONSTANT FEATURES
    # --------------------------------------------------------

    if reference.nunique() <= 1:

        reference_value = (
            reference.iloc[0]
        )

        current_same = (
            current == reference_value
        ).mean()

        if current_same == 1:
            return 0.0

        return 1.0


    # --------------------------------------------------------
    # REFERENCE QUANTILE BINS
    # --------------------------------------------------------

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    bin_edges = np.unique(
        reference.quantile(
            quantiles
        ).values
    )


    # --------------------------------------------------------
    # FALLBACK FOR LOW-CARDINALITY FEATURES
    # --------------------------------------------------------

    if len(bin_edges) < 3:

        minimum = min(
            reference.min(),
            current.min(),
        )

        maximum = max(
            reference.max(),
            current.max(),
        )

        if minimum == maximum:
            return 0.0

        bin_edges = np.linspace(
            minimum,
            maximum,
            bins + 1,
        )


    # --------------------------------------------------------
    # INCLUDE OUT-OF-RANGE VALUES
    # --------------------------------------------------------

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf


    reference_bins = pd.cut(
        reference,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    current_bins = pd.cut(
        current,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )


    reference_pct = (
        reference_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    current_pct = (
        current_bins
        .value_counts(
            normalize=True,
            sort=False,
        )
    )


    # --------------------------------------------------------
    # ALIGN BINS
    # --------------------------------------------------------

    all_bins = reference_pct.index.union(
        current_pct.index
    )

    reference_pct = (
        reference_pct
        .reindex(
            all_bins,
            fill_value=0,
        )
    )

    current_pct = (
        current_pct
        .reindex(
            all_bins,
            fill_value=0,
        )
    )


    # --------------------------------------------------------
    # AVOID LOG ZERO
    # --------------------------------------------------------

    epsilon = 0.000001

    reference_pct = (
        reference_pct.clip(
            lower=epsilon
        )
    )

    current_pct = (
        current_pct.clip(
            lower=epsilon
        )
    )


    psi_values = (
        current_pct
        - reference_pct
    ) * np.log(
        current_pct
        / reference_pct
    )


    return float(
        psi_values.sum()
    )


# ============================================================
# DRIFT CLASSIFICATION
# ============================================================

def classify_drift(
    psi
):

    if pd.isna(psi):
        return "UNKNOWN"

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "MODERATE DRIFT"

    return "SIGNIFICANT DRIFT"


# ============================================================
# EVALUATE ARTIFICIALLY DRIFTED DATA
# ============================================================

results = []


for feature in FEATURE_COLUMNS:

    psi = calculate_psi(
        reference_df[feature],
        drifted_df[feature],
        bins=10,
    )

    results.append({
        "feature":
            feature,

        "psi":
            psi,

        "drift_status":
            classify_drift(
                psi
            ),

        "reference_mean":
            pd.to_numeric(
                reference_df[feature],
                errors="coerce",
            ).mean(),

        "drifted_mean":
            pd.to_numeric(
                drifted_df[feature],
                errors="coerce",
            ).mean(),
    })


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "psi",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================

display_df = (
    results_df.copy()
)

display_df[
    [
        "psi",
        "reference_mean",
        "drifted_mean",
    ]
] = display_df[
    [
        "psi",
        "reference_mean",
        "drifted_mean",
    ]
].round(4)


print("\n" + "=" * 100)
print("DRIFT STRESS TEST RESULTS")
print("=" * 100)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# UNEXPECTED MATERIAL DRIFT
# ============================================================

material_alerts = results_df[
    (
        results_df[
            "drift_status"
        ] != "STABLE"
    )
    &
    (
        ~results_df[
            "feature"
        ].isin(
            EXPECTED_MATURITY_FEATURES
        )
    )
]


significant_material_alerts = (
    material_alerts[
        material_alerts[
            "drift_status"
        ] == "SIGNIFICANT DRIFT"
    ]
)


# ============================================================
# OVERALL STATUS
# ============================================================

if not significant_material_alerts.empty:

    overall_status = (
        "REVIEW REQUIRED"
    )

elif not material_alerts.empty:

    overall_status = (
        "MONITOR"
    )

else:

    overall_status = (
        "STABLE"
    )


# ============================================================
# SUMMARY
# ============================================================

stable_count = int(
    (
        results_df[
            "drift_status"
        ] == "STABLE"
    ).sum()
)

moderate_count = int(
    (
        results_df[
            "drift_status"
        ] == "MODERATE DRIFT"
    ).sum()
)

significant_count = int(
    (
        results_df[
            "drift_status"
        ] == "SIGNIFICANT DRIFT"
    ).sum()
)


print("\n" + "=" * 80)
print("STRESS TEST SUMMARY")
print("=" * 80)

print(
    f"Stable features:            "
    f"{stable_count}"
)

print(
    f"Moderate drift features:    "
    f"{moderate_count}"
)

print(
    f"Significant drift features: "
    f"{significant_count}"
)


print("\n" + "=" * 80)
print("UNEXPECTED DRIFT ALERTS")
print("=" * 80)


if material_alerts.empty:

    print(
        "\nNo unexpected drift detected."
    )

else:

    print(
        material_alerts[
            [
                "feature",
                "psi",
                "drift_status",
            ]
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 80)
print("EXPECTED MONITORING RESPONSE")
print("=" * 80)

print(
    "\nExpected status for this synthetic "
    "drift scenario: REVIEW REQUIRED"
)


print("\n" + "=" * 80)
print("ACTUAL MONITORING RESPONSE")
print("=" * 80)

print(
    f"\nMonitoring status: "
    f"{overall_status}"
)


# ============================================================
# AUTOMATED ASSERTION
# ============================================================

assert (
    overall_status
    == "REVIEW REQUIRED"
), (
    "Drift detector failed stress test: "
    f"received {overall_status}"
)


assert (
    len(
        significant_material_alerts
    )
    > 0
), (
    "Expected at least one significant "
    "unexpected drift feature."
)


print("\n✅ DRIFT DETECTOR STRESS TEST PASSED")


print("\n" + "=" * 80)
print("DRIFT STRESS TEST COMPLETE")
print("=" * 80)