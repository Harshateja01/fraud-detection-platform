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


# ============================================================
# FEATURES WITH EXPECTED MATURITY EFFECTS
# ============================================================

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
print("FRAUD MODEL — FEATURE DRIFT MONITORING")
print("=" * 80)

print("\nLoading feature data...")

with engine.connect() as connection:
    df = pd.read_sql(
        text(query),
        connection,
    )

print(f"Rows loaded: {len(df):,}")


# ============================================================
# REFERENCE / CURRENT PERIOD
# ============================================================

reference_df = df[
    df["tx_datetime"] < "2018-08-01"
].copy()

current_df = df[
    df["tx_datetime"] >= "2018-09-01"
].copy()


print("\nMONITORING PERIODS")

print(
    f"Reference rows: "
    f"{len(reference_df):,}"
)

print(
    f"Current rows:   "
    f"{len(current_df):,}"
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
    # CONSTANT REFERENCE FEATURES
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
    # QUANTILE-BASED BINS FROM REFERENCE DATA
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
    # INCLUDE ALL CURRENT VALUES
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
    # AVOID LOG(0)
    # --------------------------------------------------------

    epsilon = 0.000001

    reference_pct = reference_pct.clip(
        lower=epsilon
    )

    current_pct = current_pct.clip(
        lower=epsilon
    )


    # --------------------------------------------------------
    # PSI
    # --------------------------------------------------------

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
# MONITOR FEATURES
# ============================================================

results = []


for feature in FEATURE_COLUMNS:

    reference_series = (
        reference_df[feature]
    )

    current_series = (
        current_df[feature]
    )


    psi = calculate_psi(
        reference_series,
        current_series,
        bins=10,
    )


    reference_missing = (
        reference_series
        .isna()
        .mean()
        * 100
    )

    current_missing = (
        current_series
        .isna()
        .mean()
        * 100
    )


    reference_mean = (
        pd.to_numeric(
            reference_series,
            errors="coerce",
        )
        .mean()
    )

    current_mean = (
        pd.to_numeric(
            current_series,
            errors="coerce",
        )
        .mean()
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
            reference_mean,

        "current_mean":
            current_mean,

        "reference_missing_pct":
            reference_missing,

        "current_missing_pct":
            current_missing,
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

display_df = results_df.copy()

display_df[
    [
        "psi",
        "reference_mean",
        "current_mean",
        "reference_missing_pct",
        "current_missing_pct",
    ]
] = display_df[
    [
        "psi",
        "reference_mean",
        "current_mean",
        "reference_missing_pct",
        "current_missing_pct",
    ]
].round(4)


print("\n" + "=" * 110)
print("FEATURE DRIFT RESULTS")
print("=" * 110)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# RAW DRIFT SUMMARY
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
print("DRIFT SUMMARY")
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


# ============================================================
# ALL FEATURES REQUIRING REVIEW
# ============================================================

alert_features = results_df[
    results_df[
        "drift_status"
    ] != "STABLE"
]


print("\n" + "=" * 80)
print("FEATURES REQUIRING REVIEW")
print("=" * 80)

if alert_features.empty:

    print(
        "\nNo material feature drift detected."
    )

else:

    print(
        alert_features[
            [
                "feature",
                "psi",
                "drift_status",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SEPARATE EXPECTED MATURITY FROM UNEXPECTED DRIFT
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


maturity_alerts = results_df[
    (
        results_df[
            "drift_status"
        ] != "STABLE"
    )
    &
    (
        results_df[
            "feature"
        ].isin(
            EXPECTED_MATURITY_FEATURES
        )
    )
]


# ============================================================
# MONITORING INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("MONITORING INTERPRETATION")
print("=" * 80)


if not maturity_alerts.empty:

    print(
        "\nExpected feature-maturity drift detected:"
    )

    print(
        maturity_alerts[
            [
                "feature",
                "psi",
                "drift_status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nterminal_history_available changes as the "
        "transaction system matures and more terminals "
        "accumulate usable historical observations."
    )

    print(
        "This feature is therefore tracked separately "
        "from unexpected behavioral or risk-feature drift."
    )


if material_alerts.empty:

    print(
        "\nNo unexpected material drift detected "
        "in core predictive features."
    )

else:

    print(
        "\nUnexpected predictive-feature drift detected:"
    )

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


# ============================================================
# OVERALL MODEL MONITORING STATUS
# ============================================================

significant_material_alerts = (
    material_alerts[
        material_alerts[
            "drift_status"
        ] == "SIGNIFICANT DRIFT"
    ]
)


if not significant_material_alerts.empty:

    overall_status = (
        "REVIEW REQUIRED"
    )

elif not material_alerts.empty:

    overall_status = (
        "MONITOR"
    )

elif not maturity_alerts.empty:

    overall_status = (
        "STABLE — EXPECTED FEATURE MATURITY"
    )

else:

    overall_status = (
        "STABLE"
    )


print("\n" + "=" * 80)
print("OVERALL MONITORING STATUS")
print("=" * 80)

print(
    f"\nModel feature monitoring status: "
    f"{overall_status}"
)


# ============================================================
# MONITORING POLICY
# ============================================================

print("\n" + "=" * 80)
print("MONITORING POLICY")
print("=" * 80)

print(
    "\nPSI < 0.10       → STABLE"
)

print(
    "0.10 ≤ PSI < 0.25 → MODERATE DRIFT"
)

print(
    "PSI ≥ 0.25       → SIGNIFICANT DRIFT"
)

print(
    "\nExpected maturity features are reported, "
    "but do not automatically trigger model review "
    "unless accompanied by unexpected predictive-feature drift."
)


print("\n" + "=" * 80)
print("DRIFT MONITORING COMPLETE")
print("=" * 80)