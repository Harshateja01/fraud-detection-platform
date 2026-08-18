import numpy as np
import pandas as pd


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

    if reference.nunique() <= 1:

        reference_value = reference.iloc[0]

        current_same = (
            current == reference_value
        ).mean()

        if current_same == 1:
            return 0.0

        return 1.0

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

    all_bins = reference_pct.index.union(
        current_pct.index
    )

    reference_pct = reference_pct.reindex(
        all_bins,
        fill_value=0,
    )

    current_pct = current_pct.reindex(
        all_bins,
        fill_value=0,
    )

    epsilon = 0.000001

    reference_pct = reference_pct.clip(
        lower=epsilon
    )

    current_pct = current_pct.clip(
        lower=epsilon
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
# TEST: IDENTICAL DISTRIBUTION
# ============================================================

def test_identical_distribution_is_stable():

    reference = pd.Series(
        [10, 20, 30, 40, 50] * 100
    )

    current = reference.copy()

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi < 0.10

    assert (
        classify_drift(psi)
        == "STABLE"
    )


# ============================================================
# TEST: SMALL SHIFT
# ============================================================

def test_small_shift_remains_stable():

    rng = np.random.default_rng(42)

    reference = pd.Series(
        rng.normal(
            loc=50,
            scale=10,
            size=5000,
        )
    )

    current = pd.Series(
        rng.normal(
            loc=50.5,
            scale=10,
            size=5000,
        )
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi < 0.10

    assert (
        classify_drift(psi)
        == "STABLE"
    )


# ============================================================
# TEST: LARGE SHIFT
# ============================================================

def test_large_shift_detects_significant_drift():

    rng = np.random.default_rng(42)

    reference = pd.Series(
        rng.normal(
            loc=50,
            scale=10,
            size=5000,
        )
    )

    current = pd.Series(
        rng.normal(
            loc=120,
            scale=20,
            size=5000,
        )
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi >= 0.25

    assert (
        classify_drift(psi)
        == "SIGNIFICANT DRIFT"
    )


# ============================================================
# TEST: CONSTANT FEATURE
# ============================================================

def test_constant_feature_no_change():

    reference = pd.Series(
        [1] * 1000
    )

    current = pd.Series(
        [1] * 1000
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi == 0.0

    assert (
        classify_drift(psi)
        == "STABLE"
    )


# ============================================================
# TEST: CONSTANT FEATURE CHANGES
# ============================================================

def test_constant_feature_change_detected():

    reference = pd.Series(
        [0] * 1000
    )

    current = pd.Series(
        [1] * 1000
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi == 1.0

    assert (
        classify_drift(psi)
        == "SIGNIFICANT DRIFT"
    )


# ============================================================
# TEST: MISSING DATA
# ============================================================

def test_missing_values_do_not_break_psi():

    reference = pd.Series(
        [10, 20, 30, None, 40, 50] * 100
    )

    current = pd.Series(
        [10, 20, 30, None, 40, 50] * 100
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert not np.isnan(psi)

    assert psi < 0.10


# ============================================================
# TEST: EMPTY SERIES
# ============================================================

def test_empty_series_returns_nan():

    reference = pd.Series(
        dtype=float
    )

    current = pd.Series(
        [1, 2, 3]
    )

    psi = calculate_psi(
        reference,
        current,
    )

    assert np.isnan(psi)

    assert (
        classify_drift(psi)
        == "UNKNOWN"
    )