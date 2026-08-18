from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

models_dir = project_root / "models"

model_path = models_dir / "fraud_xgboost_model.joblib"
imputer_path = models_dir / "fraud_imputer.joblib"
metadata_path = models_dir / "fraud_model_metadata.joblib"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
metadata = joblib.load(metadata_path)


print("=" * 70)
print("SAVED FRAUD MODEL TEST")
print("=" * 70)

print("\nArtifacts loaded successfully.")

print(
    f"\nModel name: "
    f"{metadata['model_name']}"
)

print(
    f"Model type: "
    f"{metadata['model_type']}"
)

print(
    f"Decision threshold: "
    f"{metadata['decision_threshold']:.2f}"
)

print(
    f"Number of features: "
    f"{len(metadata['feature_columns'])}"
)


# ============================================================
# SAMPLE TRANSACTION
# ============================================================

sample = pd.DataFrame(
    [
        {
            "tx_amount": 164.75,
            "during_weekend": 1,
            "during_night": 1,

            "customer_tx_count_1h": 0,
            "customer_tx_count_6h": 0,
            "customer_tx_count_24h": 1,

            "customer_avg_amount_24h": 35.49,
            "customer_avg_amount_7d": 60.72,
            "customer_amount_deviation": 104.0259,

            "terminal_tx_count_24h": 2,

            "terminal_fraud_rate_7d": 0.0,
            "terminal_fraud_rate_30d": 0.0,
            "terminal_history_available": 1,
        }
    ],
    columns=metadata["feature_columns"],
)


# ============================================================
# SCORE
# ============================================================

sample_imputed = imputer.transform(
    sample
)

probability = model.predict_proba(
    sample_imputed
)[0, 1]

alert = (
    probability
    >= metadata["decision_threshold"]
)


print("\n" + "=" * 70)
print("SAMPLE SCORE")
print("=" * 70)

print(
    f"\nFraud probability: "
    f"{probability:.2%}"
)

print(
    f"Alert generated: "
    f"{'YES' if alert else 'NO'}"
)


print("\n" + "=" * 70)
print("SAVED MODEL TEST COMPLETE")
print("=" * 70)