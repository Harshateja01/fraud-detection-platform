from fastapi.testclient import TestClient

from src.api import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# SAMPLE TRANSACTION
# ============================================================

VALID_TRANSACTION = {
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


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"

    assert (
        data["service"]
        == "Fraud Detection Scoring API"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True

    assert data["imputer_loaded"] is True

    assert data["shap_explainer_loaded"] is True

    assert data["feature_count"] == 13

    assert data["decision_threshold"] == 0.46


# ============================================================
# VALID FRAUD SCORE
# ============================================================

def test_score_transaction():

    response = client.post(
        "/score-transaction",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data

    assert "fraud_probability_pct" in data

    assert "decision_threshold" in data

    assert "alert_generated" in data

    assert "risk_level" in data

    assert "recommended_action" in data

    assert "model_name" in data

    assert "factors_increasing_risk" in data

    assert "factors_reducing_risk" in data


    assert (
        0
        <= data["fraud_probability"]
        <= 1
    )

    assert (
        0
        <= data["fraud_probability_pct"]
        <= 100
    )

    assert data["decision_threshold"] == 0.46

    assert isinstance(
        data["alert_generated"],
        bool,
    )

    assert data["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]


# ============================================================
# EXPECTED SAMPLE BEHAVIOR
# ============================================================

def test_sample_generates_alert():

    response = client.post(
        "/score-transaction",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["fraud_probability"] > 0.46

    assert data["alert_generated"] is True

    assert data["risk_level"] in [
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]


# ============================================================
# SHAP RESPONSE
# ============================================================

def test_shap_explanations_returned():

    response = client.post(
        "/score-transaction",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    increasing = data[
        "factors_increasing_risk"
    ]

    reducing = data[
        "factors_reducing_risk"
    ]

    assert isinstance(
        increasing,
        list,
    )

    assert isinstance(
        reducing,
        list,
    )

    assert len(increasing) > 0

    assert len(reducing) > 0


    first_factor = increasing[0]

    assert "feature" in first_factor

    assert "value" in first_factor

    assert "shap_value" in first_factor


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_negative_transaction_amount_rejected():

    invalid_transaction = (
        VALID_TRANSACTION.copy()
    )

    invalid_transaction[
        "tx_amount"
    ] = -100

    response = client.post(
        "/score-transaction",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_invalid_fraud_rate_rejected():

    invalid_transaction = (
        VALID_TRANSACTION.copy()
    )

    invalid_transaction[
        "terminal_fraud_rate_7d"
    ] = 1.5

    response = client.post(
        "/score-transaction",
        json=invalid_transaction,
    )

    assert response.status_code == 422