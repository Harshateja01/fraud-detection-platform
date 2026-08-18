# Fraud Detection Platform

An end-to-end machine learning fraud detection platform built to demonstrate a production-oriented fraud modeling workflow: feature engineering, leakage prevention, model training, business-cost optimization, explainability, API serving, monitoring, automated testing, and CI/CD.

The platform uses an XGBoost classifier as the champion fraud model and exposes predictions through a FastAPI service and an investigator-facing Streamlit dashboard.

---

## Project Overview

Fraud detection is an extremely imbalanced classification problem where model quality cannot be evaluated using accuracy alone.

This project focuses on:

- Time-aware model validation
- Leakage-safe feature engineering
- Highly imbalanced classification
- XGBoost fraud scoring
- Business-cost-based threshold selection
- Precision@K and Recall@K analysis
- SHAP model explainability
- Isolation Forest anomaly detection
- Model persistence
- FastAPI model serving
- Streamlit investigator dashboard
- Feature drift monitoring
- Automated API and monitoring tests
- GitHub Actions CI
- Docker-based deployment configuration

The objective is not simply to predict fraud, but to build a fraud detection workflow that reflects how an ML system could be evaluated and operated in practice.

---

# Architecture

```text
Transaction Data
       |
       v
PostgreSQL
       |
       v
Feature Engineering
       |
       +-------------------------------+
       |                               |
       v                               v
Behavioral Features          Delayed Terminal Risk
       |                               |
       +---------------+---------------+
                       |
                       v
               Modeling Dataset
                       |
                       v
                Time-Based Split
                       |
             +---------+---------+
             |                   |
             v                   v
         XGBoost          Isolation Forest
         Champion         Anomaly Baseline
             |
             v
      Threshold Optimization
             |
             v
      Business Cost Analysis
             |
             v
       Persisted Model
             |
       +-----+------+
       |            |
       v            v
    FastAPI     Streamlit
   Scoring API   Dashboard
       |
       v
 SHAP Explanations

Production Monitoring
       |
       v
 PSI Feature Drift
       |
       v
 Stable / Monitor / Review Required

Engineering Quality
       |
       v
 Pytest -> GitHub Actions CI
```

---

# Dataset

The modeling dataset contains:

```text
1,754,155 transactions
```

A chronological split is used instead of a random train/test split.

| Dataset | Transactions |
|---|---:|
| Training | 1,169,723 |
| Validation | 296,559 |
| Test | 287,873 |

The final test period represents September transactions.

The test fraud prevalence is approximately:

```text
0.8848%
```

This extreme imbalance makes metrics such as PR-AUC, recall, precision, and Precision@K more informative than raw accuracy.

---

# Leakage-Safe Feature Engineering

The champion model uses 13 features.

### Transaction features

```text
tx_amount
during_weekend
during_night
```

### Customer behavioral features

```text
customer_tx_count_1h
customer_tx_count_6h
customer_tx_count_24h
customer_avg_amount_24h
customer_avg_amount_7d
customer_amount_deviation
```

### Terminal features

```text
terminal_tx_count_24h
terminal_fraud_rate_7d
terminal_fraud_rate_30d
terminal_history_available
```

Special attention was given to preventing target leakage.

The following fields are NOT model features:

```text
tx_fraud
tx_fraud_scenario
transaction_id
customer_id
terminal_id
```

Historical terminal fraud rates use a **7-day label-availability delay**.

This prevents the current transaction's fraud outcome or labels that would not yet have been available operationally from leaking into the prediction.

---

# Leakage Audit

Two models were compared.

### Model A — Behavioral Features Only

```text
ROC-AUC: 0.6528
PR-AUC:  0.2580
```

### Model B — Behavioral + Delayed Terminal Risk

```text
ROC-AUC: 0.8866
PR-AUC:  0.6613
```

PR-AUC improvement:

```text
+0.4033
```

The terminal-risk features therefore provide substantial predictive value.

Because this is a synthetic dataset, persistent fraudulent-terminal patterns may be stronger than would normally be observed in a real production portfolio.

---

# Champion Model

The final champion model is:

```text
XGBoost
```

Training class balance:

```text
Legitimate transactions: 1,160,258
Fraud transactions:          9,465

scale_pos_weight: 122.58
```

Class weighting is used to account for the severe class imbalance.

---

# Final Test Performance

The frozen operating threshold is:

```text
0.46
```

September test performance:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8866 |
| PR-AUC | 0.6613 |
| Precision | 0.1873 |
| Recall | 0.7550 |
| F1 | 0.3002 |
| Alert Rate | 3.57% |

Confusion matrix:

```text
True Positives:   1,923
False Positives:  8,343
False Negatives:    624
True Negatives: 276,983
```

The model identifies approximately **75.5% of fraud transactions** while sending approximately **3.57% of transactions** to the alert queue.

---

# Business Threshold Optimization

A fraud model should not necessarily operate at the threshold that maximizes F1.

The project therefore evaluates thresholds using business costs.

Base assumptions:

```text
False-positive cost: $5
False-negative cost: $500
```

Cost sensitivity analysis produced the following operating points:

| Scenario | FP Cost | FN Cost | Optimal Threshold | Recall | Precision | Alert Rate |
|---|---:|---:|---:|---:|---:|---:|
| Low fraud cost | $5 | $100 | 0.78 | 71.23% | 47.47% | 1.35% |
| Moderate | $5 | $250 | 0.65 | 72.80% | 36.30% | 1.80% |
| Base | $5 | $500 | 0.46 | 74.97% | 19.47% | 3.47% |
| High fraud cost | $5 | $1,000 | 0.39 | 76.21% | 14.21% | 4.83% |
| High review cost | $20 | $500 | 0.78 | 71.23% | 47.47% | 1.35% |

The final operating threshold was frozen at:

```text
0.46
```

On the September test set, the estimated business cost under the base assumptions was:

```text
$353,715
```

---

# Investigation Capacity — Precision@K

Fraud teams often operate with a fixed investigation capacity.

For this reason, the model was also evaluated using Precision@K and Recall@K.

| Investigation Capacity | Fraud Found | Precision@K | Recall@K | Lift vs Random |
|---:|---:|---:|---:|---:|
| 500 | 499 | 99.80% | 19.59% | 112.8x |
| 1,000 | 992 | 99.20% | 38.95% | 112.1x |
| 2,500 | 1,665 | 66.60% | 65.37% | 75.3x |
| 5,000 | 1,840 | 36.80% | 72.24% | 41.6x |
| 10,000 | 1,921 | 19.21% | 75.42% | 21.7x |

The top 1,000 highest-risk transactions contain:

```text
992 fraud transactions
Precision@1000: 99.20%
Recall@1000:    38.95%
```

This demonstrates strong ranking performance for capacity-constrained investigation teams.

---

# Anomaly Detection Benchmark

An Isolation Forest model was trained without fraud labels to provide an unsupervised benchmark.

Results:

```text
ROC-AUC: 0.8567
PR-AUC:  0.2610
```

At the top 1,000 transactions:

```text
Fraud found:     438
Precision@1000: 43.80%
Recall@1000:    17.20%
Lift:            49.5x
```

The supervised XGBoost model substantially outperforms the anomaly detector when reliable historical fraud labels are available.

---

# Model Explainability

SHAP is used to explain model predictions.

Global SHAP analysis was performed on 10,000 test transactions.

Top global model drivers:

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | customer_amount_deviation | 0.464557 |
| 2 | terminal_fraud_rate_30d | 0.431246 |
| 3 | tx_amount | 0.383059 |
| 4 | customer_avg_amount_7d | 0.371537 |
| 5 | terminal_fraud_rate_7d | 0.348273 |

The platform also generates transaction-level explanations showing factors that increase and reduce predicted fraud risk.

Example:

```text
Fraud probability: 68.84%
Risk level: MEDIUM
Alert threshold: 46%
Alert generated: YES
```

Example risk-increasing factors include:

```text
customer_amount_deviation
tx_amount
terminal_history_available
during_night
```

---

# Model Persistence

The trained production artifacts are stored under:

```text
models/
```

Artifacts:

```text
fraud_xgboost_model.joblib
fraud_imputer.joblib
fraud_model_metadata.joblib
```

The metadata contains the model configuration, feature list, and frozen decision threshold.

This allows the API to load the already-trained champion model rather than retraining at startup.

---

# FastAPI Scoring Service

The model is exposed through a FastAPI REST service.

Run locally:

```bash
uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The API returns:

```json
{
  "fraud_probability": 0.560381,
  "fraud_probability_pct": 56.04,
  "decision_threshold": 0.46,
  "alert_generated": true,
  "risk_level": "MEDIUM",
  "recommended_action": "QUEUE FOR REVIEW",
  "model_name": "XGBoost Fraud Detection Champion"
}
```

The response also includes SHAP-based factors increasing and reducing the transaction's predicted fraud risk.

---

# Streamlit Investigation Dashboard

An investigator-facing Streamlit application is included.

Run:

```bash
streamlit run src/app.py
```

The dashboard communicates with the FastAPI scoring service and provides an interactive interface for transaction risk assessment and model explanations.

---

# Feature Drift Monitoring

Production ML systems must monitor whether incoming data continues to resemble the data used to develop the model.

This project implements **Population Stability Index (PSI)** monitoring across all 13 model features.

Project monitoring thresholds:

```text
PSI < 0.10          STABLE
0.10 <= PSI < 0.25  MODERATE DRIFT
PSI >= 0.25         SIGNIFICANT DRIFT
```

September monitoring found:

```text
Stable features:             12
Moderate drift features:      0
Significant drift features:   1
```

The significant feature was:

```text
terminal_history_available
PSI: 0.7353
```

This feature naturally changes as terminals accumulate historical observations.

The monitoring system therefore separates expected **feature-maturity drift** from unexpected drift in core predictive variables.

Final monitoring status:

```text
STABLE — EXPECTED FEATURE MATURITY
```

No unexpected material drift was detected in the core predictive features.

---

# Drift Stress Testing

The monitoring system was also tested against an artificially shifted production batch.

The stress scenario changed:

- Transaction amounts
- Customer transaction velocity
- Customer spending baselines
- Customer amount deviations
- Terminal transaction velocity
- Historical terminal risk

Results:

```text
Stable features:             3
Moderate drift features:     0
Significant drift features: 10
```

The monitoring system correctly escalated to:

```text
REVIEW REQUIRED
```

This demonstrates that the monitoring policy can distinguish normal production behavior from substantial unexpected feature drift.

---

# Automated Testing

The project uses pytest for automated testing.

Run:

```bash
python -m pytest -v
```

Current suite:

```text
14 tests
```

The tests cover:

- API root endpoint
- Health endpoint
- Fraud scoring
- Alert generation
- SHAP explanations
- Request validation
- Invalid fraud-rate validation
- Stable PSI distributions
- Small distribution shifts
- Significant drift detection
- Constant features
- Changed constant features
- Missing values
- Empty input handling

---

# Continuous Integration

GitHub Actions automatically runs the full test suite for pushes and pull requests to `main`.

Workflow:

```text
.github/workflows/tests.yml
```

CI pipeline:

```text
Checkout repository
        |
        v
Set up Python 3.12
        |
        v
Install dependencies
        |
        v
Run pytest
        |
        v
14 automated tests
```

The test suite runs successfully on both Windows development environments and GitHub's Linux runner.

---

# Docker

Docker configuration is included for the API and Streamlit dashboard.

Files:

```text
Dockerfile
docker-compose.yml
```

The intended architecture is:

```text
+---------------------+
| Streamlit Dashboard |
|      :8501          |
+----------+----------+
           |
           | HTTP
           v
+---------------------+
|    FastAPI API      |
|      :8000          |
+----------+----------+
           |
           v
+---------------------+
| Persisted XGBoost   |
| Model + SHAP        |
+---------------------+
```

Run with:

```bash
docker compose up --build
```

Docker execution requires host virtualization support and a functioning Docker engine.

---

# Project Structure

```text
fraud-detection-platform/
|
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|
|-- models/
|   |-- fraud_xgboost_model.joblib
|   |-- fraud_imputer.joblib
|   `-- fraud_model_metadata.joblib
|
|-- src/
|   |-- api.py
|   |-- app.py
|   |-- audit_feature_leakage.py
|   |-- build_behavioral_features.py
|   |-- build_delayed_fraud_features.py
|   |-- create_features.py
|   |-- evaluate_business_cost.py
|   |-- evaluate_cost_sensitivity.py
|   |-- evaluate_thresholds.py
|   |-- evaluate_top_k.py
|   |-- explain_model.py
|   |-- explain_transaction.py
|   |-- explore_data.py
|   |-- final_model_report.py
|   |-- inspect_transactions.py
|   |-- load_transactions.py
|   |-- monitor_drift.py
|   |-- save_champion_model.py
|   |-- test_drift_detection.py
|   |-- test_saved_model.py
|   |-- train_anomaly_model.py
|   |-- train_baseline.py
|   `-- train_challenger.py
|
|-- tests/
|   |-- __init__.py
|   |-- test_api.py
|   `-- test_drift.py
|
|-- .gitignore
|-- Dockerfile
|-- docker-compose.yml
|-- pytest.ini
|-- requirements.txt
`-- README.md
```

---

# Technology Stack

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP

### Data

- PostgreSQL
- SQLAlchemy

### Model Serving

- FastAPI
- Uvicorn
- Pydantic

### Application

- Streamlit

### Testing & CI

- pytest
- GitHub Actions

### Deployment

- Docker
- Docker Compose

---

# Key Engineering Decisions

### Time-based validation

Transactions are split chronologically rather than randomly to better approximate deployment against future transactions.

### Leakage-aware historical risk

Terminal fraud rates use delayed labels instead of information that would not have been available when a transaction occurred.

### Business-driven threshold

The operating threshold is selected using false-positive and false-negative costs rather than automatically using `0.50`.

### Capacity-aware evaluation

Precision@K and Recall@K quantify performance when investigators can only review a limited number of alerts.

### Explainable predictions

SHAP provides both global feature importance and transaction-level explanations.

### Production monitoring

PSI-based feature monitoring detects distribution changes and distinguishes expected feature maturity from unexpected predictive drift.

### Automated quality control

API and drift tests run automatically through GitHub Actions.

---

# Limitations

This project uses a synthetic fraud dataset.

Synthetic data can contain stronger or cleaner fraud patterns than real-world payment environments. In particular, persistent terminal-risk patterns may be more predictive than they would be in a live portfolio.

The business costs used for threshold optimization are illustrative assumptions rather than costs derived from an actual fraud operations team.

The current monitoring layer focuses on feature drift. A production system should additionally monitor:

- Prediction-score drift
- Fraud-rate drift after labels mature
- Precision and recall over time
- Alert volumes
- API latency and errors
- Data-quality failures
- Feature freshness
- Model calibration
- Segment-level performance

---

# Future Improvements

Potential extensions include:

- Prediction-score monitoring
- Model performance monitoring after delayed labels arrive
- Automated retraining triggers
- Model registry and versioning
- Experiment tracking
- API authentication
- Batch scoring
- Database-backed investigation queues
- Investigator feedback capture
- Cloud deployment
- Observability and alerting
- Scheduled drift reports

---

# Summary

This repository demonstrates an end-to-end fraud detection system extending beyond model training.

It includes:

```text
Data engineering
Feature engineering
Leakage auditing
Supervised fraud modeling
Anomaly detection
Business threshold optimization
Capacity-based evaluation
SHAP explainability
Model persistence
REST API serving
Investigator dashboard
Feature drift monitoring
Drift stress testing
Automated testing
GitHub Actions CI
Docker deployment configuration
```

The final XGBoost model achieves:

```text
ROC-AUC: 0.8866
PR-AUC:  0.6613
Recall:  75.50%
```

at a business-selected operating threshold of:

```text
0.46
```

while the monitoring and CI layers provide the foundations for operating the model as an ML system rather than treating it as a standalone notebook experiment.