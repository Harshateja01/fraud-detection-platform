# Fraud Detection Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.ai/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)](https://github.com/features/actions)
[![Deployment](https://img.shields.io/badge/Deployment-Live-brightgreen)](#live-deployment)

An end-to-end machine learning fraud detection platform covering **data ingestion, leakage-safe feature engineering, XGBoost modeling, business-cost optimization, SHAP explainability, REST API serving, an investigator dashboard, production monitoring, automated testing, CI, Docker containerization, model governance, and cloud deployment**.

The project is designed around a practical ML engineering question:

> How do you take a fraud model beyond a notebook and turn it into an explainable, monitored, testable, governed, and deployable ML system?

---

## Live Deployment

The application is deployed as a cloud-hosted ML system.

### FastAPI Scoring API

**Production API**

https://fraud-detection-api-80gm.onrender.com

**Interactive API Documentation**

https://fraud-detection-api-80gm.onrender.com/docs

**Health Endpoint**

https://fraud-detection-api-80gm.onrender.com/health

### Streamlit Fraud Investigator Dashboard

**Live Dashboard**

> https://fraud-detection-platform-5gny42y99sedft8gyaerrr.streamlit.app/

The hosted Streamlit dashboard communicates with the deployed FastAPI service rather than a local scoring server.

> Cloud services hosted on free or low-resource infrastructure may require a short startup period after inactivity.

---

## Key Results

| Area | Result |
|---|---|
| Transactions processed | **1,754,155** |
| Customers | **4,990** |
| Terminals | **10,000** |
| Champion model | **XGBoost** |
| ROC-AUC | **0.8866** |
| PR-AUC | **0.6613** |
| Test recall | **75.50%** |
| Production threshold | **0.46** |
| Test alert rate | **3.57%** |
| Precision@1,000 | **99.20%** |
| Recall@1,000 | **38.95%** |
| Model features | **13** |
| Automated tests | **14** |
| Current model health | **MONITOR** |

---

## Platform Capabilities

- PostgreSQL-backed transaction ingestion
- Raw-data and database validation
- Leakage-safe behavioral feature engineering
- Delayed-label terminal fraud features
- Chronological train / validation / test methodology
- XGBoost champion model
- Isolation Forest anomaly benchmark
- Class-imbalance handling
- PR-AUC and ROC-AUC evaluation
- Business-cost optimized decision threshold
- Precision@K / Recall@K investigation-capacity analysis
- Global and transaction-level SHAP explanations
- FastAPI real-time scoring
- Streamlit fraud-investigation dashboard
- Feature drift monitoring with PSI
- Prediction-score drift monitoring
- Delayed-label model-performance monitoring
- Unified model-health reporting
- Drift stress testing
- Automated pytest suite
- GitHub Actions CI
- Docker / Docker Compose deployment
- Cloud-hosted API and dashboard
- Model governance documentation

---

# System Architecture

```text
                     Raw Transaction Files
                              |
                              v
                    Data Ingestion Pipeline
                              |
                              v
                         PostgreSQL
                              |
                 +------------+------------+
                 |                         |
                 v                         v
        Customer Behavioral         Delayed Terminal
             Features                Fraud Features
                 |                         |
                 +------------+------------+
                              |
                              v
                       Modeling Dataset
                              |
                              v
                      Time-Based Split
                              |
                 +------------+------------+
                 |                         |
                 v                         v
             XGBoost                Isolation Forest
             Champion               Anomaly Baseline
                 |
                 v
         Threshold Optimization
                 |
                 v
          Business Cost Analysis
                 |
                 v
         Persisted ML Artifacts
                 |
          +------+------+
          |             |
          v             v
       FastAPI       Streamlit
     Scoring API     Dashboard
          |
          v
   SHAP Explanations


                Production Monitoring
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
  Feature Drift    Prediction-Score   Delayed-Label
      (PSI)             Drift          Performance
        |                |                |
        +----------------+----------------+
                         |
                         v
                Unified Model Health
                         |
                         v
             HEALTHY / MONITOR / REVIEW


                  Engineering Quality
                         |
                         v
              Pytest -> GitHub Actions
                         |
                         v
                Docker Containers
                         |
                         v
                 Cloud Deployment
```

---

# Data Ingestion Pipeline

The repository contains a validation-oriented ingestion pipeline that verifies the state of the raw transaction data, PostgreSQL database, feature tables, and engineered dataset.

Run:

```bash
python src/data_ingestion.py
```

The validated dataset contains:

```text
Raw transaction files:              183
Customers:                        4,990
Terminals:                       10,000
Transactions:                 1,754,155
Transaction feature rows:      1,754,155
```

Pipeline quality checks verify:

- Raw transaction files are present
- Required raw columns exist
- PostgreSQL connectivity
- Required database tables
- Transaction and feature row counts
- Duplicate transaction IDs
- Duplicate feature IDs
- Missing feature rows
- Orphan feature rows
- Required feature schema
- Delayed-history feature population

Validated result:

```text
Transactions:                     1,754,155
Feature rows:                     1,754,155
Duplicate transaction IDs:                0
Duplicate feature transaction IDs:        0
Transactions missing features:            0
Orphan feature rows:                       0
Rows with delayed-history status:  1,754,155

STATUS: HEALTHY
```

This provides an explicit quality gate between data preparation and model training/scoring.

---

# Dataset

The modeling dataset contains:

```text
1,754,155 transactions
```

Fraud detection is highly imbalanced, so transactions are split **chronologically** rather than randomly.

| Dataset | Transactions |
|---|---:|
| Training | 1,169,723 |
| Validation | 296,559 |
| Test | 287,873 |

The final September test period has approximately:

```text
Fraud prevalence: 0.8848%
```

Because fewer than 1% of transactions are fraudulent, **accuracy is not treated as the primary evaluation metric**.

The project instead emphasizes:

- PR-AUC
- ROC-AUC
- Recall
- Precision
- Precision@K
- Recall@K
- Alert rate
- Business cost

---

# Leakage-Safe Feature Engineering

The champion model uses **13 production features**.

## Transaction Features

```text
tx_amount
during_weekend
during_night
```

## Customer Behavioral Features

```text
customer_tx_count_1h
customer_tx_count_6h
customer_tx_count_24h
customer_avg_amount_24h
customer_avg_amount_7d
customer_amount_deviation
```

## Terminal Features

```text
terminal_tx_count_24h
terminal_fraud_rate_7d
terminal_fraud_rate_30d
terminal_history_available
```

Identifiers and target fields are explicitly excluded:

```text
tx_fraud
tx_fraud_scenario
transaction_id
customer_id
terminal_id
```

Historical behavioral features are calculated **before the current transaction is added to history**, preventing a transaction from leaking into its own features.

---

# Delayed Fraud Labels

Historical terminal fraud features require special treatment because fraud labels are generally not available immediately.

The project assumes:

```text
Fraud label delay: 7 days
```

A terminal's historical fraud statistics therefore use only labels that would have become available by the time of the transaction being scored.

Conceptually:

```text
Transaction occurs
       |
       v
Fraud label unavailable
       |
       | 7-day delay
       v
Fraud label becomes available
       |
       v
Eligible for future terminal-risk features
```

This prevents future fraud outcomes from leaking backward into historical risk features.

---

# Leakage Audit

To evaluate whether delayed terminal-risk information provides legitimate predictive value, two model configurations were compared.

## Behavioral Features Only

```text
ROC-AUC: 0.6528
PR-AUC:  0.2580
```

## Behavioral + Delayed Terminal Risk

```text
ROC-AUC: 0.8866
PR-AUC:  0.6613
```

PR-AUC improvement:

```text
+0.4033
```

The delayed terminal-risk features provide substantial predictive information while respecting label availability.

Because the project uses synthetic transaction data, persistent fraudulent-terminal patterns may be stronger than those observed in a real payment portfolio.

---

# Champion Model

The selected supervised production candidate is:

```text
XGBoost Classifier
```

Training class distribution:

```text
Legitimate transactions: 1,160,258
Fraud transactions:          9,465

scale_pos_weight: 122.58
```

Class weighting helps the model learn from the highly imbalanced fraud class.

---

# Final Test Performance

The operating threshold was selected using the validation/business-cost analysis and frozen before final test evaluation.

```text
Production threshold: 0.46
```

September holdout performance:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8866** |
| PR-AUC | **0.6613** |
| Precision | 18.73% |
| Recall | **75.50%** |
| F1 | 0.3002 |
| Alert Rate | 3.57% |

Confusion matrix:

```text
True Positives:   1,923
False Positives:  8,343
False Negatives:    624
True Negatives: 276,983
```

At the selected operating point, the model identifies approximately **75.5% of fraudulent transactions** while sending approximately **3.57% of transactions** for review.

---

# Business Threshold Optimization

A fraud model should not automatically use a probability threshold of `0.50`.

The project evaluates thresholds using explicit false-positive and false-negative costs.

Base assumptions:

```text
False-positive cost: $5
False-negative cost: $500
```

Sensitivity analysis:

| Scenario | FP Cost | FN Cost | Threshold | Recall | Precision | Alert Rate |
|---|---:|---:|---:|---:|---:|---:|
| Low fraud cost | $5 | $100 | 0.78 | 71.23% | 47.47% | 1.35% |
| Moderate | $5 | $250 | 0.65 | 72.80% | 36.30% | 1.80% |
| **Base** | **$5** | **$500** | **0.46** | **74.97%** | **19.47%** | **3.47%** |
| High fraud cost | $5 | $1,000 | 0.39 | 76.21% | 14.21% | 4.83% |
| High review cost | $20 | $500 | 0.78 | 71.23% | 47.47% | 1.35% |

Selected production threshold:

```text
0.46
```

Estimated September business cost under the base assumptions:

```text
$353,715
```

The analysis demonstrates that the appropriate operating point depends on the relative cost of missed fraud and unnecessary investigations.

---

# Investigation Capacity - Precision@K

Fraud operations teams usually have limited investigation capacity.

The project therefore evaluates ranking performance at fixed review volumes.

| Investigation Capacity | Fraud Found | Precision@K | Recall@K | Lift vs Random |
|---:|---:|---:|---:|---:|
| 500 | 499 | 99.80% | 19.59% | 112.8x |
| 1,000 | 992 | 99.20% | 38.95% | 112.1x |
| 2,500 | 1,665 | 66.60% | 65.37% | 75.3x |
| 5,000 | 1,840 | 36.80% | 72.24% | 41.6x |
| 10,000 | 1,921 | 19.21% | 75.42% | 21.7x |

Among the **1,000 highest-risk transactions**:

```text
Fraud transactions found: 992
Precision@1000:          99.20%
Recall@1000:             38.95%
```

This demonstrates the model's usefulness when investigation capacity is constrained.

---

# Anomaly Detection Benchmark

An **Isolation Forest** provides an unsupervised comparison.

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

The supervised XGBoost model substantially outperforms the anomaly benchmark when historical fraud labels are available.

---

# Model Explainability

The platform uses **SHAP** for global and transaction-level model explanations.

Global SHAP analysis was performed on 10,000 test transactions.

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | customer_amount_deviation | 0.464557 |
| 2 | terminal_fraud_rate_30d | 0.431246 |
| 3 | tx_amount | 0.383059 |
| 4 | customer_avg_amount_7d | 0.371537 |
| 5 | terminal_fraud_rate_7d | 0.348273 |

At scoring time, explanations distinguish between factors that:

```text
Increase fraud risk
Reduce fraud risk
```

This gives an investigator context around a prediction rather than exposing only a probability.

---

# Fraud Investigator Dashboard

The Streamlit application provides an interactive interface for transaction-level fraud investigation.

![Fraud Detection Dashboard](docs/images/dashboard-overview.png)

The dashboard provides:

- Transaction input controls
- API health status
- Fraud probability
- Risk classification
- Production threshold
- Alert decision
- Recommended action
- Transaction context
- SHAP explanations
- Model information

## Explainability View

![Fraud Model Explainability](docs/images/dashboard-explainability.png)

The application communicates with the FastAPI scoring service using the configurable:

```text
API_URL
```

This allows the same dashboard code to work with local Docker infrastructure and the deployed cloud API.

---

# FastAPI Scoring Service

The persisted champion model is exposed through a FastAPI REST API.

![FastAPI Swagger Documentation](docs/images/api-swagger.png)

Main endpoints:

```text
GET  /
GET  /health
POST /score-transaction
```

Local API startup:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Local Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Production Swagger documentation:

```text
https://fraud-detection-api-80gm.onrender.com/docs
```

Example scoring response:

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

The scoring response also includes SHAP-based factors that increase or reduce predicted fraud risk.

---

# API Health Monitoring

The `/health` endpoint verifies that the production scoring dependencies are available.

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "imputer_loaded": true,
  "shap_explainer_loaded": true,
  "decision_threshold": 0.46,
  "feature_count": 13
}
```

This provides a lightweight operational readiness check for the scoring service.

---

# Production Monitoring

The project monitors the model at three separate levels.

```text
Incoming Features
      |
      v
Feature Drift
      |
      v
Prediction Scores
      |
      v
Score Drift
      |
      v
Delayed Fraud Labels
      |
      v
Performance Monitoring
      |
      v
Unified Model Health
```

Separating these signals helps distinguish input-population changes from prediction-behavior changes and actual predictive degradation.

---

## 1. Feature Drift Monitoring

Population Stability Index (**PSI**) is calculated across all 13 production features.

Monitoring policy:

```text
PSI < 0.10           -> STABLE
0.10 <= PSI < 0.25   -> MODERATE DRIFT
PSI >= 0.25          -> SIGNIFICANT DRIFT
```

September results:

```text
Stable features:             12
Moderate drift features:      0
Significant drift features:   1
```

The significant change was:

```text
terminal_history_available
PSI: 0.7353
```

This feature naturally changes as terminals accumulate sufficient historical observations.

The monitoring framework therefore separates expected **feature-maturity drift** from unexpected predictive-feature drift.

```text
Feature Drift Status:
STABLE - EXPECTED FEATURE MATURITY
```

---

## 2. Prediction-Score Monitoring

The distribution of model fraud probabilities is monitored independently from feature drift.

| Metric | Reference | Current |
|---|---:|---:|
| Mean score | 0.1836 | 0.1898 |
| Median score | 0.1595 | 0.1629 |
| P95 | 0.3717 | 0.3840 |
| P99 | 0.8734 | 0.9163 |
| Alert Rate | 3.31% | 3.56% |

Prediction-score PSI:

```text
0.0129
```

Status:

```text
STABLE
```

Alert-rate change:

```text
+0.25 percentage points
```

Fraud transactions continued to receive substantially higher scores:

```text
Average legitimate score: 0.1846
Average fraud score:      0.7759
Median fraud score:       0.9909
```

---

## 3. Delayed-Label Performance Monitoring

Once fraud labels mature, the monitoring layer evaluates actual predictive performance.

September performance:

```text
PR-AUC:    0.6620
Precision: 0.1878
Recall:    0.7554
```

Compared with the pooled reference period:

```text
Reference PR-AUC:    0.6960
Current PR-AUC:      0.6620
Change:             -0.0340

Reference Precision: 0.1982
Current Precision:   0.1878
Change:             -0.0104

Reference Recall:    0.8098
Current Recall:      0.7554
Change:             -0.0544
```

The recall decrease crosses the project's moderate degradation threshold.

Therefore:

```text
Model Performance Status: MONITOR
```

This does not automatically trigger retraining. It indicates that performance should be monitored across subsequent mature-label periods.

---

# Unified Model Health

The monitoring layers are consolidated into one operational model-health report.

Current state:

```text
Feature Drift:          STABLE - EXPECTED FEATURE MATURITY
Prediction Score Drift: STABLE
Model Performance:      MONITOR

----------------------------------------
Overall Model Health:   MONITOR
----------------------------------------
```

Active alert:

```text
MODERATE RECALL DEGRADATION
```

The governance policy uses tiered escalation:

```text
HEALTHY
   |
   v
MONITOR
   |
   v
REVIEW REQUIRED
```

A single moderate signal does not automatically trigger retraining. Significant or persistent degradation would require model review.

Run:

```bash
python src/model_health_report.py
```

---

# Drift Stress Testing

The drift detector was tested against an artificially shifted production population.

Synthetic drift was introduced into:

- Transaction amounts
- Customer transaction velocity
- Customer spending behavior
- Customer amount deviation
- Terminal transaction velocity
- Historical terminal risk

Stress-test result:

```text
Stable features:             3
Moderate drift features:     0
Significant drift features: 10
```

The monitoring system escalated to:

```text
REVIEW REQUIRED
```

This validates that the monitoring policy can distinguish expected population evolution from substantial unexpected drift.

---

# Model Governance

Formal model-governance documentation is maintained at:

```text
docs/MODEL_GOVERNANCE.md
```

The governance layer documents the operating principles around:

- Model purpose and scope
- Champion model selection
- Feature and leakage controls
- Frozen decision threshold
- Model evaluation
- Explainability
- Monitoring
- Health-state interpretation
- Retraining/review principles
- Known limitations
- Deployment considerations

The project deliberately separates:

```text
Monitoring signal
       |
       v
Operational assessment
       |
       v
Human/model review
       |
       v
Retraining decision
```

Retraining is therefore treated as a governed decision rather than an automatic response to a single monitoring alert.

---

# Automated Testing

The repository contains **14 automated tests**.

Run:

```bash
python -m pytest -v
```

Coverage includes:

- API root endpoint
- Health endpoint
- Fraud scoring
- Alert generation
- SHAP explanations
- Request validation
- Invalid fraud-rate validation
- Stable PSI distributions
- Small distribution shifts
- Significant distribution shifts
- Constant features
- Changed constant features
- Missing values
- Empty input handling

Validated result:

```text
14 passed
```

---

# Continuous Integration

GitHub Actions executes the test suite for pushes and pull requests to `main`.

![GitHub Actions CI](docs/images/github-actions-ci.png)

Workflow:

```text
.github/workflows/tests.yml
```

Pipeline:

```text
Checkout Repository
        |
        v
Set Up Python 3.12
        |
        v
Install Dependencies
        |
        v
Run Pytest
        |
        v
14 Automated Tests
```

The test suite has been validated on both the Windows development environment and GitHub's Linux runner.

---

# Docker

The API and Streamlit dashboard are containerized and can be launched together using Docker Compose.

```text
Dockerfile
docker-compose.yml
```

Container architecture:

```text
+----------------------+
| Streamlit Dashboard  |
|        :8501         |
+----------+-----------+
           |
           | HTTP
           v
+----------------------+
| FastAPI Scoring API  |
|        :8000         |
+----------+-----------+
           |
           v
+----------------------+
| Persisted XGBoost    |
| Model + Imputer      |
| Metadata + SHAP      |
+----------------------+
```

Build and start:

```bash
docker compose up --build
```

Verify:

```bash
docker compose ps
```

Expected services:

```text
fraud-api
fraud-dashboard
```

Local endpoints:

```text
API:       http://localhost:8000
Health:    http://localhost:8000/health
Swagger:   http://localhost:8000/docs
Dashboard: http://localhost:8501
```

Stop:

```bash
docker compose down
```

The Docker deployment has been validated using the WSL2-backed Docker Desktop Linux engine.

---

# Cloud Deployment

The application has also been deployed beyond the local Docker environment.

```text
GitHub Repository
       |
       +-----------------------+
       |                       |
       v                       v
 FastAPI Service        Streamlit Dashboard
       |                       |
       v                       |
    Render                     |
       |                       |
       +-----------<-----------+
                   |
                   v
             HTTPS Scoring
```

The FastAPI service runs as a containerized web service and exposes the trained model through HTTPS.

Production API:

```text
https://fraud-detection-api-80gm.onrender.com
```

The Streamlit deployment uses the production API through:

```text
API_URL
```

rather than the local Docker hostname.

This allows the project to support three execution modes:

```text
Local Python
     |
     v
Local Docker Compose
     |
     v
Cloud Deployment
```

---

# Model Artifacts

Production model artifacts are persisted under:

```text
models/
```

Artifacts include:

```text
fraud_xgboost_model.joblib
fraud_imputer.joblib
fraud_model_metadata.joblib
```

The metadata stores the production feature configuration and frozen decision threshold so the serving layer can load the champion model without retraining at startup.

---

# Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Harshateja01/fraud-detection-platform.git
cd fraud-detection-platform
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the API

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

## 5. Start the Dashboard

In another terminal:

```bash
python -m streamlit run src/app.py
```

Open:

```text
http://localhost:8501
```

Alternatively, run the complete serving stack with Docker:

```bash
docker compose up --build
```

---

# Project Structure

```text
fraud-detection-platform/
|
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|
|-- data/
|   |-- processed/
|   `-- raw/
|
|-- docs/
|   |-- MODEL_GOVERNANCE.md
|   `-- images/
|       |-- api-swagger.png
|       |-- dashboard-explainability.png
|       |-- dashboard-overview.png
|       `-- github-actions-ci.png
|
|-- models/
|   |-- fraud_imputer.joblib
|   |-- fraud_model_metadata.joblib
|   `-- fraud_xgboost_model.joblib
|
|-- src/
|   |-- api.py
|   |-- app.py
|   |-- audit_feature_leakage.py
|   |-- build_behavioral_features.py
|   |-- build_delayed_fraud_features.py
|   |-- create_features.py
|   |-- data_ingestion.py
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
|   |-- model_health_report.py
|   |-- monitor_drift.py
|   |-- monitor_model_performance.py
|   |-- monitor_prediction_scores.py
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

| Area | Technologies |
|---|---|
| Language | Python 3.12 |
| Data Processing | pandas, NumPy |
| Database | PostgreSQL, SQLAlchemy |
| Machine Learning | scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Monitoring | PSI, scikit-learn metrics |
| Testing | pytest |
| CI | GitHub Actions |
| Containers | Docker, Docker Compose |
| Cloud Deployment | Render, Streamlit Community Cloud |
| Persistence | joblib |
| Governance | Model health policy and governance documentation |

---

# Key Engineering Decisions

## Time-Based Validation

Transactions are split chronologically instead of randomly so evaluation more closely represents deployment against future transactions.

## Leakage-Aware Feature Engineering

Historical features use only information that would have been available at transaction time.

## Delayed Fraud Labels

Terminal fraud rates use fraud labels only after the assumed seven-day label delay.

## Imbalance-Aware Evaluation

PR-AUC, recall, Precision@K, Recall@K, alert rate, and business cost are emphasized because fraud prevalence is below 1%.

## Business-Driven Thresholding

The production threshold is selected from explicit false-positive and false-negative costs rather than automatically using `0.50`.

## Capacity-Aware Evaluation

Precision@K and Recall@K measure model usefulness when investigators can review only a limited number of transactions.

## Explainable Predictions

SHAP provides both global model interpretation and transaction-level explanations.

## Multi-Layer Monitoring

Feature distributions, prediction scores, and delayed-label model performance are monitored independently before being consolidated into overall model health.

## Tiered Model Governance

The monitoring framework distinguishes:

```text
HEALTHY
MONITOR
REVIEW REQUIRED
```

Moderate degradation does not automatically cause retraining.

## Automated Quality Control

API and drift-monitoring tests execute automatically through GitHub Actions.

## Portable Serving

The same application can run through local Python processes, Docker Compose, or cloud-hosted services.

---

# Limitations

This project uses a **synthetic fraud dataset**.

Synthetic data can contain stronger or cleaner fraud patterns than real payment environments. Persistent terminal-risk patterns, in particular, may be more predictive than they would be in a live financial portfolio.

The false-positive and false-negative costs used for threshold optimization are illustrative assumptions rather than values obtained from a real fraud-operations organization.

The application demonstrates production-oriented ML engineering patterns but is **not a production banking system**.

A real financial deployment would require additional controls such as:

- Authentication and authorization
- Enterprise secrets management
- Encryption and key management
- Formal audit controls
- Data-quality SLAs
- Centralized observability
- Infrastructure hardening
- Rate limiting
- Persistent monitoring storage
- Incident management
- Formal independent model validation
- Regulatory and compliance controls

---

# Future Improvements

Potential extensions include:

- Automated retraining workflows
- Persistent monitoring history
- Model registry and versioning
- Experiment tracking
- API authentication and authorization
- Batch scoring
- Database-backed investigation queues
- Investigator feedback capture
- Model calibration monitoring
- Segment-level performance analysis
- Data-quality and feature-freshness monitoring
- API latency and error monitoring
- Scheduled monitoring reports
- Production alerting
- Centralized logging and observability
- Infrastructure-as-code deployment

---

# What This Project Demonstrates

This repository goes beyond training a fraud classifier.

It demonstrates an end-to-end ML system spanning:

```text
Raw Data
   |
   v
Data Ingestion & Validation
   |
   v
PostgreSQL
   |
   v
Leakage-Safe Feature Engineering
   |
   v
Supervised + Unsupervised Modeling
   |
   v
Business Threshold Optimization
   |
   v
Capacity-Aware Evaluation
   |
   v
SHAP Explainability
   |
   v
Model Persistence
   |
   v
REST API Serving
   |
   v
Investigator Dashboard
   |
   v
Feature + Score + Performance Monitoring
   |
   v
Unified Model Health
   |
   v
Model Governance
   |
   v
Automated Testing
   |
   v
Continuous Integration
   |
   v
Docker
   |
   v
Cloud Deployment
```

Final champion model:

```text
Model:      XGBoost
ROC-AUC:    0.8866
PR-AUC:     0.6613
Recall:     75.50%
Threshold:  0.46
```

Current unified monitoring state:

```text
Overall Model Health: MONITOR

Reason:
Moderate recall degradation while feature
and prediction-score distributions remain stable.
```

The goal is not simply to produce a high-performing fraud classifier, but to demonstrate how a machine-learning model can be **built, evaluated, explained, served, tested, monitored, governed, containerized, and deployed as an end-to-end ML system**.