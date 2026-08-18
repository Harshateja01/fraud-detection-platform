# Model Governance — Fraud Detection Platform

## 1. Purpose

This document defines the governance framework for the Fraud Detection Platform, including model ownership, intended use, model validation, explainability, monitoring, risk controls, and retraining considerations.

The platform is designed to identify potentially fraudulent payment transactions using machine-learning-based risk scoring while supporting investigation through interpretable model explanations.

The system should be treated as a decision-support system rather than an autonomous financial decision maker.

---

## 2. Model Overview

| Component | Description |
|---|---|
| Model | XGBoost Fraud Detection Model |
| Problem Type | Binary classification |
| Positive Class | Fraudulent transaction |
| Scoring Output | Fraud probability |
| Decision Threshold | 0.46 |
| Primary Evaluation Metric | PR-AUC |
| Explainability | SHAP |
| API Layer | FastAPI |
| Investigation UI | Streamlit |
| Containerization | Docker / Docker Compose |
| CI | GitHub Actions |
| Monitoring | Feature drift, prediction-score drift, delayed-label performance |

The production scoring architecture is:

```text
Transaction
    |
    v
FastAPI Scoring Service
    |
    +--> Feature validation
    |
    +--> Missing-value imputation
    |
    +--> XGBoost model
    |
    +--> SHAP explanation
    |
    v
Fraud Probability
    |
    +--> Decision threshold
    |
    v
Alert / No Alert
```

---

## 3. Data

The project processes simulated payment transaction data covering approximately 1.75 million transactions.

Validated ingestion statistics:

| Dataset Component | Count |
|---|---:|
| Raw daily transaction files | 183 |
| Transactions | 1,754,155 |
| Feature rows | 1,754,155 |
| Customers | 4,990 |
| Terminals | 10,000 |

The ingestion pipeline validates:

- raw source availability,
- source schema,
- PostgreSQL connectivity,
- required database tables,
- feature schema,
- transaction/feature row alignment,
- duplicate transaction IDs,
- missing feature rows,
- orphan feature rows,
- delayed fraud-history feature population.

The validated pipeline contains no duplicate transaction IDs, missing feature rows, or orphan feature rows.

---

## 4. Feature Engineering

The model uses behavioral, transaction, temporal, and terminal-risk information.

Core features include:

```text
tx_amount
during_weekend
during_night
customer_tx_count_1h
customer_tx_count_6h
customer_tx_count_24h
customer_avg_amount_24h
customer_avg_amount_7d
customer_amount_deviation
terminal_tx_count_24h
terminal_fraud_rate_7d
terminal_fraud_rate_30d
terminal_history_available
```

Customer behavioral features are calculated only from transactions occurring before the transaction being scored.

This prevents the current transaction from leaking into its own historical features.

---

## 5. Delayed Fraud Labels

Fraud labels may not be immediately available in a real production environment.

The platform therefore models delayed terminal fraud information using an assumed:

```text
7-day fraud-label delay
```

Terminal fraud-rate features use only fraud labels that would have been available at the transaction's scoring time.

Examples include:

```text
terminal_fraud_rate_7d
terminal_fraud_rate_30d
terminal_history_available
```

This design reduces target leakage and more closely represents real-time production scoring.

---

## 6. Leakage Controls

Fraud models are particularly vulnerable to temporal and target leakage.

The project explicitly audits feature leakage.

Major controls include:

1. Historical customer features exclude the current transaction.
2. Terminal fraud features respect delayed fraud-label availability.
3. Current fraud labels are not used to generate the current transaction's model features.
4. Historical windows are calculated relative to transaction time.
5. Model evaluation uses time-aware data separation rather than assuming future observations were available in the past.

These controls are essential because a high offline metric is not meaningful if the model relies on information unavailable during production scoring.

---

## 7. Class Imbalance

Fraud detection is a highly imbalanced classification problem.

Because legitimate transactions significantly outnumber fraudulent transactions, accuracy is not treated as the primary model-selection metric.

For example, a classifier predicting nearly every transaction as legitimate could achieve high accuracy while failing to detect fraud.

The platform therefore emphasizes:

- PR-AUC,
- recall,
- precision,
- fraud alert rate,
- business cost,
- threshold analysis.

---

## 8. Model Selection

Multiple modeling and evaluation stages were used during development, including baseline modeling, challenger evaluation, threshold analysis, cost sensitivity analysis, and top-K analysis.

The selected champion model is an XGBoost classifier.

The final model was selected based on its ability to identify fraudulent transactions while maintaining an operationally meaningful alert volume.

---

## 9. Model Performance

The final evaluation metrics include:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8866 |
| PR-AUC | 0.6613 |
| Recall | 75.50% |
| Decision Threshold | 0.46 |

PR-AUC is emphasized because it provides a more informative assessment than accuracy for highly imbalanced fraud-detection problems.

Recall measures the proportion of actual fraudulent transactions detected.

Precision measures the proportion of generated fraud alerts that are actually fraudulent.

Both must be evaluated together because aggressively increasing recall can substantially increase false-positive investigation workload.

---

## 10. Decision Threshold

The production decision threshold is:

```text
0.46
```

A transaction with a predicted fraud probability greater than or equal to the configured threshold can generate an alert.

The threshold was not selected solely from a generic probability cutoff such as 0.50.

Threshold evaluation considered model performance and operational trade-offs between:

- fraud detection,
- false positives,
- investigation workload,
- missed fraud,
- alert volume,
- business cost.

The threshold should therefore be considered an operational policy parameter in addition to a machine-learning parameter.

---

## 11. Explainability

The platform uses SHAP values to explain individual model predictions.

For each scored transaction, the scoring service can identify features that:

- increase predicted fraud risk, and
- reduce predicted fraud risk.

Example explanatory features include:

```text
customer_amount_deviation
tx_amount
terminal_fraud_rate_30d
customer_avg_amount_7d
customer_tx_count_24h
```

The Streamlit investigation dashboard presents these explanations alongside the fraud probability and alert decision.

SHAP explanations support investigation and model transparency but should not be interpreted as proof that a feature caused fraud.

---

## 12. API Governance

The champion model is exposed through a FastAPI scoring service.

Primary endpoints include:

```text
GET  /
GET  /health
POST /score-transaction
```

The health endpoint validates availability of key scoring components, including:

- trained model,
- imputer,
- SHAP explainer,
- decision threshold,
- expected feature configuration.

Input validation rejects invalid transaction values before model scoring.

---

## 13. Investigator Dashboard

A Streamlit dashboard provides an investigation interface over the scoring API.

The dashboard displays:

- transaction inputs,
- scoring API status,
- fraud probability,
- risk level,
- alert threshold,
- alert decision,
- recommended action,
- transaction context,
- SHAP explanations,
- model information.

The dashboard communicates with the FastAPI service rather than independently implementing model-scoring logic.

This separation keeps scoring behavior centralized in the API.

---

## 14. Model Monitoring

The platform contains three major model-monitoring layers.

### 14.1 Feature Drift

Population Stability Index (PSI) is used to compare reference and current feature distributions.

Monitoring policy:

```text
PSI < 0.10          -> STABLE
0.10 <= PSI < 0.25  -> MODERATE DRIFT
PSI >= 0.25         -> SIGNIFICANT DRIFT
```

Some features may naturally mature over time.

For example:

```text
terminal_history_available
```

changes as historical observations accumulate.

Such expected maturity is tracked separately rather than automatically interpreted as harmful model drift.

---

### 14.2 Prediction-Score Drift

The distribution of model fraud probabilities is monitored using PSI.

Observed score monitoring:

```text
Prediction-score PSI: 0.0129
Status: STABLE
```

Observed alert rates:

```text
Reference alert rate: 3.31%
Current alert rate:   3.56%
Change:              +0.25 percentage points
```

The prediction-score distribution therefore remains stable under the defined monitoring policy.

---

### 14.3 Delayed-Label Performance

Once fraud labels mature, actual model performance is evaluated.

Monitored metrics include:

- PR-AUC,
- precision,
- recall,
- alert rate,
- fraud rate,
- confusion matrix.

Current monitored performance includes:

```text
Current PR-AUC:    0.6620
Current precision: 0.1878
Current recall:    0.7554
```

The current confusion matrix is:

```text
TP = 1,924
FP = 8,321
FN =   623
TN = 277,005
```

The unified health report currently identifies:

```text
PERFORMANCE: MODERATE RECALL DEGRADATION
```

Therefore:

```text
Overall Model Health: MONITOR
```

This does not automatically mean the model should be replaced. It means performance should continue to be observed and investigated according to the monitoring policy.

---

## 15. Monitoring Policy

Performance changes are categorized using the following policy.

For PR-AUC, recall, and precision:

```text
Drop of 5–10 percentage points -> MONITOR
Drop of 10+ percentage points  -> REVIEW REQUIRED
```

Operational monitoring is additionally triggered when:

```text
Absolute alert-rate change >= 2 percentage points
Absolute fraud-rate change >= 0.5 percentage points
```

Delayed-label performance monitoring should only be interpreted after fraud outcomes have sufficiently matured.

---

## 16. Retraining and Review Triggers

Model retraining should not occur simply because a scheduled date has arrived.

Retraining or deeper model review should be considered when monitoring identifies persistent evidence such as:

- significant feature drift,
- significant prediction-score drift,
- sustained PR-AUC degradation,
- sustained recall degradation,
- sustained precision degradation,
- material changes in fraud prevalence,
- unacceptable alert volume,
- changing fraud patterns,
- changes in operational investigation capacity.

Before promoting a retrained model, it should be evaluated against the existing champion using the same temporal, leakage, threshold, and business-cost controls.

---

## 17. Human Oversight

A fraud alert represents elevated model-estimated risk.

It does not establish that fraud actually occurred.

The platform is designed to support a workflow such as:

```text
Transaction
    |
    v
ML Risk Score
    |
    v
Alert Threshold
    |
    v
Investigation Queue
    |
    v
Human Review / Downstream Decision
```

High-impact actions should incorporate appropriate operational controls and human review rather than relying solely on a model probability.

---

## 18. Known Limitations

The platform has several important limitations.

### Simulated data

The current project uses simulated transaction data. Production financial behavior may contain patterns that are absent from the development dataset.

### Fraud evolution

Fraud strategies change over time. Historical predictive relationships may therefore weaken.

### Label delay

Confirmed fraud outcomes may arrive after a delay, meaning real-time performance cannot always be measured immediately.

### False positives

Increasing fraud recall generally increases the number of legitimate transactions sent for investigation.

### Explainability

SHAP values describe model behavior but do not prove causal relationships.

### Historical maturity

Some historical features naturally change as more transaction history becomes available and must be distinguished from harmful distribution drift.

---

## 19. Reproducibility and Testing

The project includes automated tests for:

- API endpoints,
- transaction scoring,
- alert generation,
- SHAP explanations,
- input validation,
- PSI drift behavior,
- constant features,
- missing values,
- empty distributions.

The current automated test suite contains:

```text
14 tests
```

and is executed through GitHub Actions.

The project also supports containerized execution using Docker Compose for both:

```text
FastAPI scoring service
Streamlit investigation dashboard
```

This improves reproducibility between development and deployment environments.

---

## 20. Deployment Controls

The application is containerized into separate services for:

```text
fraud-api
fraud-dashboard
```

The API exposes port:

```text
8000
```

and the dashboard exposes:

```text
8501
```

Docker Compose provides a reproducible multi-service environment in which the dashboard communicates with the scoring API.

Before deployment, the following should be verified:

- automated tests pass,
- model artifacts load successfully,
- API health check succeeds,
- dashboard can reach the API,
- model threshold matches the approved threshold,
- required environment configuration is available,
- monitoring remains operational.

---

## 21. Governance Responsibilities

For this project, governance responsibilities can be organized conceptually as follows:

| Responsibility | Required Control |
|---|---|
| Data quality | Ingestion validation and schema checks |
| Feature integrity | Leakage-safe historical features |
| Model quality | Temporal performance evaluation |
| Operational policy | Threshold and cost evaluation |
| Explainability | SHAP transaction explanations |
| Reliability | Automated tests and CI |
| Reproducibility | Dockerized services |
| Monitoring | Drift and delayed-label performance |
| Model review | Defined monitoring thresholds |
| Human oversight | Alerts treated as investigation signals |

---

## 22. Model Lifecycle

The complete governed model lifecycle implemented by the project is:

```text
Raw Transaction Data
        |
        v
Data Ingestion
        |
        v
Data Validation
        |
        v
Behavioral Feature Engineering
        |
        v
Delayed Fraud Feature Engineering
        |
        v
Leakage Audit
        |
        v
Model Training
        |
        v
Champion / Challenger Evaluation
        |
        v
Threshold + Business Cost Analysis
        |
        v
Champion Model
        |
        v
FastAPI Scoring Service
        |
        +------> SHAP Explainability
        |
        v
Streamlit Investigation Dashboard
        |
        v
Dockerized Runtime
        |
        v
CI Testing
        |
        v
Feature + Score + Performance Monitoring
        |
        v
Review / Retraining Decision
```

---

## 23. Current Governance Status

| Area | Status |
|---|---|
| Data ingestion validation | HEALTHY |
| Feature leakage controls | IMPLEMENTED |
| Automated API testing | PASSING |
| Drift testing | PASSING |
| Docker containerization | OPERATIONAL |
| Prediction-score monitoring | STABLE |
| Feature monitoring | STABLE / EXPECTED FEATURE MATURITY |
| Delayed model performance | MONITOR |
| Explainability | IMPLEMENTED |
| CI pipeline | PASSING |

Current overall model-health status:

```text
MONITOR
```

The status is driven by moderate recall degradation detected by delayed-label performance monitoring.

---

## 24. Conclusion

The Fraud Detection Platform implements governance controls across the machine-learning lifecycle rather than treating model development as an isolated training exercise.

The project includes:

- validated data ingestion,
- leakage-aware behavioral features,
- delayed fraud-label handling,
- imbalanced-class evaluation,
- threshold optimization,
- business-cost analysis,
- explainable predictions,
- API-based model serving,
- investigator-facing visualization,
- automated testing,
- CI,
- Docker containerization,
- feature drift monitoring,
- prediction-score monitoring,
- delayed-label performance monitoring,
- unified model-health reporting.

These controls provide a foundation for operating the fraud model as a monitored decision-support system rather than only as an offline machine-learning experiment.