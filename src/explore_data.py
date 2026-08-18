import os

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

connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)


def run_query(query):
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


print("=" * 70)
print("FRAUD DETECTION — EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 1. OVERALL DATASET
# ============================================================

overview = run_query("""
SELECT
    COUNT(*) AS transactions,
    COUNT(DISTINCT customer_id) AS customers,
    COUNT(DISTINCT terminal_id) AS terminals,
    MIN(tx_datetime) AS first_transaction,
    MAX(tx_datetime) AS last_transaction,
    ROUND(AVG(tx_amount), 2) AS avg_amount,
    ROUND(MAX(tx_amount), 2) AS max_amount
FROM transactions;
""")

print("\nDATASET OVERVIEW")
print(overview.to_string(index=False))


# ============================================================
# 2. FRAUD DISTRIBUTION
# ============================================================

fraud_distribution = run_query("""
SELECT
    tx_fraud,
    COUNT(*) AS transactions,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        4
    ) AS percentage
FROM transactions
GROUP BY tx_fraud
ORDER BY tx_fraud;
""")

print("\nFRAUD DISTRIBUTION")
print(fraud_distribution.to_string(index=False))


# ============================================================
# 3. TRANSACTION AMOUNT BY FRAUD STATUS
# ============================================================

amount_analysis = run_query("""
SELECT
    tx_fraud,
    COUNT(*) AS transactions,
    ROUND(AVG(tx_amount), 2) AS avg_amount,
    ROUND(MIN(tx_amount), 2) AS min_amount,
    ROUND(MAX(tx_amount), 2) AS max_amount
FROM transactions
GROUP BY tx_fraud
ORDER BY tx_fraud;
""")

print("\nTRANSACTION AMOUNT BY FRAUD STATUS")
print(amount_analysis.to_string(index=False))


# ============================================================
# 4. FRAUD SCENARIOS
# ============================================================

fraud_scenarios = run_query("""
SELECT
    tx_fraud_scenario,
    COUNT(*) AS transactions,
    SUM(tx_fraud) AS fraud_transactions
FROM transactions
GROUP BY tx_fraud_scenario
ORDER BY tx_fraud_scenario;
""")

print("\nFRAUD SCENARIOS")
print(fraud_scenarios.to_string(index=False))


# ============================================================
# 5. FRAUD BY HOUR
# ============================================================

fraud_hour = run_query("""
SELECT
    EXTRACT(HOUR FROM tx_datetime)::INTEGER AS hour,
    COUNT(*) AS transactions,
    SUM(tx_fraud) AS fraud_transactions,
    ROUND(
        SUM(tx_fraud) * 100.0 / COUNT(*),
        4
    ) AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;
""")

print("\nFRAUD BY HOUR")
print(fraud_hour.to_string(index=False))


# ============================================================
# 6. FRAUD BY DAY OF WEEK
# ============================================================

fraud_day = run_query("""
SELECT
    EXTRACT(DOW FROM tx_datetime)::INTEGER AS day_of_week,
    COUNT(*) AS transactions,
    SUM(tx_fraud) AS fraud_transactions,
    ROUND(
        SUM(tx_fraud) * 100.0 / COUNT(*),
        4
    ) AS fraud_rate_pct
FROM transactions
GROUP BY day_of_week
ORDER BY day_of_week;
""")

print("\nFRAUD BY DAY OF WEEK")
print(fraud_day.to_string(index=False))


# ============================================================
# 7. HIGHEST-RISK TERMINALS
# ============================================================

terminal_risk = run_query("""
SELECT
    terminal_id,
    COUNT(*) AS transactions,
    SUM(tx_fraud) AS fraud_transactions,
    ROUND(
        SUM(tx_fraud) * 100.0 / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM transactions
GROUP BY terminal_id
HAVING COUNT(*) >= 50
ORDER BY fraud_rate_pct DESC, fraud_transactions DESC
LIMIT 20;
""")

print("\nTOP HIGH-RISK TERMINALS")
print(terminal_risk.to_string(index=False))


# ============================================================
# 8. CUSTOMERS WITH MOST FRAUD
# ============================================================

customer_fraud = run_query("""
SELECT
    customer_id,
    COUNT(*) AS transactions,
    SUM(tx_fraud) AS fraud_transactions,
    ROUND(
        SUM(tx_fraud) * 100.0 / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM transactions
GROUP BY customer_id
HAVING SUM(tx_fraud) > 0
ORDER BY fraud_transactions DESC
LIMIT 20;
""")

print("\nCUSTOMERS WITH MOST FRAUD TRANSACTIONS")
print(customer_fraud.to_string(index=False))


print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)