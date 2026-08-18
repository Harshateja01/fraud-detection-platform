from collections import defaultdict, deque
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

connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)


# ============================================================
# SETTINGS
# ============================================================

READ_CHUNK_SIZE = 50000
WRITE_CHUNK_SIZE = 5000


# ============================================================
# HISTORY CONTAINERS
# ============================================================

# Customer history stores:
# (timestamp, amount)

customer_history = defaultdict(deque)

# Terminal history stores:
# timestamp only

terminal_history = defaultdict(deque)


# ============================================================
# HISTORY PRUNING
# ============================================================

def prune_customer_history(history, current_time):

    cutoff = current_time - pd.Timedelta(days=7)

    while history and history[0][0] < cutoff:
        history.popleft()


def prune_terminal_history(history, current_time):

    cutoff = current_time - pd.Timedelta(days=1)

    while history and history[0] < cutoff:
        history.popleft()


# ============================================================
# CUSTOMER FEATURES
# ============================================================

def calculate_customer_features(
    history,
    current_time,
    current_amount
):

    if not history:

        return {
            "customer_tx_count_1h": 0,
            "customer_tx_count_6h": 0,
            "customer_tx_count_24h": 0,
            "customer_avg_amount_24h": np.nan,
            "customer_avg_amount_7d": np.nan,
            "customer_amount_deviation": 0.0,
        }


    one_hour = (
        current_time
        - pd.Timedelta(hours=1)
    )

    six_hours = (
        current_time
        - pd.Timedelta(hours=6)
    )

    one_day = (
        current_time
        - pd.Timedelta(days=1)
    )

    seven_days = (
        current_time
        - pd.Timedelta(days=7)
    )


    count_1h = 0
    count_6h = 0
    count_24h = 0

    amounts_24h = []
    amounts_7d = []


    for timestamp, amount in history:

        if timestamp >= one_hour:
            count_1h += 1

        if timestamp >= six_hours:
            count_6h += 1

        if timestamp >= one_day:
            count_24h += 1
            amounts_24h.append(amount)

        if timestamp >= seven_days:
            amounts_7d.append(amount)


    avg_24h = (
        float(np.mean(amounts_24h))
        if amounts_24h
        else np.nan
    )

    avg_7d = (
        float(np.mean(amounts_7d))
        if amounts_7d
        else np.nan
    )


    if np.isnan(avg_7d):
        deviation = 0.0

    else:
        deviation = (
            current_amount
            - avg_7d
        )


    return {
        "customer_tx_count_1h": count_1h,
        "customer_tx_count_6h": count_6h,
        "customer_tx_count_24h": count_24h,
        "customer_avg_amount_24h": avg_24h,
        "customer_avg_amount_7d": avg_7d,
        "customer_amount_deviation": deviation,
    }


# ============================================================
# TERMINAL FEATURES
# ============================================================

def calculate_terminal_features(
    history,
    current_time
):

    if not history:
        return 0

    one_day = (
        current_time
        - pd.Timedelta(days=1)
    )

    count_24h = sum(
        1
        for timestamp in history
        if timestamp >= one_day
    )

    return count_24h


# ============================================================
# QUERY
# ============================================================

query = """
SELECT
    transaction_id,
    tx_datetime,
    customer_id,
    terminal_id,
    tx_amount
FROM transactions
ORDER BY tx_datetime, transaction_id;
"""


print("=" * 70)
print("BUILDING BEHAVIORAL FEATURES")
print("=" * 70)


processed_rows = 0


# ============================================================
# PROCESS DATABASE IN CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(

    pd.read_sql(
        text(query),
        engine,
        chunksize=READ_CHUNK_SIZE
    ),

    start=1
):

    feature_rows = []


    for row in chunk.itertuples(index=False):

        transaction_id = row.transaction_id

        current_time = pd.Timestamp(
            row.tx_datetime
        )

        customer_id = row.customer_id

        terminal_id = row.terminal_id

        amount = float(
            row.tx_amount
        )


        # ====================================================
        # REMOVE OLD HISTORY
        # ====================================================

        prune_customer_history(
            customer_history[customer_id],
            current_time
        )

        prune_terminal_history(
            terminal_history[terminal_id],
            current_time
        )


        # ====================================================
        # CUSTOMER FEATURES
        # ====================================================

        customer_features = (
            calculate_customer_features(
                customer_history[customer_id],
                current_time,
                amount
            )
        )


        # ====================================================
        # TERMINAL FEATURES
        # ====================================================

        terminal_tx_count_24h = (
            calculate_terminal_features(
                terminal_history[terminal_id],
                current_time
            )
        )


        # ====================================================
        # TIME FEATURES
        # ====================================================

        during_weekend = int(
            current_time.dayofweek >= 5
        )

        during_night = int(
            current_time.hour < 6
        )


        # ====================================================
        # CREATE FEATURE ROW
        # ====================================================

        feature_rows.append({

            "transaction_id":
                transaction_id,

            "during_weekend":
                during_weekend,

            "during_night":
                during_night,

            **customer_features,

            "terminal_tx_count_24h":
                terminal_tx_count_24h,
        })


        # ====================================================
        # ADD CURRENT TRANSACTION TO HISTORY
        #
        # IMPORTANT:
        # We do this AFTER calculating features.
        #
        # This prevents the current transaction from leaking
        # into its own historical features.
        # ====================================================

        customer_history[
            customer_id
        ].append(
            (
                current_time,
                amount
            )
        )

        terminal_history[
            terminal_id
        ].append(
            current_time
        )


    # ========================================================
    # WRITE FEATURES TO POSTGRESQL
    # ========================================================

    features_df = pd.DataFrame(
        feature_rows
    )


    features_df.to_sql(
        "transaction_features",
        engine,
        if_exists="append",
        index=False,
        chunksize=WRITE_CHUNK_SIZE,
        method="multi"
    )


    processed_rows += len(
        features_df
    )


    print(
        f"Chunk {chunk_number}: "
        f"{len(features_df):,} rows | "
        f"Total: {processed_rows:,}"
    )


print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"Total feature rows created: "
    f"{processed_rows:,}"
)