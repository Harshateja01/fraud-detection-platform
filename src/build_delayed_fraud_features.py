from collections import defaultdict, deque
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


# ============================================================
# SETTINGS
# ============================================================

LABEL_DELAY_DAYS = 7
READ_CHUNK_SIZE = 50000


# ============================================================
# ENSURE DATABASE COLUMNS EXIST
# ============================================================

with engine.begin() as connection:

    connection.execute(
        text("""
        ALTER TABLE transaction_features
        ADD COLUMN IF NOT EXISTS terminal_fraud_count_7d INTEGER;
        """)
    )

    connection.execute(
        text("""
        ALTER TABLE transaction_features
        ADD COLUMN IF NOT EXISTS terminal_fraud_count_30d INTEGER;
        """)
    )

    connection.execute(
        text("""
        ALTER TABLE transaction_features
        ADD COLUMN IF NOT EXISTS terminal_history_count_7d INTEGER;
        """)
    )

    connection.execute(
        text("""
        ALTER TABLE transaction_features
        ADD COLUMN IF NOT EXISTS terminal_history_count_30d INTEGER;
        """)
    )

    connection.execute(
        text("""
        ALTER TABLE transaction_features
        ADD COLUMN IF NOT EXISTS terminal_history_available SMALLINT;
        """)
    )


# ============================================================
# TERMINAL HISTORY
# ============================================================

# Transactions whose fraud labels are not yet available.
#
# terminal_pending[terminal_id]
#     -> deque[(timestamp, fraud_label)]

terminal_pending = defaultdict(deque)


# Transactions whose labels have become available.
#
# We retain at most 30 days of confirmed history.

terminal_confirmed = defaultdict(deque)


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_terminal_risk(
    terminal_id,
    current_time
):

    pending = terminal_pending[terminal_id]
    confirmed = terminal_confirmed[terminal_id]

    # --------------------------------------------------------
    # LABEL AVAILABILITY CUTOFF
    # --------------------------------------------------------

    label_cutoff = (
        current_time
        - pd.Timedelta(days=LABEL_DELAY_DAYS)
    )

    # --------------------------------------------------------
    # MOVE LABELS THAT HAVE BECOME AVAILABLE
    # --------------------------------------------------------

    while (
        pending
        and pending[0][0] <= label_cutoff
    ):

        confirmed.append(
            pending.popleft()
        )


    # --------------------------------------------------------
    # REMOVE CONFIRMED HISTORY OLDER THAN 30 DAYS
    #
    # Windows are measured relative to the label cutoff,
    # not the current transaction timestamp.
    # --------------------------------------------------------

    cutoff_30d = (
        label_cutoff
        - pd.Timedelta(days=30)
    )

    while (
        confirmed
        and confirmed[0][0] < cutoff_30d
    ):

        confirmed.popleft()


    # --------------------------------------------------------
    # 7-DAY WINDOW
    # --------------------------------------------------------

    cutoff_7d = (
        label_cutoff
        - pd.Timedelta(days=7)
    )

    history_7d = [
        fraud
        for timestamp, fraud in confirmed
        if timestamp >= cutoff_7d
    ]


    # --------------------------------------------------------
    # 30-DAY WINDOW
    # --------------------------------------------------------

    history_30d = [
        fraud
        for timestamp, fraud in confirmed
    ]


    history_count_7d = len(history_7d)
    history_count_30d = len(history_30d)

    fraud_count_7d = sum(history_7d)
    fraud_count_30d = sum(history_30d)


    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    if history_count_7d > 0:

        fraud_rate_7d = (
            fraud_count_7d
            / history_count_7d
        )

    else:

        fraud_rate_7d = None


    if history_count_30d > 0:

        fraud_rate_30d = (
            fraud_count_30d
            / history_count_30d
        )

    else:

        fraud_rate_30d = None


    history_available = int(
        history_count_30d > 0
    )


    return {
        "terminal_fraud_count_7d":
            fraud_count_7d,

        "terminal_fraud_count_30d":
            fraud_count_30d,

        "terminal_history_count_7d":
            history_count_7d,

        "terminal_history_count_30d":
            history_count_30d,

        "terminal_fraud_rate_7d":
            fraud_rate_7d,

        "terminal_fraud_rate_30d":
            fraud_rate_30d,

        "terminal_history_available":
            history_available,
    }


# ============================================================
# DATABASE QUERY
# ============================================================

query = """
SELECT
    transaction_id,
    tx_datetime,
    terminal_id,
    tx_fraud
FROM transactions
ORDER BY tx_datetime, transaction_id;
"""


print("=" * 70)
print("BUILDING DELAYED TERMINAL FRAUD FEATURES")
print("=" * 70)

print(
    f"\nAssumed fraud-label delay: "
    f"{LABEL_DELAY_DAYS} days"
)


processed_rows = 0


# ============================================================
# PROCESS TRANSACTIONS CHRONOLOGICALLY
# ============================================================

for chunk_number, chunk in enumerate(

    pd.read_sql(
        text(query),
        engine,
        chunksize=READ_CHUNK_SIZE,
    ),

    start=1
):

    feature_rows = []


    for row in chunk.itertuples(index=False):

        transaction_id = row.transaction_id

        current_time = pd.Timestamp(
            row.tx_datetime
        )

        terminal_id = row.terminal_id

        fraud_label = int(
            row.tx_fraud
        )


        # ====================================================
        # CALCULATE USING ONLY LABELS AVAILABLE AT THIS TIME
        # ====================================================

        terminal_features = (
            calculate_terminal_risk(
                terminal_id,
                current_time
            )
        )


        feature_rows.append({

            "transaction_id":
                transaction_id,

            **terminal_features,
        })


        # ====================================================
        # CURRENT LABEL IS NOT AVAILABLE YET
        #
        # Add to pending history AFTER feature calculation.
        # ====================================================

        terminal_pending[
            terminal_id
        ].append(
            (
                current_time,
                fraud_label
            )
        )


    # ========================================================
    # STAGING DATAFRAME
    # ========================================================

    feature_df = pd.DataFrame(
        feature_rows
    )


    # ========================================================
    # UPDATE POSTGRESQL
    # ========================================================

    with engine.begin() as connection:

        connection.execute(
            text("""
            CREATE TEMP TABLE tmp_delayed_features (

                transaction_id BIGINT,

                terminal_fraud_count_7d INTEGER,

                terminal_fraud_count_30d INTEGER,

                terminal_history_count_7d INTEGER,

                terminal_history_count_30d INTEGER,

                terminal_fraud_rate_7d NUMERIC(10, 6),

                terminal_fraud_rate_30d NUMERIC(10, 6),

                terminal_history_available SMALLINT

            ) ON COMMIT DROP;
            """)
        )


        feature_df.to_sql(
            "tmp_delayed_features",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )


        connection.execute(
            text("""
            UPDATE transaction_features tf

            SET
                terminal_fraud_count_7d =
                    tmp.terminal_fraud_count_7d,

                terminal_fraud_count_30d =
                    tmp.terminal_fraud_count_30d,

                terminal_history_count_7d =
                    tmp.terminal_history_count_7d,

                terminal_history_count_30d =
                    tmp.terminal_history_count_30d,

                terminal_fraud_rate_7d =
                    tmp.terminal_fraud_rate_7d,

                terminal_fraud_rate_30d =
                    tmp.terminal_fraud_rate_30d,

                terminal_history_available =
                    tmp.terminal_history_available

            FROM tmp_delayed_features tmp

            WHERE
                tf.transaction_id =
                tmp.transaction_id;
            """)
        )


    processed_rows += len(
        feature_df
    )


    print(
        f"Chunk {chunk_number}: "
        f"{len(feature_df):,} rows | "
        f"Total: {processed_rows:,}"
    )


print("\n" + "=" * 70)
print("DELAYED FRAUD FEATURES COMPLETE")
print("=" * 70)

print(
    f"Total transactions processed: "
    f"{processed_rows:,}"
)