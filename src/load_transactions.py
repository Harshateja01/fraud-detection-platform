from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_dir = (
    project_root
    / "data"
    / "raw"
    / "simulated-data-transformed"
    / "data"
)


# ============================================================
# ENVIRONMENT VARIABLES
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
# FIND FILES
# ============================================================

files = sorted(data_dir.glob("*.pkl"))

print("=" * 70)
print("TRANSACTION INGESTION")
print("=" * 70)

print(f"\nDaily files found: {len(files)}")


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file_number, file_path in enumerate(files, start=1):

    print(
        f"\n[{file_number}/{len(files)}] "
        f"Loading {file_path.name}"
    )

    df = pd.read_pickle(file_path)


    # --------------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------------

    customers = (
        df[["CUSTOMER_ID"]]
        .drop_duplicates()
        .rename(
            columns={
                "CUSTOMER_ID": "customer_id"
            }
        )
    )


    # --------------------------------------------------------
    # TERMINALS
    # --------------------------------------------------------

    terminals = (
        df[["TERMINAL_ID"]]
        .drop_duplicates()
        .rename(
            columns={
                "TERMINAL_ID": "terminal_id"
            }
        )
    )


    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    transactions = df[
        [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    ].rename(
        columns={
            "TRANSACTION_ID": "transaction_id",
            "TX_DATETIME": "tx_datetime",
            "CUSTOMER_ID": "customer_id",
            "TERMINAL_ID": "terminal_id",
            "TX_AMOUNT": "tx_amount",
            "TX_TIME_SECONDS": "tx_time_seconds",
            "TX_TIME_DAYS": "tx_time_days",
            "TX_FRAUD": "tx_fraud",
            "TX_FRAUD_SCENARIO": "tx_fraud_scenario",
        }
    )


    # --------------------------------------------------------
    # INSERT DATA
    # --------------------------------------------------------

    with engine.begin() as connection:

        connection.execute(
            text("""
                CREATE TEMP TABLE temp_customers (
                    customer_id BIGINT
                ) ON COMMIT DROP;
            """)
        )

        customers.to_sql(
            "temp_customers",
            connection,
            if_exists="append",
            index=False,
            method="multi",
        )

        connection.execute(
            text("""
                INSERT INTO customers (customer_id)
                SELECT DISTINCT customer_id
                FROM temp_customers
                ON CONFLICT (customer_id) DO NOTHING;
            """)
        )


        connection.execute(
            text("""
                CREATE TEMP TABLE temp_terminals (
                    terminal_id BIGINT
                ) ON COMMIT DROP;
            """)
        )

        terminals.to_sql(
            "temp_terminals",
            connection,
            if_exists="append",
            index=False,
            method="multi",
        )

        connection.execute(
            text("""
                INSERT INTO terminals (terminal_id)
                SELECT DISTINCT terminal_id
                FROM temp_terminals
                ON CONFLICT (terminal_id) DO NOTHING;
            """)
        )


        transactions.to_sql(
            "transactions",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2000,
        )

    print(
        f"Inserted {len(transactions):,} transactions"
    )


print("\n" + "=" * 70)
print("INGESTION COMPLETE")
print("=" * 70)