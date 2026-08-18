import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simulated-data-transformed"
    / "data"
)


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


# ============================================================
# REQUIRED DATABASE TABLES
# ============================================================

REQUIRED_TABLES = {
    "customers",
    "terminals",
    "transactions",
    "transaction_features",
}


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

REQUIRED_FEATURE_COLUMNS = {
    "transaction_id",
    "during_weekend",
    "during_night",
    "customer_tx_count_1h",
    "customer_tx_count_6h",
    "customer_tx_count_24h",
    "customer_avg_amount_24h",
    "customer_avg_amount_7d",
    "customer_amount_deviation",
    "terminal_tx_count_24h",
    "terminal_fraud_rate_7d",
    "terminal_fraud_rate_30d",
    "terminal_history_available",
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_detection")


if not DB_USER or not DB_PASSWORD:
    raise RuntimeError(
        "Database credentials are missing. "
        "Check DB_USER and DB_PASSWORD in your .env file."
    )


CONNECTION_STRING = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    CONNECTION_STRING
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title, width=80):

    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_success(message):

    print(
        f"[OK] {message}"
    )


def print_warning(message):

    print(
        f"[WARNING] {message}"
    )


def print_failure(message):

    print(
        f"[FAILED] {message}"
    )


# ============================================================
# RAW DATA VALIDATION
# ============================================================

def validate_raw_data():

    print_header(
        "1. RAW DATA VALIDATION"
    )

    if not RAW_DATA_DIR.exists():

        raise FileNotFoundError(
            f"Raw data directory does not exist:\n"
            f"{RAW_DATA_DIR}"
        )


    files = sorted(
        RAW_DATA_DIR.glob("*.pkl")
    )


    if not files:

        raise FileNotFoundError(
            f"No .pkl transaction files found in:\n"
            f"{RAW_DATA_DIR}"
        )


    print_success(
        f"Found {len(files)} raw transaction files."
    )


    # --------------------------------------------------------
    # VALIDATE FIRST AND LAST FILE
    #
    # This provides schema validation without reading every
    # large pickle file into memory.
    # --------------------------------------------------------

    files_to_check = [
        files[0]
    ]

    if len(files) > 1:
        files_to_check.append(
            files[-1]
        )


    for file_path in files_to_check:

        df = pd.read_pickle(
            file_path
        )

        missing_columns = (
            REQUIRED_RAW_COLUMNS
            - set(df.columns)
        )


        if missing_columns:

            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )


        print_success(
            f"{file_path.name}: "
            f"{len(df):,} rows, schema valid."
        )


    return files


# ============================================================
# DATABASE CONNECTIVITY
# ============================================================

def validate_database_connection():

    print_header(
        "2. DATABASE CONNECTION"
    )


    with engine.connect() as connection:

        result = connection.execute(
            text(
                "SELECT current_database();"
            )
        )

        current_database = (
            result.scalar_one()
        )


    print_success(
        f"Connected to PostgreSQL database: "
        f"{current_database}"
    )


# ============================================================
# TABLE VALIDATION
# ============================================================

def get_existing_tables():

    inspector = inspect(
        engine
    )

    return set(
        inspector.get_table_names()
    )


def validate_required_tables():

    print_header(
        "3. DATABASE TABLE VALIDATION"
    )


    existing_tables = (
        get_existing_tables()
    )


    missing_tables = (
        REQUIRED_TABLES
        - existing_tables
    )


    if missing_tables:

        print_failure(
            "Required database tables are missing."
        )

        print(
            "Missing:"
        )

        for table in sorted(
            missing_tables
        ):
            print(
                f"- {table}"
            )

        return False


    for table in sorted(
        REQUIRED_TABLES
    ):

        print_success(
            f"Table exists: {table}"
        )


    return True


# ============================================================
# TABLE COUNTS
# ============================================================

def get_table_count(
    connection,
    table_name,
):

    query = text(
        f"SELECT COUNT(*) "
        f"FROM {table_name};"
    )

    return int(
        connection.execute(
            query
        ).scalar_one()
    )


def report_table_counts():

    print_header(
        "4. INGESTION ROW COUNTS"
    )


    with engine.connect() as connection:

        counts = {
            table:
                get_table_count(
                    connection,
                    table,
                )
            for table in sorted(
                REQUIRED_TABLES
            )
        }


    for table, count in (
        counts.items()
    ):

        print(
            f"{table:<25} "
            f"{count:>12,}"
        )


    return counts


# ============================================================
# FEATURE TABLE SCHEMA VALIDATION
# ============================================================

def validate_feature_schema():

    print_header(
        "5. FEATURE SCHEMA VALIDATION"
    )


    inspector = inspect(
        engine
    )

    columns = inspector.get_columns(
        "transaction_features"
    )

    existing_columns = {
        column["name"]
        for column in columns
    }


    missing_columns = (
        REQUIRED_FEATURE_COLUMNS
        - existing_columns
    )


    if missing_columns:

        print_failure(
            "transaction_features is missing "
            "required model feature columns."
        )

        for column in sorted(
            missing_columns
        ):

            print(
                f"- {column}"
            )


        return False


    for column in sorted(
        REQUIRED_FEATURE_COLUMNS
    ):

        print_success(
            f"Feature available: {column}"
        )


    return True


# ============================================================
# PIPELINE QUALITY CHECKS
# ============================================================

def validate_pipeline_quality():

    print_header(
        "6. PIPELINE QUALITY CHECKS"
    )


    checks = {}


    with engine.connect() as connection:

        # ----------------------------------------------------
        # TRANSACTION COUNT
        # ----------------------------------------------------

        transaction_count = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM transactions;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # FEATURE COUNT
        # ----------------------------------------------------

        feature_count = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM transaction_features;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # DUPLICATE TRANSACTIONS
        # ----------------------------------------------------

        transaction_duplicates = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT transaction_id
                        FROM transactions
                        GROUP BY transaction_id
                        HAVING COUNT(*) > 1
                    ) duplicates;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # DUPLICATE FEATURE ROWS
        # ----------------------------------------------------

        feature_duplicates = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT transaction_id
                        FROM transaction_features
                        GROUP BY transaction_id
                        HAVING COUNT(*) > 1
                    ) duplicates;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # TRANSACTIONS WITHOUT FEATURE ROW
        # ----------------------------------------------------

        missing_features = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)

                    FROM transactions t

                    LEFT JOIN transaction_features f
                        ON t.transaction_id =
                           f.transaction_id

                    WHERE f.transaction_id IS NULL;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # FEATURE ROWS WITHOUT TRANSACTION
        # ----------------------------------------------------

        orphan_features = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)

                    FROM transaction_features f

                    LEFT JOIN transactions t
                        ON f.transaction_id =
                           t.transaction_id

                    WHERE t.transaction_id IS NULL;
                    """
                )
            ).scalar_one()
        )


        # ----------------------------------------------------
        # DELAYED FEATURE POPULATION
        # ----------------------------------------------------

        delayed_feature_rows = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)

                    FROM transaction_features

                    WHERE
                        terminal_history_available
                        IS NOT NULL;
                    """
                )
            ).scalar_one()
        )


    # ========================================================
    # DISPLAY CHECKS
    # ========================================================

    checks[
        "transaction_feature_row_match"
    ] = (
        transaction_count
        == feature_count
    )

    checks[
        "transaction_duplicates"
    ] = (
        transaction_duplicates
        == 0
    )

    checks[
        "feature_duplicates"
    ] = (
        feature_duplicates
        == 0
    )

    checks[
        "missing_feature_rows"
    ] = (
        missing_features
        == 0
    )

    checks[
        "orphan_feature_rows"
    ] = (
        orphan_features
        == 0
    )

    checks[
        "delayed_features_populated"
    ] = (
        delayed_feature_rows
        == feature_count
    )


    print(
        f"Transactions:                    "
        f"{transaction_count:,}"
    )

    print(
        f"Feature rows:                    "
        f"{feature_count:,}"
    )

    print(
        f"Duplicate transaction IDs:       "
        f"{transaction_duplicates:,}"
    )

    print(
        f"Duplicate feature transaction IDs:"
        f" {feature_duplicates:,}"
    )

    print(
        f"Transactions missing features:    "
        f"{missing_features:,}"
    )

    print(
        f"Orphan feature rows:              "
        f"{orphan_features:,}"
    )

    print(
        f"Rows with delayed-history status: "
        f"{delayed_feature_rows:,}"
    )


    print(
        "\nQuality checks:"
    )


    for check_name, passed in (
        checks.items()
    ):

        if passed:

            print_success(
                check_name
            )

        else:

            print_failure(
                check_name
            )


    return all(
        checks.values()
    )


# ============================================================
# RUN EXISTING PIPELINE SCRIPT
# ============================================================

def run_script(
    script_name,
):

    script_path = (
        SRC_DIR
        / script_name
    )


    if not script_path.exists():

        raise FileNotFoundError(
            f"Pipeline script not found: "
            f"{script_path}"
        )


    print_header(
        f"RUNNING {script_name}"
    )


    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"{script_name} failed "
            f"with exit code "
            f"{result.returncode}."
        )


    print_success(
        f"{script_name} completed successfully."
    )


# ============================================================
# CLEAR EXISTING INGESTED DATA
# ============================================================

def reset_database_data():

    print_header(
        "RESETTING PIPELINE TABLES"
    )


    existing_tables = (
        get_existing_tables()
    )


    required_for_reset = (
        REQUIRED_TABLES
        & existing_tables
    )


    if not required_for_reset:

        raise RuntimeError(
            "No pipeline tables were found."
        )


    with engine.begin() as connection:

        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    transaction_features,
                    transactions,
                    customers,
                    terminals
                RESTART IDENTITY
                CASCADE;
                """
            )
        )


    print_success(
        "Existing ingestion and feature data cleared."
    )


# ============================================================
# FULL REBUILD
# ============================================================

def rebuild_pipeline():

    print_header(
        "FULL DATA INGESTION PIPELINE REBUILD"
    )


    print(
        "\nWARNING:"
    )

    print(
        "This operation deletes existing rows from:"
    )

    print(
        "- transaction_features"
    )

    print(
        "- transactions"
    )

    print(
        "- customers"
    )

    print(
        "- terminals"
    )


    confirmation = input(
        "\nType REBUILD to continue: "
    )


    if confirmation != "REBUILD":

        print(
            "\nRebuild cancelled."
        )

        return False


    reset_database_data()


    # --------------------------------------------------------
    # STAGE 1 — RAW TRANSACTION INGESTION
    # --------------------------------------------------------

    run_script(
        "load_transactions.py"
    )


    # --------------------------------------------------------
    # STAGE 2 — BEHAVIORAL FEATURE ENGINEERING
    # --------------------------------------------------------

    run_script(
        "build_behavioral_features.py"
    )


    # --------------------------------------------------------
    # STAGE 3 — DELAYED TERMINAL-RISK FEATURES
    # --------------------------------------------------------

    run_script(
        "build_delayed_fraud_features.py"
    )


    return True


# ============================================================
# VALIDATION PIPELINE
# ============================================================

def run_validation():

    print_header(
        "FRAUD DETECTION — DATA INGESTION PIPELINE"
    )


    validate_raw_data()

    validate_database_connection()


    tables_ok = (
        validate_required_tables()
    )


    if not tables_ok:

        raise RuntimeError(
            "Database table validation failed."
        )


    report_table_counts()


    schema_ok = (
        validate_feature_schema()
    )


    if not schema_ok:

        raise RuntimeError(
            "Feature schema validation failed."
        )


    quality_ok = (
        validate_pipeline_quality()
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header(
        "DATA INGESTION PIPELINE STATUS"
    )


    if quality_ok:

        print(
            "\nSTATUS: HEALTHY"
        )

        print(
            "\nRaw transaction ingestion, "
            "behavioral feature engineering, "
            "and delayed terminal-risk feature "
            "generation are complete."
        )

        print(
            "\nThe database is ready for "
            "model training and scoring."
        )


    else:

        print(
            "\nSTATUS: REVIEW REQUIRED"
        )

        print(
            "\nOne or more pipeline quality "
            "checks failed."
        )


    print_header(
        "DATA INGESTION PIPELINE COMPLETE"
    )


    return quality_ok


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Validate or rebuild the fraud "
            "detection data ingestion pipeline."
        )
    )


    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete existing ingestion data and "
            "run the complete pipeline from raw files."
        ),
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()


    try:

        # ----------------------------------------------------
        # ALWAYS VALIDATE RAW SOURCE + DATABASE FIRST
        # ----------------------------------------------------

        validate_raw_data()

        validate_database_connection()


        if args.rebuild:

            existing_tables = (
                validate_required_tables()
            )


            if not existing_tables:

                raise RuntimeError(
                    "Required database tables must "
                    "exist before rebuilding."
                )


            rebuilt = rebuild_pipeline()


            if not rebuilt:
                return


        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        run_validation()


    except Exception as exc:

        print_header(
            "DATA INGESTION PIPELINE FAILED"
        )

        print(
            f"\nError: {exc}"
        )

        sys.exit(
            1
        )


if __name__ == "__main__":

    main()