from pathlib import Path
import pandas as pd


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
# FIND DAILY FILES
# ============================================================

files = sorted(data_dir.glob("*.pkl"))

print("=" * 70)
print("FRAUD TRANSACTION DATA INSPECTION")
print("=" * 70)

print(f"\nNumber of daily files: {len(files)}")

print(f"First file: {files[0].name}")
print(f"Last file:  {files[-1].name}")


# ============================================================
# LOAD ONE DAY
# ============================================================

sample_file = files[0]

df = pd.read_pickle(sample_file)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FILE")
print("=" * 70)

print(f"File: {sample_file.name}")

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


print("\nColumn names:")

for column in df.columns:
    print(f"- {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# FIRST ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST FIVE TRANSACTIONS")
print("=" * 70)

print(df.head())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isna().sum())


# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

if "TX_FRAUD" in df.columns:

    print("\n" + "=" * 70)
    print("FRAUD DISTRIBUTION")
    print("=" * 70)

    fraud_counts = df["TX_FRAUD"].value_counts()

    print(fraud_counts)

    fraud_rate = df["TX_FRAUD"].mean() * 100

    print(
        f"\nFraud rate for sample day: "
        f"{fraud_rate:.4f}%"
    )


# ============================================================
# TRANSACTION AMOUNTS
# ============================================================

if "TX_AMOUNT" in df.columns:

    print("\n" + "=" * 70)
    print("TRANSACTION AMOUNT")
    print("=" * 70)

    print(df["TX_AMOUNT"].describe())


# ============================================================
# UNIQUE ENTITIES
# ============================================================

if "CUSTOMER_ID" in df.columns:

    print(
        f"\nUnique customers: "
        f"{df['CUSTOMER_ID'].nunique():,}"
    )

if "TERMINAL_ID" in df.columns:

    print(
        f"Unique terminals: "
        f"{df['TERMINAL_ID'].nunique():,}"
    )