"""
ETL Pipeline - Financial Fraud Detection
------------------------------------------
Extract: Read raw synthetic transactions CSV
Transform: Clean, validate, normalize, handle missing values
Load: Write into SQLite database (fraud_detection.db) as the 'transactions' table

This mirrors the ETL pattern referenced in the project requirement document.
"""

import sqlite3
import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/fraud_project/fraud_transactions.csv"
DB_PATH = "/home/claude/fraud_project/fraud_detection.db"

print("EXTRACT: Reading raw transaction data...")
df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

print("\nTRANSFORM: Cleaning & normalizing...")

# --- Handle missing values ---
missing_before = df.isnull().sum().sum()
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")

missing_after = df.isnull().sum().sum()
print(f"  Missing values: {missing_before} -> {missing_after}")

# --- Validate / clip negative or impossible values ---
df["amount"] = df["amount"].clip(lower=0)
df["distance_from_home_km"] = df["distance_from_home_km"].clip(lower=0)
df["minutes_since_last_txn"] = df["minutes_since_last_txn"].clip(lower=0)

# --- Type normalization ---
df["is_fraud"] = df["is_fraud"].astype(int)
df["is_new_device"] = df["is_new_device"].astype(bool)
df["is_weekend"] = df["is_weekend"].astype(bool)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Deduplicate ---
dupes = df.duplicated(subset=["transaction_id"]).sum()
df = df.drop_duplicates(subset=["transaction_id"])
print(f"  Duplicate transaction_ids removed: {dupes}")

# --- Min-max normalize amount for use as a model feature (kept alongside raw amount) ---
df["amount_normalized"] = (
    (df["amount"] - df["amount"].min()) / (df["amount"].max() - df["amount"].min())
).round(4)

print(f"\n  Final shape: {df.shape}")

print("\nLOAD: Writing to SQLite database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS transactions")
cursor.execute("""
    CREATE TABLE transactions (
        transaction_id TEXT PRIMARY KEY,
        customer_id TEXT,
        timestamp TEXT,
        amount REAL,
        amount_normalized REAL,
        merchant_category TEXT,
        channel TEXT,
        city TEXT,
        latitude REAL,
        longitude REAL,
        distance_from_home_km REAL,
        device_type TEXT,
        is_new_device INTEGER,
        card_type TEXT,
        customer_risk_segment TEXT,
        account_age_days INTEGER,
        hour_of_day INTEGER,
        day_of_week TEXT,
        is_weekend INTEGER,
        month TEXT,
        minutes_since_last_txn REAL,
        avg_monthly_spend REAL,
        amount_to_avg_spend_ratio REAL,
        is_fraud INTEGER,
        fraud_pattern TEXT
    )
""")

# Convert timestamp to string for SQLite storage, bools to int
df_sql = df.copy()
df_sql["timestamp"] = df_sql["timestamp"].astype(str)
df_sql["is_new_device"] = df_sql["is_new_device"].astype(int)
df_sql["is_weekend"] = df_sql["is_weekend"].astype(int)

df_sql.to_sql("transactions", conn, if_exists="replace", index=False)

# Helpful indexes for dashboard query performance
cursor.execute("CREATE INDEX idx_customer ON transactions(customer_id)")
cursor.execute("CREATE INDEX idx_fraud ON transactions(is_fraud)")
cursor.execute("CREATE INDEX idx_timestamp ON transactions(timestamp)")

conn.commit()

row_count = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
fraud_count = cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
conn.close()

print(f"  Rows in DB: {row_count:,}")
print(f"  Fraud rows: {fraud_count:,} ({fraud_count/row_count*100:.2f}%)")
print("\nETL process completed.")
