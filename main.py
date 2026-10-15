"""
main.py
-------
Seeds the SQLite database (fraud_detection.db) from transactions.csv.
Run after generate_data.py has produced the CSV.

Usage:  python main.py
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "fraud_detection.db"
CSV_PATH = "transactions.csv"


def seed_database():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run generate_data.py first.")
        return

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded.")

    # Connect / create SQLite DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop + recreate table for a clean seed
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("""
        CREATE TABLE transactions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            TransactionID     TEXT,
            Date              TEXT,
            Hour              INTEGER,
            DayOfWeek         INTEGER,
            IsWeekend         INTEGER,
            Amount            REAL,
            Type              TEXT,
            Merchant          TEXT,
            MerchantCategory  TEXT,
            MerchantRiskScore REAL,
            UserID            TEXT,
            TxFrequency       INTEGER,
            AvgUserAmount     REAL,
            StdUserAmount     REAL,
            AmountDeviation   REAL,
            IsHighAmount      INTEGER,
            IsNightTx         INTEGER,
            HourSin           REAL,
            HourCos           REAL,
            Suspicious        INTEGER
        )
    """)

    # Insert rows
    cols = [
        "TransactionID", "Date", "Hour", "DayOfWeek", "IsWeekend",
        "Amount", "Type", "Merchant", "MerchantCategory", "MerchantRiskScore",
        "UserID", "TxFrequency", "AvgUserAmount", "StdUserAmount",
        "AmountDeviation", "IsHighAmount", "IsNightTx", "HourSin", "HourCos",
        "Suspicious",
    ]
    for _, row in df[cols].iterrows():
        cursor.execute(
            f"INSERT INTO transactions ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})",
            tuple(row[c] for c in cols)
        )

    conn.commit()

    # Verify
    count = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud = cursor.execute("SELECT COUNT(*) FROM transactions WHERE Suspicious=1").fetchone()[0]
    cursor.close()
    conn.close()

    print(f"Database seeded: {count} rows in {DB_PATH}")
    print(f"  Suspicious: {fraud} | Safe: {count - fraud}")

    # Also save suspicious_transactions.csv
    suspicious_df = df[df["Suspicious"] == 1]
    suspicious_df.to_csv("suspicious_transactions.csv", index=False)
    print(f"  suspicious_transactions.csv updated ({len(suspicious_df)} rows)")


if __name__ == "__main__":
    seed_database()
