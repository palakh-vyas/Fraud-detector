"""
test_local.py — Smoke test for Fraud Detector project.
Run with: python test_local.py
"""

import sys
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = 0

print("=" * 55)
print("  Fraud Detector - Local Smoke Test")
print("=" * 55)

# ── Test 1: Python version ──────────────────────────────────
print("\n[1] Python Version")
version = sys.version_info
print(f"    Python {version.major}.{version.minor}.{version.micro}", end="  ")
if version.major == 3 and version.minor >= 9:
    print(PASS)
else:
    print(FAIL + " (need Python 3.9+)")
    errors += 1

# ── Test 2: Required imports ────────────────────────────────
print("\n[2] Required Package Imports")
packages = {
    "pandas": "pandas",
    "mysql.connector": "mysql-connector-python",
    "plotly.express": "plotly",
    "streamlit": "streamlit",
}
for module, pip_name in packages.items():
    try:
        __import__(module)
        print(f"    import {module:<22}", PASS)
    except ImportError:
        print(f"    import {module:<22}", FAIL, f"-> run: pip install {pip_name}")
        errors += 1

# ── Test 3: CSV files exist ─────────────────────────────────
print("\n[3] CSV File Presence")
for fname in ["transactions.csv", "suspicious_transactions.csv"]:
    exists = os.path.isfile(fname)
    print(f"    {fname:<38}", PASS if exists else FAIL)
    if not exists:
        errors += 1

# ── Test 4: CSV loading & fraud detection logic ─────────────
print("\n[4] CSV Load & Fraud Logic")
try:
    import pandas as pd

    df = pd.read_csv("transactions.csv")
    required_cols = {"Amount", "Merchant"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"    transactions.csv missing columns: {missing}", FAIL)
        errors += 1
    else:
        print(f"    transactions.csv loads OK ({len(df)} rows)    ", PASS)

    def is_suspicious(row):
        return row["Amount"] > 30000 or row["Merchant"] in ["Unknown", "Hackermarket"]

    df["Suspicious"] = df.apply(is_suspicious, axis=1)
    flagged = df[df["Suspicious"] == True]
    expected_merchants = {"Unknown", "Hackermarket"}
    actual_merchants = set(flagged["Merchant"].tolist())
    if actual_merchants == expected_merchants:
        print(f"    Fraud rules fire correctly ({len(flagged)} flagged) ", PASS)  # noqa
    else:
        print(f"    Fraud logic mismatch - flagged: {actual_merchants}", FAIL)
        errors += 1

    # Verify suspicious_transactions.csv matches
    sus_df = pd.read_csv("suspicious_transactions.csv")
    if set(sus_df["Merchant"].tolist()) == expected_merchants:
        print(f"    suspicious_transactions.csv matches  ", PASS)
    else:
        print(f"    suspicious_transactions.csv out of sync ", WARN)

except Exception as e:
    print(f"    Exception: {e}", FAIL)
    errors += 1

# ── Test 5: MySQL connectivity ──────────────────────────────
print("\n[5] MySQL Connectivity")
try:
    import mysql.connector

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="MaTaJ@090",
        database="fraud_detection_db",
        connection_timeout=4,
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions;")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print(f"    Connected to fraud_detection_db         ", PASS)
    print(f"    Rows in transactions table: {count}")
except Exception as e:
    short = str(e).split("\n")[0]
    print(f"    {WARN}  MySQL not reachable -> {short}")
    print(f"    (This is OK if you haven't run main.py yet or MySQL is stopped)")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 55)
if errors == 0:
    print("  All core checks passed!")
    print("  -> Run the dashboard with:  streamlit run app.py")
    print("  -> Seed the database with:  python main.py")
else:
    print(f"  {errors} check(s) failed — see above for details.")
print("=" * 55 + "\n")
