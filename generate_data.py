"""
generate_data.py
----------------
Generates a realistic 2000-row synthetic transaction dataset for the
Fraud Detector ML portfolio project.

Features generated:
  Date, Hour, DayOfWeek, IsWeekend, Amount, Type, Merchant,
  MerchantCategory, UserID, TxFrequency, AvgUserAmount,
  AmountDeviation, Label (0=safe, 1=fraud)

Run:  python generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_TRANSACTIONS = 2000
N_USERS = 120
FRAUD_RATE = 0.08          # 8% positive (realistic imbalance)
START_DATE = datetime(2025, 1, 1)

# Merchant catalogue  {name: (category, base_risk)}
MERCHANTS = {
    # Legit low-risk
    "Zomato":        ("Food & Dining",       0.01),
    "Swiggy":        ("Food & Dining",       0.01),
    "Amazon":        ("E-Commerce",          0.03),
    "Flipkart":      ("E-Commerce",          0.03),
    "NEFT Transfer": ("Bank Transfer",       0.05),
    "HDFC Bank":     ("Banking",             0.02),
    "BigBazaar":     ("Grocery",             0.01),
    "Netflix":       ("Streaming",           0.01),
    "Uber":          ("Transport",           0.02),
    "Ola":           ("Transport",           0.02),
    "PhonePe":       ("UPI Payment",         0.04),
    "Paytm":         ("UPI Payment",         0.04),
    "MakeMyTrip":    ("Travel",              0.06),
    "BookMyShow":    ("Entertainment",       0.02),
    "Reliance Fresh":("Grocery",             0.01),
    # Moderate risk
    "CryptoXchange": ("Cryptocurrency",      0.35),
    "OnlineLottery": ("Gambling",            0.40),
    "QuickLoan":     ("Lending",             0.30),
    # High-risk / fraud-associated
    "Unknown":       ("Unclassified",        0.80),
    "Hackermarket":  ("Dark Web",            0.95),
    "ShellCorp LLC": ("Shell Company",       0.90),
    "FastCashNow":   ("Predatory Lending",   0.85),
}

MERCHANT_NAMES = list(MERCHANTS.keys())
TX_TYPES = ["Debit", "Credit", "UPI", "NEFT", "IMPS"]


def _random_amount(is_fraud: bool, merchant: str) -> float:
    if is_fraud:
        # Fraudulent tx tend to be very high or suspiciously round
        choice = np.random.choice(["high", "round"], p=[0.6, 0.4])
        if choice == "high":
            return round(np.random.uniform(45000, 200000), 2)
        else:
            return float(np.random.choice([50000, 99999, 100000, 75000, 60000]))
    else:
        cat = MERCHANTS[merchant][0]
        ranges = {
            "Food & Dining":    (100,   2000),
            "E-Commerce":       (200,  15000),
            "Bank Transfer":    (1000, 40000),
            "Banking":          (500,  20000),
            "Grocery":          (100,   5000),
            "Streaming":        (199,    799),
            "Transport":        (50,    2000),
            "UPI Payment":      (50,   10000),
            "Travel":           (500,  50000),
            "Entertainment":    (100,   3000),
            "Cryptocurrency":   (5000, 80000),
            "Gambling":         (500,  10000),
            "Lending":          (5000, 50000),
            "Unclassified":     (100, 100000),
            "Dark Web":         (1000, 100000),
            "Shell Company":    (10000,150000),
            "Predatory Lending":(5000, 80000),
        }
        lo, hi = ranges.get(cat, (100, 20000))
        return round(np.random.uniform(lo, hi), 2)


def generate_dataset() -> pd.DataFrame:
    records = []
    user_ids = [f"USR{str(i).zfill(4)}" for i in range(1, N_USERS + 1)]

    # Track per-user stats for feature engineering
    user_tx_counts: dict[str, list] = {u: [] for u in user_ids}

    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    fraud_flags = [1] * n_fraud + [0] * (N_TRANSACTIONS - n_fraud)
    random.shuffle(fraud_flags)

    for i in range(N_TRANSACTIONS):
        is_fraud = bool(fraud_flags[i])

        # Choose merchant (fraudulent tx favour high-risk merchants)
        if is_fraud:
            weights = [MERCHANTS[m][1] for m in MERCHANT_NAMES]
            total = sum(weights)
            weights = [w / total for w in weights]
            merchant = np.random.choice(MERCHANT_NAMES, p=weights)
        else:
            # Safe tx mostly pick legit merchants
            legit = [m for m, (_, r) in MERCHANTS.items() if r < 0.10]
            merchant = random.choice(legit)

        amount = _random_amount(is_fraud, merchant)
        category = MERCHANTS[merchant][0]
        tx_type = random.choice(TX_TYPES)
        user_id = random.choice(user_ids)

        # Time features — fraud slightly more common at night
        if is_fraud:
            fraud_hours = list(range(0, 5)) + list(range(22, 24)) + list(range(5, 22))
            fraud_raw_p = [0.05]*5 + [0.05]*2 + [0.03]*17
            fraud_norm_p = [x / sum(fraud_raw_p) for x in fraud_raw_p]
            hour = int(np.random.choice(fraud_hours, p=fraud_norm_p))
        else:
            raw_p = [
                0.01,0.005,0.005,0.005,0.01,0.02,0.04,0.06,0.07,
                0.07,0.07,0.06,0.06,0.06,0.06,0.06,0.06,0.05,
                0.04,0.04,0.03,0.03,0.02,0.015
            ]
            norm_p = [x / sum(raw_p) for x in raw_p]
            hour = int(np.random.choice(range(24), p=norm_p))

        days_offset = random.randint(0, 364)
        tx_date = START_DATE + timedelta(days=days_offset, hours=hour,
                                         minutes=random.randint(0, 59))
        day_of_week = tx_date.weekday()   # 0=Mon … 6=Sun
        is_weekend = int(day_of_week >= 5)

        user_tx_counts[user_id].append(amount)

        records.append({
            "TransactionID": f"TXN{str(i+1).zfill(5)}",
            "Date":          tx_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Hour":          hour,
            "DayOfWeek":     day_of_week,
            "IsWeekend":     is_weekend,
            "Amount":        amount,
            "Type":          tx_type,
            "Merchant":      merchant,
            "MerchantCategory": category,
            "MerchantRiskScore": MERCHANTS[merchant][1],
            "UserID":        user_id,
            "Label":         int(is_fraud),
        })

    df = pd.DataFrame(records)

    # ── Feature Engineering ──────────────────────────────────────────────────
    # Per-user aggregate features (computed on full dataset, not leaking future)
    user_stats = (
        df.groupby("UserID")["Amount"]
          .agg(AvgUserAmount="mean", StdUserAmount="std")
          .fillna(0)
          .reset_index()
    )
    df = df.merge(user_stats, on="UserID", how="left")

    user_freq = df.groupby("UserID").size().reset_index(name="TxFrequency")
    df = df.merge(user_freq, on="UserID", how="left")

    df["AmountDeviation"] = (df["Amount"] - df["AvgUserAmount"]) / (df["StdUserAmount"] + 1)
    df["IsHighAmount"]    = (df["Amount"] > 30000).astype(int)
    df["IsNightTx"]       = (df["Hour"].between(0, 5) | df["Hour"].between(22, 23)).astype(int)
    df["HourSin"]         = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"]         = np.cos(2 * np.pi * df["Hour"] / 24)

    # Rename Label -> Suspicious for compatibility with existing schema
    df.rename(columns={"Label": "Suspicious"}, inplace=True)

    # Sort by date
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    df = generate_dataset()
    df.to_csv("transactions.csv", index=False)
    print(f"Done: {len(df)} transactions saved to transactions.csv")
    print(f"  Fraud:  {df['Suspicious'].sum()} ({df['Suspicious'].mean()*100:.1f}%)")
    print(f"  Safe:   {(df['Suspicious']==0).sum()}")
    print(f"  Users:  {df['UserID'].nunique()}")
    print(f"  Period: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Columns: {list(df.columns)}")
