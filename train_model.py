"""
train_model.py
--------------
XGBoost fraud classifier with SMOTE oversampling and SHAP explainability.

Steps:
  1. Load transactions.csv
  2. Feature engineering
  3. Train/test split (stratified 80/20)
  4. SMOTE on training set
  5. XGBoost classifier
  6. Evaluation metrics + confusion matrix
  7. SHAP feature importance
  8. Save model & feature list to model/

Run:  python train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")      # non-interactive backend for headless saving
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "fraud_model.pkl")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_names.pkl")
SHAP_PATH   = os.path.join(MODEL_DIR, "shap_values.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "model_report.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Amount", "Hour", "DayOfWeek", "IsWeekend",
    "MerchantRiskScore", "TxFrequency", "AvgUserAmount",
    "AmountDeviation", "IsHighAmount", "IsNightTx",
    "HourSin", "HourCos",
    # Encoded categoricals
    "Type_enc", "MerchantCategory_enc",
]

TARGET = "Suspicious"


def load_and_prepare(csv_path: str = "transactions.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Encode categoricals
    le_type = LabelEncoder()
    le_cat  = LabelEncoder()
    df["Type_enc"]             = le_type.fit_transform(df["Type"].astype(str))
    df["MerchantCategory_enc"] = le_cat.fit_transform(df["MerchantCategory"].astype(str))

    # Save encoders alongside model
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump({"type": le_type, "merchant_cat": le_cat}, f)

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET].astype(int)
    return X, y, df


def train(csv_path: str = "transactions.csv"):
    print("=" * 60)
    print("  Fraud Detector — ML Training Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading & preparing data...")
    X, y, df = load_and_prepare(csv_path)
    print(f"      Samples: {len(X)}  |  Fraud: {y.sum()}  |  Safe: {(y==0).sum()}")

    # 2. Train/test split
    print("\n[2/6] Stratified 80/20 split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. SMOTE
    print("\n[3/6] Applying SMOTE to balance training set...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"      After SMOTE: {len(X_res)} samples  "
          f"({y_res.sum()} fraud / {(y_res==0).sum()} safe)")

    # 4. Train XGBoost
    print("\n[4/6] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   # SMOTE already balanced
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_res, y_res)

    # 5. Evaluate
    print("\n[5/6] Evaluating on held-out test set...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Precision : {prec:.4f}")
    print(f"      Recall    : {rec:.4f}")
    print(f"      F1 Score  : {f1:.4f}")
    print(f"      ROC-AUC   : {roc_auc:.4f}")
    print(f"      Confusion Matrix:\n{cm}")

    # Cross-validation (quick 5-fold on full data)
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5),
                                scoring="roc_auc", n_jobs=-1)
    print(f"      5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    report_str = (
        f"Accuracy : {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall   : {rec:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
        f"ROC-AUC  : {roc_auc:.4f}\n"
        f"CV AUC   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}"
    )
    with open(REPORT_PATH, "w") as f:
        f.write(report_str)

    # 6. SHAP
    print("\n[6/6] Computing SHAP values...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)

    # Save SHAP values + test set for dashboard use
    shap_data = {
        "explainer":    explainer,
        "shap_values":  shap_vals,
        "X_test":       X_test.reset_index(drop=True),
        "y_test":       y_test.reset_index(drop=True),
        "y_pred":       y_pred,
        "y_proba":      y_proba,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   roc_auc,
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std":  float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
        },
    }
    with open(SHAP_PATH, "wb") as f:
        pickle.dump(shap_data, f)

    # 7. Save model & feature list
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(FEAT_PATH, "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    print(f"\n[DONE] Model saved to {MODEL_PATH}")
    print(f"       SHAP data saved to {SHAP_PATH}")
    print(f"       Feature list saved to {FEAT_PATH}")
    print(f"       Report saved to {REPORT_PATH}")
    print("=" * 60)

    return model, shap_data


if __name__ == "__main__":
    train()
