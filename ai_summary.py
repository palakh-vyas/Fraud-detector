"""
ai_summary.py
-------------
Heuristic Fraud Analyst Report Engine.

Generates a professional, structured analyst memo for any transaction
using SHAP feature contributions and rule-based risk logic — no API key needed.

Public API:
    generate_report(tx_row: dict, risk_prob: float, shap_contrib: dict) -> str
    generate_pdf_report(report_text: str, tx_id: str) -> bytes
"""

from __future__ import annotations
import io
from datetime import datetime
from fpdf import FPDF


# ── Risk-level helpers ────────────────────────────────────────────────────────

def _risk_level(prob: float) -> tuple[str, str]:
    if prob >= 0.80:
        return "CRITICAL", "red"
    elif prob >= 0.55:
        return "HIGH", "orange"
    elif prob >= 0.30:
        return "MODERATE", "yellow"
    else:
        return "LOW", "green"


def _risk_label(prob: float) -> str:
    level, _ = _risk_level(prob)
    return level


# ── Feature-to-human-readable explanations ───────────────────────────────────

_FEATURE_EXPLAIN = {
    "Amount":              ("Transaction amount", "unusually high transaction amount", "normal transaction amount"),
    "MerchantRiskScore":   ("Merchant risk score", "high-risk or unverified merchant", "trusted merchant"),
    "AmountDeviation":     ("Amount vs user average", "amount significantly deviates from user's typical spending", "amount consistent with user's spending history"),
    "IsHighAmount":        ("High-value flag", "transaction exceeds high-value threshold (>30,000)", "below high-value threshold"),
    "IsNightTx":           ("Off-hours flag", "transaction made during unusual hours (midnight–5am or 10pm+)", "transaction during normal business hours"),
    "TxFrequency":         ("User activity frequency", "abnormally high transaction frequency for this user", "normal user transaction frequency"),
    "MerchantCategory_enc":("Merchant category", "high-risk merchant category detected", "low-risk merchant category"),
    "HourSin":             ("Time of day (sin)", "suspicious time pattern", "normal time pattern"),
    "HourCos":             ("Time of day (cos)", "suspicious time pattern", "normal time pattern"),
    "DayOfWeek":           ("Day of week", "suspicious day-of-week pattern", "typical transaction day"),
    "IsWeekend":           ("Weekend flag", "weekend transaction (higher fraud risk)", "weekday transaction"),
    "AvgUserAmount":       ("User average spend", "inconsistent with user's average spending", "consistent with user's average spending"),
    "Type_enc":            ("Transaction type", "unusual transaction type for this account", "typical transaction type"),
    "Hour":                ("Transaction hour", "transaction at unusual hour", "transaction at normal hour"),
}


def _explain_feature(feat: str, shap_val: float) -> str:
    info = _FEATURE_EXPLAIN.get(feat, (feat, "anomalous value detected", "normal value"))
    label, high_msg, low_msg = info
    direction = high_msg if shap_val > 0 else low_msg
    impact = "strongly " if abs(shap_val) > 0.3 else ("moderately " if abs(shap_val) > 0.1 else "slightly ")
    return f"  - **{label}**: {impact}{direction} (SHAP impact: {shap_val:+.3f})"


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    tx_row: dict,
    risk_prob: float,
    shap_contrib: dict,          # {feature_name: shap_value}
    top_n: int = 5,
) -> str:
    """
    Generate a professional fraud analyst report as a markdown string.

    Args:
        tx_row       : dict of transaction fields (from DataFrame row)
        risk_prob    : model output probability [0,1]
        shap_contrib : dict mapping feature name -> SHAP value for this tx
        top_n        : number of top contributing features to highlight
    """
    now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk       = _risk_label(risk_prob)
    tx_id      = tx_row.get("TransactionID", "N/A")
    merchant   = tx_row.get("Merchant",      "Unknown")
    category   = tx_row.get("MerchantCategory", "N/A")
    amount     = tx_row.get("Amount",         0.0)
    tx_type    = tx_row.get("Type",           "N/A")
    user_id    = tx_row.get("UserID",         "N/A")
    tx_date    = tx_row.get("Date",           "N/A")
    hour       = int(tx_row.get("Hour",        0))
    merchant_risk = float(tx_row.get("MerchantRiskScore", 0))

    # Sort SHAP contributions by absolute magnitude
    sorted_contribs = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    top_factors     = sorted_contribs[:top_n]
    top_positive    = [(f, v) for f, v in sorted_contribs if v > 0][:3]
    top_negative    = [(f, v) for f, v in sorted_contribs if v < 0][:2]

    # Recommendation
    if risk_prob >= 0.80:
        action  = "BLOCK IMMEDIATELY"
        summary = (
            "This transaction exhibits multiple high-confidence fraud indicators. "
            "Immediate blocking and account review are strongly recommended. "
            "The combination of merchant risk profile, transaction amount deviation, "
            "and temporal anomalies presents a critical risk scenario."
        )
    elif risk_prob >= 0.55:
        action  = "FLAG FOR REVIEW"
        summary = (
            "This transaction has triggered several fraud risk signals above threshold. "
            "Manual review by a senior analyst is recommended before allowing the "
            "transaction to proceed. Consider contacting the account holder for verification."
        )
    elif risk_prob >= 0.30:
        action  = "MONITOR CLOSELY"
        summary = (
            "This transaction shows some indicators of elevated risk but does not "
            "conclusively meet fraud threshold criteria. Enhanced monitoring is advised "
            "for subsequent transactions from this user and merchant combination."
        )
    else:
        action  = "APPROVE"
        summary = (
            "This transaction presents low fraud risk. Features are consistent with "
            "the user's established spending patterns and the merchant has a verified, "
            "low-risk profile. No immediate action required."
        )

    time_str = f"{hour:02d}:00"
    time_note = "off-hours" if (hour < 6 or hour >= 22) else "business hours"

    lines = [
        "# FRAUD ANALYST REPORT",
        f"**Generated:** {now}  |  **Analyst Engine:** Heuristic AI v2.0",
        "",
        "---",
        "",
        "## 1. Transaction Summary",
        "",
        f"| Field              | Value                            |",
        f"|:-------------------|:---------------------------------|",
        f"| Transaction ID     | `{tx_id}`                        |",
        f"| Date & Time        | {tx_date} ({time_str}, {time_note}) |",
        f"| Amount             | ₹{amount:,.2f}                   |",
        f"| Type               | {tx_type}                        |",
        f"| Merchant           | {merchant}                       |",
        f"| Merchant Category  | {category}                       |",
        f"| Merchant Risk Score| {merchant_risk:.2f} / 1.00       |",
        f"| User ID            | {user_id}                        |",
        "",
        "---",
        "",
        "## 2. Risk Assessment",
        "",
        f"| Risk Level | Fraud Probability | Recommended Action |",
        f"|:----------:|:-----------------:|:------------------:|",
        f"| **{risk}** | **{risk_prob*100:.1f}%** | **{action}** |",
        "",
        f"**Analyst Summary:**",
        f"> {summary}",
        "",
        "---",
        "",
        "## 3. Key Risk Drivers (SHAP Analysis)",
        "",
        "_The following features most strongly influenced this fraud prediction:_",
        "",
    ]

    # Top fraud-increasing factors
    if top_positive:
        lines.append("### Factors Increasing Fraud Risk")
        for feat, val in top_positive:
            lines.append(_explain_feature(feat, val))
        lines.append("")

    # Top fraud-decreasing factors
    if top_negative:
        lines.append("### Factors Reducing Fraud Risk")
        for feat, val in top_negative:
            lines.append(_explain_feature(feat, val))
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Detailed Feature Contributions",
        "",
        "| Feature | SHAP Value | Direction |",
        "|:--------|:----------:|:---------:|",
    ]
    for feat, val in sorted_contribs:
        direction = "Fraud ↑" if val > 0 else "Fraud ↓"
        lines.append(f"| {feat} | {val:+.4f} | {direction} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Audit Trail",
        "",
        f"- Report generated at: {now}",
        f"- Model: XGBoost Classifier (SMOTE-balanced, 2000-tx training set)",
        f"- Explainability: SHAP TreeExplainer",
        f"- Engine version: Heuristic Analyst v2.0",
        f"- Disclaimer: This report is generated by an automated fraud detection",
        f"  system and should be used as a decision-support tool. Final decisions",
        f"  should involve human review for high-risk transactions.",
    ]

    return "\n".join(lines)


# ── PDF Export ────────────────────────────────────────────────────────────────

def generate_pdf_report(report_text: str, tx_id: str = "TXN") -> bytes:
    """Convert a markdown report string to a PDF (bytes) for download."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, "FRAUD ANALYST REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Transaction: {tx_id}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(108, 99, 255)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Body — strip markdown and write as plain text
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    for line in report_text.splitlines():
        # Clean markdown markers
        clean = (
            line.replace("**", "").replace("##", "").replace("#", "")
                .replace("`", "").replace("|", " | ").strip()
        )
        if not clean or clean == "---":
            pdf.ln(3)
            continue
        if line.startswith("## ") or line.startswith("# "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 50, 160)
            pdf.multi_cell(0, 6, clean)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)
        else:
            try:
                pdf.multi_cell(0, 5, clean)
            except Exception:
                pdf.multi_cell(0, 5, clean.encode("latin-1", "replace").decode("latin-1"))

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
