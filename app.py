"""
app.py  —  Fraud Detection AI Portfolio Dashboard
==================================================
3-tab professional Streamlit app:
  Tab 1: EDA & Analytics
  Tab 2: ML Model & Explainability
  Tab 3: Live Detection & AI Report
"""

import io
import os
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI — Portfolio Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-mode custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
  --bg-primary:   #0a0e1a;
  --bg-card:      #111827;
  --bg-card2:     #1a2235;
  --accent:       #6c63ff;
  --accent2:      #00d4aa;
  --accent3:      #ff6b6b;
  --text-primary: #e8eaf0;
  --text-muted:   #8892a4;
  --border:       #1f2d3d;
  --glow:         rgba(108, 99, 255, 0.15);
}

/* Global body */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* Main area */
.main .block-container {
  padding: 1.5rem 2rem 2rem 2rem;
  max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}

/* KPI metric cards */
.metric-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 4px 24px var(--glow);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  margin-bottom: 0.5rem;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 32px var(--glow);
}
.metric-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
}
.metric-delta {
  font-size: 0.8rem;
  margin-top: 0.3rem;
}
.metric-delta.up   { color: #00d4aa; }
.metric-delta.down { color: #ff6b6b; }
.metric-delta.neutral { color: var(--text-muted); }

/* Section headers */
.section-header {
  display: flex; align-items: center; gap: 0.6rem;
  font-size: 1.3rem; font-weight: 600; color: var(--text-primary);
  margin: 1.4rem 0 0.8rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent);
}

/* Tab styling */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg-card) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
  gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--accent) !important;
  color: white !important;
}

/* Risk badge */
.risk-badge {
  display: inline-block; padding: 0.25rem 0.9rem;
  border-radius: 20px; font-size: 0.82rem; font-weight: 600;
}
.risk-critical { background: rgba(255,107,107,0.2); color: #ff6b6b; border: 1px solid #ff6b6b; }
.risk-high     { background: rgba(255,167,38,0.2);  color: #ffa726; border: 1px solid #ffa726; }
.risk-moderate { background: rgba(255,213,79,0.2);  color: #ffd94f; border: 1px solid #ffd94f; }
.risk-low      { background: rgba(0,212,170,0.2);   color: #00d4aa; border: 1px solid #00d4aa; }

/* Report box */
.report-box {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 1.2rem 1.5rem;
  font-size: 0.85rem;
  line-height: 1.7;
  color: var(--text-primary);
  max-height: 500px;
  overflow-y: auto;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* Plotly charts dark background */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Divider */
.styled-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  border: none; margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme defaults ────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.7)",
    font=dict(family="Inter", color="#e8eaf0"),
    margin=dict(l=30, r=30, t=50, b=30),
)
ACCENT_SEQ  = ["#6c63ff", "#00d4aa", "#ff6b6b", "#ffa726", "#4fc3f7", "#ab47bc"]
STATUS_COLORS = {"Safe": "#00d4aa", "Suspicious": "#ff6b6b"}

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH    = "fraud_detection.db"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_names.pkl")
SHAP_PATH  = os.path.join(MODEL_DIR, "shap_values.pkl")

FEATURE_COLS = [
    "Amount", "Hour", "DayOfWeek", "IsWeekend",
    "MerchantRiskScore", "TxFrequency", "AvgUserAmount",
    "AmountDeviation", "IsHighAmount", "IsNightTx",
    "HourSin", "HourCos", "Type_enc", "MerchantCategory_enc",
]

MERCHANT_RISK = {
    "Zomato": 0.01, "Swiggy": 0.01, "Amazon": 0.03, "Flipkart": 0.03,
    "NEFT Transfer": 0.05, "HDFC Bank": 0.02, "BigBazaar": 0.01,
    "Netflix": 0.01, "Uber": 0.02, "Ola": 0.02, "PhonePe": 0.04,
    "Paytm": 0.04, "MakeMyTrip": 0.06, "BookMyShow": 0.02,
    "Reliance Fresh": 0.01, "CryptoXchange": 0.35, "OnlineLottery": 0.40,
    "QuickLoan": 0.30, "Unknown": 0.80, "Hackermarket": 0.95,
    "ShellCorp LLC": 0.90, "FastCashNow": 0.85,
}
MERCHANT_CAT = {
    "Zomato": "Food & Dining", "Swiggy": "Food & Dining",
    "Amazon": "E-Commerce", "Flipkart": "E-Commerce",
    "NEFT Transfer": "Bank Transfer", "HDFC Bank": "Banking",
    "BigBazaar": "Grocery", "Netflix": "Streaming",
    "Uber": "Transport", "Ola": "Transport",
    "PhonePe": "UPI Payment", "Paytm": "UPI Payment",
    "MakeMyTrip": "Travel", "BookMyShow": "Entertainment",
    "Reliance Fresh": "Grocery", "CryptoXchange": "Cryptocurrency",
    "OnlineLottery": "Gambling", "QuickLoan": "Lending",
    "Unknown": "Unclassified", "Hackermarket": "Dark Web",
    "ShellCorp LLC": "Shell Company", "FastCashNow": "Predatory Lending",
}

# ── Data & model loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["Date"]   = pd.to_datetime(df["Date"])
    df["Status"] = df["Suspicious"].map({0: "Safe", 1: "Suspicious"})
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feats = pickle.load(f)
    return model, feats

@st.cache_resource
def load_shap_data():
    if not os.path.exists(SHAP_PATH):
        return None
    with open(SHAP_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    if not os.path.exists(enc_path):
        return None
    with open(enc_path, "rb") as f:
        return pickle.load(f)

# ── Metric card helper ────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = "", delta_type: str = "neutral"):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"" if not delta else f'<div class="metric-delta {delta_type}">{delta}</div>'}
    </div>
    """, unsafe_allow_html=True)

# ── Risk badge helper ─────────────────────────────────────────────────────────
def risk_badge(prob: float) -> str:
    if prob >= 0.80:
        return '<span class="risk-badge risk-critical">CRITICAL</span>'
    elif prob >= 0.55:
        return '<span class="risk-badge risk-high">HIGH</span>'
    elif prob >= 0.30:
        return '<span class="risk-badge risk-moderate">MODERATE</span>'
    else:
        return '<span class="risk-badge risk-low">LOW</span>'

# ── Load everything ───────────────────────────────────────────────────────────
df_full  = load_data()
model, feature_names = load_model()
shap_data = load_shap_data()
encoders  = load_encoders()

db_ok    = not df_full.empty
model_ok = model is not None

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([0.12, 0.88])
with col_logo:
    st.markdown("<div style='font-size:3.5rem; line-height:1; padding-top:0.3rem'>🛡️</div>",
                unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin:0; font-size:2.1rem; font-weight:700; color:#e8eaf0;
               letter-spacing:-0.01em; line-height:1.2;'>
        FraudShield <span style='color:#6c63ff;'>AI</span>
    </h1>
    <p style='margin:0; color:#8892a4; font-size:0.9rem;'>
        Enterprise Fraud Detection & ML Explainability Platform
    </p>
    """, unsafe_allow_html=True)

st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
      <div style='font-size:1.5rem; font-weight:700; color:#6c63ff;'>🛡️ FraudShield</div>
      <div style='font-size:0.75rem; color:#8892a4;'>AI Portfolio v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicators
    st.markdown("**System Status**")
    db_status    = "🟢 Connected" if db_ok    else "🔴 No Database"
    model_status = "🟢 Loaded"    if model_ok else "🟡 Not trained yet"
    st.markdown(f"- Database: {db_status}")
    st.markdown(f"- ML Model: {model_status}")
    st.markdown("---")

    if db_ok:
        st.markdown("**Filters**")
        date_min = df_full["Date"].min().date()
        date_max = df_full["Date"].max().date()
        date_range = st.date_input(
            "Date Range", value=(date_min, date_max),
            min_value=date_min, max_value=date_max
        )

        cats = ["All"] + sorted(df_full["MerchantCategory"].unique().tolist())
        sel_cat = st.multiselect("Merchant Categories", cats[1:], default=[])

        amt_min = float(df_full["Amount"].min())
        amt_max = float(df_full["Amount"].max())
        amt_range = st.slider("Amount Range (₹)", amt_min, amt_max, (amt_min, amt_max),
                              format="₹%.0f")
        st.markdown("---")

    if not model_ok:
        st.warning("Model not found.\nRun `python train_model.py` first.")
    else:
        if st.button("🔄 Re-train Model", use_container_width=True):
            with st.spinner("Training..."):
                import subprocess
                result = subprocess.run(
                    ["python", "train_model.py"],
                    capture_output=True, text=True, cwd="."
                )
            if result.returncode == 0:
                st.success("Model re-trained!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"Training failed:\n{result.stderr[:300]}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8892a4; text-align:center;'>
      Built with Streamlit · XGBoost · SHAP<br>
      Portfolio Demo — Heuristic AI Engine
    </div>
    """, unsafe_allow_html=True)

# ── Apply sidebar filters ─────────────────────────────────────────────────────
if db_ok:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df_full[(df_full["Date"] >= d0) & (df_full["Date"] <= d1)].copy()
    else:
        df = df_full.copy()

    if sel_cat:
        df = df[df["MerchantCategory"].isin(sel_cat)]

    df = df[(df["Amount"] >= amt_range[0]) & (df["Amount"] <= amt_range[1])]
else:
    df = pd.DataFrame()

# ── Setup check ───────────────────────────────────────────────────────────────
if not db_ok:
    st.error("""
    **No database found.** Run the setup pipeline first:
    ```
    python generate_data.py
    python main.py
    python train_model.py
    ```
    """)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  EDA & Analytics",
    "🤖  ML Model & Explainability",
    "🚨  Live Detection & AI Report",
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — EDA & Analytics                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    # KPI Row
    total   = len(df)
    fraud_n = int(df["Suspicious"].sum())
    safe_n  = total - fraud_n
    fraud_p = (fraud_n / total * 100) if total else 0
    avg_amt = df["Amount"].mean() if total else 0
    max_amt = df["Amount"].max() if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Transactions", f"{total:,}", "", "neutral")
    with c2: metric_card("Fraud Flagged", f"{fraud_n:,}", f"+{fraud_p:.1f}% rate", "down")
    with c3: metric_card("Safe Transactions", f"{safe_n:,}", "", "up")
    with c4: metric_card("Avg Transaction", f"₹{avg_amt:,.0f}", "", "neutral")
    with c5: metric_card("Max Transaction", f"₹{max_amt:,.0f}", "", "neutral")

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    # Row 1 — Distribution + Fraud by Category
    c_left, c_right = st.columns([0.55, 0.45])

    with c_left:
        st.markdown("<div class='section-header'>📈 Transaction Amount Distribution</div>",
                    unsafe_allow_html=True)
        fig = px.histogram(
            df, x="Amount", color="Status", nbins=60, barmode="overlay",
            color_discrete_map=STATUS_COLORS, opacity=0.8,
            labels={"Amount": "Amount (₹)", "count": "Count"}
        )
fig.update_layout(**PLOTLY_DARK)
fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1))
fig.update_traces(marker_line_width=0)
st.plotly_chart(fig, width="stretch")

with c_right:
        st.markdown("<div class='section-header'>🏪 Fraud Rate by Merchant Category</div>",
                    unsafe_allow_html=True)
        cat_stats = (
            df.groupby("MerchantCategory")
              .agg(Total=("Suspicious", "count"), Fraud=("Suspicious", "sum"))
              .assign(FraudRate=lambda x: x["Fraud"] / x["Total"] * 100)
              .sort_values("FraudRate", ascending=True)
              .reset_index()
        )
        fig2 = px.bar(
        cat_stats, x="FraudRate", y="MerchantCategory", orientation="h",
        color="FraudRate", color_continuous_scale="RdYlGn_r",
        labels={"FraudRate": "Fraud Rate (%)", "MerchantCategory": ""}
    )
fig2.update_layout(**PLOTLY_DARK)
fig2.update_layout(coloraxis_showscale=False, height=380)
st.plotly_chart(fig2, width="stretch")

    # Row 2 — Time heatmap + Merchant Risk Scatter
c_left2, c_right2 = st.columns([0.5, 0.5])

with c_left2:
        st.markdown("<div class='section-header'>🕒 Fraud Heatmap by Hour & Day</div>",
                    unsafe_allow_html=True)
        day_labels  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heatmap_df  = (
            df.groupby(["DayOfWeek", "Hour"])["Suspicious"]
              .mean().reset_index()
        )
        pivot = heatmap_df.pivot(index="DayOfWeek", columns="Hour", values="Suspicious").fillna(0)
        pivot.index = [day_labels[i] for i in pivot.index]
        fig3 = px.imshow(
        pivot, color_continuous_scale="RdPu", aspect="auto",
        labels={"x": "Hour of Day", "y": "Day of Week", "color": "Fraud Rate"}
    )
fig3.update_layout(**PLOTLY_DARK)
fig3.update_layout(height=310, coloraxis_colorbar=dict(title="Fraud Rate"))
st.plotly_chart(fig3, width="stretch")

with c_right2:
        st.markdown("<div class='section-header'>💹 Amount vs. Merchant Risk Score</div>",
                    unsafe_allow_html=True)
        fig4 = px.scatter(
        df.sample(min(500, len(df)), random_state=42),
        x="MerchantRiskScore", y="Amount", color="Status",
        color_discrete_map=STATUS_COLORS, opacity=0.7,
        size_max=10, hover_data=["Merchant", "TransactionID"],
        labels={"MerchantRiskScore": "Merchant Risk Score",
                "Amount": "Amount (₹)"}
    )
fig4.update_layout(**PLOTLY_DARK)
fig4.update_layout(
        height=310,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
st.plotly_chart(fig4, width="stretch")

    # Row 3 — Time series + User frequency
st.markdown("<div class='section-header'>📅 Transaction Volume Over Time</div>",
                unsafe_allow_html=True)
ts_df = (
        df.assign(Week=df["Date"].dt.to_period("W").astype(str))
          .groupby(["Week", "Status"])["Amount"]
          .agg(Count="count", TotalAmount="sum")
          .reset_index()
    )
fig5 = px.line(
        ts_df, x="Week", y="Count", color="Status",
        color_discrete_map=STATUS_COLORS, markers=True,
        labels={"Week": "Week", "Count": "Transaction Count"}
    )
fig5.update_layout(**PLOTLY_DARK)
st.plotly_chart(fig5, width="stretch")
fig5.update_traces(line_width=2)
st.plotly_chart(fig5, use_container_width=True)

    # Row 4 — Correlation matrix + Summary stats
c_corr, c_stats = st.columns([0.55, 0.45])
with c_corr:
        st.markdown("<div class='section-header'>🔗 Feature Correlation Matrix</div>",
                    unsafe_allow_html=True)
        num_cols = ["Amount", "MerchantRiskScore", "AmountDeviation",
                    "TxFrequency", "Hour", "IsWeekend", "Suspicious"]
        corr = df[num_cols].corr()
        fig6 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            **PLOTLY_DARK
        )
        fig6.update_layout(height=380, coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig6, use_container_width=True)

with c_stats:
        st.markdown("<div class='section-header'>📋 Summary Statistics</div>",
                    unsafe_allow_html=True)
        summary = df[["Amount", "MerchantRiskScore", "TxFrequency",
                       "AmountDeviation"]].describe().T.round(2)
        summary.index.name = "Feature"
        st.dataframe(summary, use_container_width=True, height=360)

    # Data table
st.markdown("<div class='section-header'>🗃️ Transaction Records</div>",
                unsafe_allow_html=True)
status_filter = st.selectbox("Filter by status",
                                 ["All", "Suspicious Only", "Safe Only"])
disp = df.copy()
if status_filter == "Suspicious Only":
        disp = df[df["Suspicious"] == 1]
elif status_filter == "Safe Only":
        disp = df[df["Suspicious"] == 0]

show_cols = ["TransactionID", "Date", "Amount", "Type", "Merchant",
                 "MerchantCategory", "UserID", "Status"]
st.dataframe(disp[show_cols].reset_index(drop=True),
                 use_container_width=True, height=350)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — ML Model & Explainability                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    if not model_ok or shap_data is None:
        st.warning("""
        **ML model not found.** Train the model first:
        ```powershell
        python train_model.py
        ```
        Then refresh this page.
        """)
        st.stop()

    metrics = shap_data["metrics"]
    X_test  = shap_data["X_test"]
    y_test  = shap_data["y_test"]
    y_pred  = shap_data["y_pred"]
    y_proba = shap_data["y_proba"]
    sv      = shap_data["shap_values"]

    # KPI cards — model metrics
    st.markdown("<div class='section-header'>🎯 Model Performance Metrics</div>",
                unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: metric_card("Accuracy",  f"{metrics['accuracy']*100:.1f}%",  "", "up")
    with m2: metric_card("Precision", f"{metrics['precision']*100:.1f}%", "", "up")
    with m3: metric_card("Recall",    f"{metrics['recall']*100:.1f}%",    "", "up")
    with m4: metric_card("F1 Score",  f"{metrics['f1']*100:.1f}%",        "", "up")
    with m5: metric_card("ROC-AUC",   f"{metrics['roc_auc']:.4f}",
                         f"CV: {metrics['cv_auc_mean']:.3f} ±{metrics['cv_auc_std']:.3f}", "up")

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    # Row 1 — Confusion Matrix + ROC Curve
    c_cm, c_roc = st.columns(2)

    with c_cm:
        st.markdown("<div class='section-header'>🔲 Confusion Matrix</div>",
                    unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        labels = ["Safe (0)", "Fraud (1)"]
        fig_cm = px.imshow(
            cm, x=labels, y=labels, text_auto=True,
            color_continuous_scale=[[0, "#111827"], [0.5, "#6c63ff"], [1, "#ff6b6b"]],
            labels={"x": "Predicted", "y": "Actual"},
            **PLOTLY_DARK
        )
        fig_cm.update_layout(height=360, coloraxis_showscale=False,
                              xaxis_title="Predicted Label",
                              yaxis_title="True Label")
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_roc:
        st.markdown("<div class='section-header'>📉 ROC Curve</div>",
                    unsafe_allow_html=True)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure(**{k: v for k, v in PLOTLY_DARK.items()
                               if k != "template"})
        fig_roc.update_layout(template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(17,24,39,0.7)",
                               font=dict(family="Inter", color="#e8eaf0"),
                               margin=dict(l=30, r=30, t=50, b=30))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={metrics['roc_auc']:.3f})",
            line=dict(color="#6c63ff", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#8892a4", dash="dash", width=1.5)
        ))
        fig_roc.update_layout(
            height=360, xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    # Row 2 — Global SHAP + per-transaction waterfall
    c_shap, c_water = st.columns([0.5, 0.5])

    with c_shap:
        st.markdown("<div class='section-header'>🌐 Global Feature Importance (SHAP)</div>",
                    unsafe_allow_html=True)
        mean_abs_shap = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": np.abs(sv).mean(axis=0),
        }).sort_values("Importance", ascending=True)

        fig_shap = px.bar(
            mean_abs_shap, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Purples",
            labels={"Importance": "Mean |SHAP|", "Feature": ""},
            **PLOTLY_DARK
        )
        fig_shap.update_layout(height=400, coloraxis_showscale=False)
        fig_shap.update_traces(marker_line_width=0)
        st.plotly_chart(fig_shap, use_container_width=True)

    with c_water:
        st.markdown("<div class='section-header'>🔍 Per-Transaction SHAP Waterfall</div>",
                    unsafe_allow_html=True)

        # Pick from test set fraud transactions
        fraud_idx = np.where(y_test == 1)[0]
        if len(fraud_idx) == 0:
            st.info("No fraud transactions in test set.")
        else:
            sel_idx = st.selectbox(
                "Select test-set transaction",
                options=list(range(len(fraud_idx))),
                format_func=lambda i: f"Fraud Tx #{fraud_idx[i]+1} — "
                                      f"Prob: {y_proba[fraud_idx[i]]:.1%}"
            )
            chosen = fraud_idx[sel_idx]
            shap_row = sv[chosen]
            base_val = float(np.mean(y_proba))

            # Build waterfall data
            order      = np.argsort(np.abs(shap_row))[::-1][:10]
            feat_names = [FEATURE_COLS[j] for j in order]
            feat_shaps = [shap_row[j] for j in order]

            fig_wf = go.Figure(go.Waterfall(
                orientation="h",
                y=feat_names[::-1],
                x=feat_shaps[::-1],
                base=base_val,
                connector={"mode": "between",
                            "line": {"width": 1, "color": "#2d3748", "dash": "solid"}},
                increasing={"marker": {"color": "#ff6b6b"}},
                decreasing={"marker": {"color": "#00d4aa"}},
                totals={"marker": {"color": "#6c63ff"}},
            ))
            fig_wf.update_layout(
                height=400, showlegend=False,
                xaxis_title="SHAP Value Impact",
                **{k: v for k, v in PLOTLY_DARK.items() if k != "template"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    # SHAP beeswarm summary (using bar as proxy — shap.plots require matplotlib)
    st.markdown("<div class='section-header'>📊 Feature Impact Distribution</div>",
                unsafe_allow_html=True)
    # Violin-style per-feature distribution of SHAP values
    shap_df = pd.DataFrame(sv, columns=FEATURE_COLS)
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(8).index.tolist()
    shap_long = shap_df[top_features].melt(var_name="Feature", value_name="SHAP Value")

    fig_box = px.box(
        shap_long, x="Feature", y="SHAP Value",
        color="Feature", color_discrete_sequence=ACCENT_SEQ,
        labels={"SHAP Value": "SHAP Value", "Feature": ""},
        **PLOTLY_DARK
    )
    fig_box.update_layout(height=350, showlegend=False)
    fig_box.add_hline(y=0, line_dash="dash", line_color="#8892a4", line_width=1)
    st.plotly_chart(fig_box, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — Live Detection & AI Report                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    if not model_ok:
        st.warning("Train the model first: `python train_model.py`")
        st.stop()

    # Import report engine
    try:
        from ai_summary import generate_report, generate_pdf_report
        ai_ok = True
    except ImportError:
        ai_ok = False

    sub1, sub2 = st.tabs(["🔢 Manual Input", "📁 Batch CSV Upload"])

    # ── Manual Input ──────────────────────────────────────────────────────────
    with sub1:
        st.markdown("<div class='section-header'>Enter Transaction Details</div>",
                    unsafe_allow_html=True)

        form_c1, form_c2, form_c3 = st.columns(3)
        with form_c1:
            inp_amount   = st.number_input("Amount (₹)", min_value=1.0,
                                           max_value=500000.0, value=5000.0, step=100.0)
            inp_merchant = st.selectbox("Merchant", list(MERCHANT_RISK.keys()))
            inp_type     = st.selectbox("Transaction Type",
                                        ["Debit", "Credit", "UPI", "NEFT", "IMPS"])
        with form_c2:
            inp_hour      = st.slider("Hour of Day", 0, 23, 14)
            inp_dow       = st.selectbox("Day of Week",
                                         ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            inp_user      = st.text_input("User ID (optional)", value="USR0001")
        with form_c3:
            inp_avg_amt   = st.number_input("User Avg Amount (₹)", 0.0, 500000.0, 8000.0, 500.0)
            inp_tx_freq   = st.number_input("User Tx Frequency", 1, 200, 15)
            inp_std_amt   = st.number_input("User Std Dev Amount", 0.0, 200000.0, 3000.0, 100.0)

        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            # Build feature vector
            dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
                       "Fri": 4, "Sat": 5, "Sun": 6}
            type_map = {"Debit": 0, "Credit": 1, "IMPS": 2, "NEFT": 3, "UPI": 4}
            cat_list = [
                "Bank Transfer", "Banking", "Cryptocurrency", "Dark Web",
                "E-Commerce", "Entertainment", "Food & Dining", "Gambling",
                "Grocery", "Lending", "Predatory Lending", "Shell Company",
                "Streaming", "Transport", "UPI Payment", "Travel", "Unclassified",
            ]
            merchant_cat = MERCHANT_CAT.get(inp_merchant, "Unclassified")
            cat_enc = cat_list.index(merchant_cat) if merchant_cat in cat_list else 0

            dow_val    = dow_map.get(inp_dow, 0)
            is_weekend = int(dow_val >= 5)
            is_night   = int(inp_hour < 6 or inp_hour >= 22)
            risk_score = MERCHANT_RISK.get(inp_merchant, 0.5)
            amt_dev    = (inp_amount - inp_avg_amt) / (inp_std_amt + 1)
            is_high    = int(inp_amount > 30000)
            hr_sin     = float(np.sin(2 * np.pi * inp_hour / 24))
            hr_cos     = float(np.cos(2 * np.pi * inp_hour / 24))
            type_enc   = type_map.get(inp_type, 0)

            feat_vec = pd.DataFrame([[
                inp_amount, inp_hour, dow_val, is_weekend,
                risk_score, inp_tx_freq, inp_avg_amt,
                amt_dev, is_high, is_night, hr_sin, hr_cos,
                type_enc, cat_enc,
            ]], columns=FEATURE_COLS)

            prob = float(model.predict_proba(feat_vec)[0, 1])

            # SHAP for this transaction
            from ai_summary import generate_report, generate_pdf_report
            explainer = shap_data["explainer"]
            sv_single = explainer.shap_values(feat_vec)[0]
            shap_contrib = dict(zip(FEATURE_COLS, sv_single))

            # ── Result display ────────────────────────────────────────────
            st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

            res_c1, res_c2 = st.columns([0.35, 0.65])

            with res_c1:
                st.markdown("<div class='section-header'>Risk Assessment</div>",
                            unsafe_allow_html=True)
                badge_html = risk_badge(prob)
                st.markdown(f"""
                <div class="metric-card" style="text-align:center;">
                  <div class="metric-label">Fraud Probability</div>
                  <div class="metric-value">{prob*100:.1f}%</div>
                  <div style="margin-top:0.6rem;">{badge_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 28, "color": "#e8eaf0"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1,
                                 "tickcolor": "#8892a4"},
                        "bar": {"color": "#6c63ff"},
                        "bgcolor": "#111827",
                        "borderwidth": 2,
                        "bordercolor": "#1f2d3d",
                        "steps": [
                            {"range": [0, 30],   "color": "rgba(0,212,170,0.15)"},
                            {"range": [30, 55],  "color": "rgba(255,213,79,0.15)"},
                            {"range": [55, 80],  "color": "rgba(255,167,38,0.15)"},
                            {"range": [80, 100], "color": "rgba(255,107,107,0.15)"},
                        ],
                        "threshold": {"line": {"color": "#ff6b6b", "width": 3},
                                      "thickness": 0.8, "value": 55},
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                ))
                fig_gauge.update_layout(
                    height=220, margin=dict(l=20, r=20, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e8eaf0"),
                    template="plotly_dark",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Top 3 SHAP reasons
                st.markdown("**Top Risk Drivers**")
                top3 = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                for feat, val in top3:
                    color = "#ff6b6b" if val > 0 else "#00d4aa"
                    arrow = "↑" if val > 0 else "↓"
                    st.markdown(
                        f'<span style="color:{color}; font-size:0.88rem;">'
                        f'{arrow} <b>{feat}</b>: {val:+.3f}</span>',
                        unsafe_allow_html=True
                    )

            with res_c2:
                st.markdown("<div class='section-header'>SHAP Explanation</div>",
                            unsafe_allow_html=True)
                shap_series = pd.Series(shap_contrib).sort_values(key=lambda x: x.abs(),
                                                                    ascending=False).head(10)
                colors = ["#ff6b6b" if v > 0 else "#00d4aa" for v in shap_series.values]
                fig_exp = go.Figure(go.Bar(
                    x=shap_series.values[::-1],
                    y=shap_series.index[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    marker_line_width=0,
                ))
                fig_exp.update_layout(
                    height=320, xaxis_title="SHAP Value",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.7)",
                    font=dict(family="Inter", color="#e8eaf0"),
                    margin=dict(l=10, r=20, t=10, b=30),
                )
                fig_exp.add_vline(x=0, line_width=1, line_color="#8892a4",
                                   line_dash="dash")
                st.plotly_chart(fig_exp, use_container_width=True)

            # AI Analyst Report
            st.markdown("<div class='section-header'>📝 AI Analyst Report</div>",
                        unsafe_allow_html=True)
            tx_row = {
                "TransactionID": f"TXN-LIVE-{inp_merchant[:3].upper()}",
                "Date":          f"2025-XX-XX {inp_hour:02d}:00",
                "Amount":        inp_amount,
                "Type":          inp_type,
                "Merchant":      inp_merchant,
                "MerchantCategory": merchant_cat,
                "MerchantRiskScore": risk_score,
                "UserID":        inp_user,
                "Hour":          inp_hour,
            }
            report = generate_report(tx_row, prob, shap_contrib)
            st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)

            # PDF download
            pdf_bytes = generate_pdf_report(report, tx_row["TransactionID"])
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"fraud_report_{tx_row['TransactionID']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ── Batch CSV Upload ──────────────────────────────────────────────────────
    with sub2:
        st.markdown("<div class='section-header'>Upload Batch Transactions CSV</div>",
                    unsafe_allow_html=True)
        st.info("""
        **CSV must have these columns:**
        `Amount`, `Type`, `Merchant`, `Hour`, `DayOfWeek`
        _(UserID, AvgUserAmount, TxFrequency, StdUserAmount are optional — defaults used if missing)_
        """)

        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_df)} rows.")

            # Validate required columns
            required = {"Amount", "Type", "Merchant", "Hour", "DayOfWeek"}
            missing  = required - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Build feature matrix for all rows
                def build_features(row):
                    type_map = {"Debit": 0, "Credit": 1, "IMPS": 2, "NEFT": 3, "UPI": 4}
                    cat_list = [
                        "Bank Transfer", "Banking", "Cryptocurrency", "Dark Web",
                        "E-Commerce", "Entertainment", "Food & Dining", "Gambling",
                        "Grocery", "Lending", "Predatory Lending", "Shell Company",
                        "Streaming", "Transport", "UPI Payment", "Travel", "Unclassified",
                    ]
                    merchant    = str(row.get("Merchant", "Unknown"))
                    merchant_cat = MERCHANT_CAT.get(merchant, "Unclassified")
                    cat_enc     = cat_list.index(merchant_cat) if merchant_cat in cat_list else 0
                    risk_score  = MERCHANT_RISK.get(merchant, 0.5)
                    hour        = int(row.get("Hour", 12))
                    dow         = int(row.get("DayOfWeek", 0))
                    is_weekend  = int(dow >= 5)
                    is_night    = int(hour < 6 or hour >= 22)
                    amount      = float(row.get("Amount", 1000))
                    avg_amt     = float(row.get("AvgUserAmount", 8000))
                    std_amt     = float(row.get("StdUserAmount", 3000))
                    tx_freq     = int(row.get("TxFrequency", 15))
                    amt_dev     = (amount - avg_amt) / (std_amt + 1)
                    is_high     = int(amount > 30000)
                    hr_sin      = float(np.sin(2 * np.pi * hour / 24))
                    hr_cos      = float(np.cos(2 * np.pi * hour / 24))
                    type_enc    = type_map.get(str(row.get("Type", "Debit")), 0)
                    return [
                        amount, hour, dow, is_weekend, risk_score, tx_freq,
                        avg_amt, amt_dev, is_high, is_night, hr_sin, hr_cos,
                        type_enc, cat_enc,
                    ]

                feat_matrix = pd.DataFrame(
                    [build_features(r) for _, r in batch_df.iterrows()],
                    columns=FEATURE_COLS
                )
                probs = model.predict_proba(feat_matrix)[:, 1]
                preds = (probs >= 0.5).astype(int)

                batch_df["FraudProbability"] = (probs * 100).round(2)
                batch_df["Prediction"]       = ["Fraud" if p else "Safe" for p in preds]

                # Summary
                b1, b2, b3 = st.columns(3)
                with b1: metric_card("Total Rows", f"{len(batch_df)}", "", "neutral")
                with b2: metric_card("Fraud Detected", f"{preds.sum()}", "", "down")
                with b3: metric_card("Avg Risk Score",
                                     f"{probs.mean()*100:.1f}%", "", "neutral")

                # Color-coded table
                st.dataframe(
                    batch_df.sort_values("FraudProbability", ascending=False),
                    use_container_width=True, height=400,
                )

                # Distribution chart
                fig_batch = px.histogram(
                    batch_df, x="FraudProbability", color="Prediction",
                    color_discrete_map={"Fraud": "#ff6b6b", "Safe": "#00d4aa"},
                    nbins=30, barmode="overlay",
                    labels={"FraudProbability": "Fraud Probability (%)"},
                    **PLOTLY_DARK
                )
                st.plotly_chart(fig_batch, use_container_width=True)

                # Download results
                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )