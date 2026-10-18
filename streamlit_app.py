"""
DSA 3900 – Patricia Kiarie (669781) | USIU-Africa
African Stock Market Prediction Dashboard
Streamlit Prototype — deploy free on streamlit.io/cloud
Run locally: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AfriStockAI — Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#C9A84C; margin-bottom:0; }
    .subtitle   { font-size:1rem; color:#a0aec0; margin-bottom:1.5rem; }
    .metric-card {
        background:#1e2a3a; border-radius:10px; padding:1.2rem 1rem;
        border-left:4px solid #C9A84C; margin-bottom:0.5rem;
    }
    .pred-up   { font-size:2rem; font-weight:700; color:#34d399; }
    .pred-down { font-size:2rem; font-weight:700; color:#f87171; }
    .badge-up   { background:#065f46; color:#6ee7b7; border-radius:20px;
                  padding:2px 14px; font-size:0.85rem; font-weight:600; }
    .badge-down { background:#7f1d1d; color:#fca5a5; border-radius:20px;
                  padding:2px 14px; font-size:0.85rem; font-weight:600; }
    .section-header { font-size:1.15rem; font-weight:600; color:#C9A84C;
                      border-bottom:2px solid #C9A84C; padding-bottom:4px;
                      margin:1rem 0 0.6rem 0; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0D1B3E/C9A84C?text=AfriStockAI",
             width=200)
    st.markdown("### ⚙️ Dashboard Settings")

    STOCKS = {
        "MTN Nigeria — NGX  (MTNN.LG)": "MTNN.LG",
        "Naspers — JSE  (NPN.JO)":       "NPN.JO",
    }
    stock_label = st.selectbox("Select Stock", list(STOCKS.keys()))
    ticker      = STOCKS[stock_label]

    MODELS = {
        "Random Forest – Hybrid ✦ Best": "RF_Hybrid",
        "XGBoost – Hybrid":              "XGB_Hybrid",
        "LSTM – Hybrid":                 "LSTM_Hybrid",
        "Random Forest – Baseline":      "RF_Baseline",
        "XGBoost – Baseline":            "XGB_Baseline",
        "LSTM – Baseline":               "LSTM_Baseline",
    }
    model_label = st.selectbox("Select Model", list(MODELS.keys()))
    model_key   = MODELS[model_label]

    st.markdown("---")
    data_source = st.radio("Data Mode", ["Live (yfinance)", "Pilot Simulation"])
    st.markdown("---")
    st.markdown("**About**  \nDSA 3900 Project · USIU-Africa  \nPatricia Kiarie — ID 669781")

# ── Pilot simulation results ──────────────────────────────────────
METRICS = {
    "RF_Baseline":   {"accuracy":0.4574, "precision":0.4283, "recall":0.7236, "f1":0.5370},
    "RF_Hybrid":     {"accuracy":0.7660, "precision":0.6791, "recall":0.8812, "f1":0.7658},
    "XGB_Baseline":  {"accuracy":0.4255, "precision":0.4126, "recall":0.6930, "f1":0.5140},
    "XGB_Hybrid":    {"accuracy":0.7660, "precision":0.6932, "recall":0.8318, "f1":0.7552},
    "LSTM_Baseline": {"accuracy":0.5106, "precision":0.4608, "recall":0.6941, "f1":0.5518},
    "LSTM_Hybrid":   {"accuracy":0.5000, "precision":0.4530, "recall":0.7141, "f1":0.5526},
}

# Simulate a prediction (deterministic, not random, for demo)
STOCK_PREDS = {
    "MTNN.LG": {"direction": "UP",   "confidence": 0.78, "price_est": 207.40,
                "prev_close": 201.25, "change": +3.06},
    "NPN.JO":  {"direction": "DOWN", "confidence": 0.72, "price_est": 3468.10,
                "prev_close": 3512.00, "change": -1.25},
}
pred = STOCK_PREDS[ticker]
met  = METRICS[model_key]

# ── Header ────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 AfriStockAI — Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hybrid ML · Technical Indicators + News Sentiment · NGX & JSE</p>',
            unsafe_allow_html=True)

# ── Row 1: Prediction + Metrics ───────────────────────────────────
col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])

with col1:
    st.markdown('<div class="section-header">🎯 Next-Day Prediction</div>', unsafe_allow_html=True)
    direction = pred["direction"]
    arrow = "▲" if direction == "UP" else "▼"
    css   = "pred-up" if direction == "UP" else "pred-down"
    badge = "badge-up" if direction == "UP" else "badge-down"
    st.markdown(f'<p class="{css}">{arrow} {direction}</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="{badge}">Confidence: {pred["confidence"]*100:.0f}%</span>',
                unsafe_allow_html=True)
    st.markdown(f"""
    | | |
    |---|---|
    | **Ticker** | `{ticker}` |
    | **Prev. Close** | {pred['prev_close']:.2f} |
    | **Est. Move** | {pred['change']:+.2f}% |
    | **Model** | {model_label.split('–')[0].strip()} |
    """)

for label, val in zip(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        [met["accuracy"], met["precision"], met["recall"], met["f1"]]):
    with [col2, col3, col4, col5][["Accuracy","Precision","Recall","F1-Score"].index(label)]:
        st.metric(label=label, value=f"{val:.4f}")

st.divider()

# ── Row 2: Price chart + SHAP ─────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="section-header">📊 Simulated Price History — {}</div>'.format(ticker),
                unsafe_allow_html=True)

    # Generate reproducible fake price series
    rng = np.random.default_rng(42 if ticker == "MTNN.LG" else 99)
    n   = 60
    sp  = 200.0 if ticker == "MTNN.LG" else 3500.0
    ret = rng.normal(0.0003, 0.02, n)
    closes = sp * np.cumprod(1 + ret)
    dates  = pd.bdate_range(end=datetime.today(), periods=n)

    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(dates, closes, color="#93c5fd", linewidth=1.4)
    # 5-day & 20-day MA
    ma5  = pd.Series(closes).rolling(5).mean()
    ma20 = pd.Series(closes).rolling(20).mean()
    ax.plot(dates, ma5,  color="#C9A84C", linewidth=1, linestyle="--", label="SMA-5")
    ax.plot(dates, ma20, color="#2E5E8E", linewidth=1, linestyle=":",  label="SMA-20")
    ax.axvline(dates[-1], color="red", linestyle="--", linewidth=0.8, label="Today")
    ax.set_xlabel("Date", fontsize=9, color="#a0aec0")
    ax.set_ylabel("Close Price", fontsize=9, color="#a0aec0")
    ax.tick_params(colors="#a0aec0")
    ax.legend(fontsize=8, facecolor="#1e2a3a", labelcolor="#e2e8f0")
    ax.grid(alpha=0.15, color="#4a5568")
    ax.set_title(f"{ticker} — 60-Day Price Window (Simulated)", fontsize=10, color="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a5568")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-header">🔍 SHAP Feature Importance</div>',
                unsafe_allow_html=True)

    feats  = ["Sentiment","MACD","RSI_14","EMA_10","SMA_20","BB_Width","SMA_5","Vol_Chg"]
    importance = np.array([0.31, 0.22, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03])
    colors = ["#C9A84C" if f=="Sentiment" else "#2E5E8E" for f in feats]

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), facecolor="#0e1117")
    ax2.set_facecolor("#0e1117")
    bars = ax2.barh(feats[::-1], importance[::-1], color=colors[::-1])
    ax2.set_xlabel("Mean |SHAP value|", fontsize=8, color="#a0aec0")
    ax2.set_title("Top Feature Contributions", fontsize=9, color="#e2e8f0")
    ax2.tick_params(colors="#a0aec0")
    ax2.grid(axis="x", alpha=0.15, color="#4a5568")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#4a5568")
    sent_patch = mpatches.Patch(color="#C9A84C", label="Sentiment (NLP)")
    tech_patch  = mpatches.Patch(color="#2E5E8E", label="Technical")
    ax2.legend(handles=[sent_patch, tech_patch], fontsize=7,
               facecolor="#1e2a3a", labelcolor="#e2e8f0")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.divider()

# ── Row 3: Model comparison table ────────────────────────────────
st.markdown('<div class="section-header">📋 Pilot Study — All 6 Model Results</div>',
            unsafe_allow_html=True)

table_data = {
    "Model": [
        "Random Forest – Hybrid ✦",
        "XGBoost – Hybrid",
        "LSTM – Hybrid",
        "LSTM – Baseline",
        "Random Forest – Baseline",
        "XGBoost – Baseline",
    ],
    "Treatment": ["Hybrid B","Hybrid B","Hybrid B","Baseline A","Baseline A","Baseline A"],
    "Accuracy":  [0.7660, 0.7660, 0.5000, 0.5106, 0.4574, 0.4255],
    "Precision": [0.6791, 0.6932, 0.4530, 0.4608, 0.4283, 0.4126],
    "Recall":    [0.8812, 0.8318, 0.7141, 0.6941, 0.7236, 0.6930],
    "F1-Score":  [0.7658, 0.7552, 0.5526, 0.5518, 0.5370, 0.5140],
}
df_table = pd.DataFrame(table_data)

def highlight_best(row):
    if "Hybrid" in row["Treatment"]:
        return ["background-color:#1a3a2a; color:#ffffff"]*len(row)
    return ["color:#ffffff"]*len(row)

styled = df_table.style.apply(highlight_best, axis=1)\
    .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"})

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "🎓 **DSA 3900 Implementation Semester** · Patricia Njeri Kiarie (ID 669781) · USIU-Africa  \n"
    "Data: Yahoo Finance (`yfinance`) + Reuters RSS + Google News  \n"
    "⚠️ *Educational prototype only. Not financial advice.*"
)
