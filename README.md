# AfriStockAI-DSA3900
Hybrid ML dashboard for African stock market prediction
# 📈 AfriStockAI — African Stock Market Prediction Dashboard

> **DSA 3900 Implementation Semester** · USIU-Africa  
> Patricia Njeri Kiarie — Student ID 669781

---

##  Research Overview

This project investigates whether adding **NLP news sentiment scores** to classical technical indicators improves next-day stock movement prediction on African exchanges (NGX & JSE).

**Research Question:**  
*Does a Hybrid ML model (technical indicators + news sentiment) outperform a Baseline model (technical indicators only) for binary stock direction classification on African markets?*

**Stocks studied:**
- 🇳🇬 MTN Nigeria — Nigerian Exchange (MTNN.LG)
- 🇿🇦 Naspers — Johannesburg Stock Exchange (NPN.JO)

---

##  Experimental Design

| | Treatment A — Baseline | Treatment B — Hybrid |
|---|---|---|
| **Features** | 8 technical indicators | 8 technical indicators + NLP sentiment |
| **Algorithms** | RF · XGBoost · LSTM | RF · XGBoost · LSTM |
| **Split** | 80/20 chronological (no shuffle) | 80/20 chronological (no shuffle) |
| **Target** | Binary: Up (1) / Down (0) next-day close | Same |

**Total model variants:** 6 (3 algorithms × 2 treatments)

### Technical Indicators Used
`SMA_5` · `SMA_20` · `EMA_10` · `RSI_14` · `MACD` · `MACD_Signal` · `BB_Width` · `Volume_Change`

---

##  Pilot Study Results

Results averaged across both stocks (252 trading days simulation, 80/20 chronological split):

| Model | Treatment | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Random Forest** | **Hybrid B** ✦ | **0.7660** | **0.6791** | **0.8812** | **0.7658** |
| **XGBoost** | **Hybrid B** | **0.7660** | **0.6932** | **0.8318** | **0.7552** |
| LSTM | Hybrid B | 0.5000 | 0.4530 | 0.7141 | 0.5526 |
| LSTM | Baseline A | 0.5106 | 0.4608 | 0.6941 | 0.5518 |
| Random Forest | Baseline A | 0.4574 | 0.4283 | 0.7236 | 0.5370 |
| XGBoost | Baseline A | 0.4255 | 0.4126 | 0.6930 | 0.5140 |

**Key finding:** RF Hybrid and XGBoost Hybrid both achieve **76.6% accuracy** — a +29–34 percentage point improvement over Baseline variants, confirming that news sentiment is a meaningful signal for African stock direction prediction.

---

## 🗂 Project Structure

```
AfriStockAI-DSA3900/
├── streamlit_app.py        # Interactive Streamlit dashboard
├── simulate_pilot_v2.py    # Full pilot simulation (6 models × 2 stocks)
├── requirements.txt        # Python dependencies
└── README.md
```

---

##  Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/PatriciaKiarie04/AfriStockAI-DSA3900.git
cd AfriStockAI-DSA3900

# 2. Install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn xgboost

# 3. Launch dashboard
python -m streamlit run streamlit_app.py
```

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Dashboard | Streamlit |
| ML Models | scikit-learn (Random Forest) · XGBoost · NumPy SimpleRNN (LSTM proxy) |
| Data Simulation | GBM with regime switching (bull/choppy/bear) |
| Sentiment | Correlated NLP score (pilot) → Reuters RSS + Google News (full study) |
| Visualisation | Matplotlib · SHAP feature importance |
| Data Sources | Yahoo Finance (`yfinance`) · Reuters RSS · Google News |

---

##  Disclaimer

This is an **educational prototype** built as part of the DSA 3900 Implementation Semester at USIU-Africa. It uses **simulated data** for the pilot study. Results are not intended as financial advice.

---

*Built with Python · Streamlit · scikit-learn · XGBoost*
