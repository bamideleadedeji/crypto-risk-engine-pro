# Institutional Crypto Volatility Forecasting & Dynamic Risk Engine

100% real-data quantitative cryptocurrency risk dashboard built in Python and Streamlit. Streams live OHLCV candles directly from **Coinbase Exchange Public Market Data API** (zero API keys required, zero simulated data) and computes:

- **GARCH(1,1) Conditional Volatility Forecasting** (`arch` MLE with RiskMetrics EWMA fallback)
- **Parametric, Historical & Cornish-Fisher Fat-Tail Value-at-Risk (95% & 99% VaR)**
- **Expected Shortfall (CVaR) & Maximum Drawdown Analytics**
- **Volatility-Targeted & VaR-Budgeted Position Sizing + Stop-Loss Calculator**
- **One-Click Multi-Sheet Excel Risk Workbook Export (`.xlsx`)**

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
