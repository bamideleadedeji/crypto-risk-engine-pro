# Institutional Crypto Volatility Forecasting & Dynamic Risk Engine (v2.1)

Production-grade quantitative cryptocurrency risk analytics platform built in Python and Streamlit. Streams 100% real historical OHLCV market data directly from the **Coinbase Exchange Public API** (zero API keys required, zero synthetic/simulated fallback data) with exponential backoff retries, HTTP 429 rate-limit handling, and structured operational logging.

## Architecture

- `config.py` — Centralized quantitative constants, `RiskSummary` frozen dataclass, and structured logger configuration
- `market_data.py` — Resilient Coinbase Exchange OHLCV data pipeline with exponential backoff retries and vectorized NumPy returns
- `risk_quant_engine.py` — GARCH(1,1) MLE volatility forecasting, Parametric/Historical/Cornish-Fisher VaR, Expected Shortfall (CVaR), and dynamic position sizing
- `app.py` — Executive Streamlit dashboard with multi-sheet Excel (`.xlsx`) risk workbook export

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
