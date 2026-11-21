import logging
from dataclasses import dataclass

APP_TITLE = "Institutional Crypto Volatility Forecasting & Dynamic Risk Engine"
APP_VERSION = "2.2.0"

SUPPORTED_PAIRS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Chainlink (LINK-USD)": "LINK-USD",
    "Avalanche (AVAX-USD)": "AVAX-USD",
}

CURRENCIES = {
    "USD ($)": {"symbol": "$", "default_fx": 1.0, "default_equity": 10000.0, "step": 500.0},
    "NGN (₦)": {"symbol": "₦", "default_fx": 1550.0, "default_equity": 61075.0, "step": 5000.0},
    "EUR (€)": {"symbol": "€", "default_fx": 0.92, "default_equity": 10000.0, "step": 500.0},
    "GBP (£)": {"symbol": "£", "default_fx": 0.79, "default_equity": 10000.0, "step": 500.0},
}

COINBASE_HOST = ".".join(["api", "exchange", "coinbase", "com"])
MIN_REQUIRED_CANDLES = 40
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 15
CACHE_TTL_SECONDS = 900

ANNUAL_DAYS_CRYPTO = 365
CONFIDENCE_LEVEL_95 = 0.05
CONFIDENCE_LEVEL_99 = 0.01
EWMA_DECAY_ALPHA = 0.06
STOP_LOSS_VAR_MULTIPLIER = 1.5
DEFAULT_EQUITY_USD = 10000.0
DEFAULT_TARGET_VOL_PCT = 25.0
DEFAULT_MAX_RISK_PCT = 2.0


@dataclass(frozen=True)
class RiskSummary:
    """Strongly typed output container for institutional GARCH, VaR, and sizing metrics."""
    model_label: str
    latest_price: float
    forecast_ann_vol_pct: float
    realized_ann_vol_pct: float
    param_var_95_pct: float
    hist_var_95_pct: float
    cf_var_95_pct: float
    hist_var_99_pct: float
    cvar_95_pct: float
    max_dd_pct: float
    sharpe_ratio: float
    skewness: float
    excess_kurtosis: float
    recommended_usd: float
    recommended_pct: float
    recommended_units: float
    suggested_stop_px: float
    stop_distance_pct: float


def get_logger(name: str = "CryptoRiskEngine") -> logging.Logger:
    """Returns a configured structured console logger without duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
