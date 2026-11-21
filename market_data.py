import time
import numpy as np
import pandas as pd
import requests

from config import (
    BACKOFF_BASE_SECONDS,
    COINBASE_HOST,
    MAX_RETRIES,
    MIN_REQUIRED_CANDLES,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_PAIRS,
    get_logger,
)

logger = get_logger("CryptoRiskEngine.MarketData")


def fetch_coinbase_candles(product_id: str = "BTC-USD", granularity: int = 86400) -> tuple[pd.DataFrame, str]:
    """
    Fetches 100% real historical OHLCV candles from Coinbase Exchange Public Market Data API
    using exponential backoff retries, HTTP 429 rate-limit handling, and vectorized NumPy returns.

    Parameters
    ----------
    product_id : str
        Spot trading pair identifier (e.g., 'BTC-USD', 'ETH-USD').
    granularity : int
        Candle interval in seconds (3600 for 1H, 86400 for 1D).

    Returns
    -------
    tuple[pd.DataFrame, str]
        Cleaned OHLCV DataFrame with vectorized log/percentage returns and status banner text.

    Raises
    ------
    ValueError
        If product_id or granularity is invalid.
    RuntimeError
        If Coinbase API fails after all retry attempts or returns insufficient data.
    """
    if product_id not in SUPPORTED_PAIRS.values():
        raise ValueError(f"Unsupported market pair '{product_id}'. Allowed: {list(SUPPORTED_PAIRS.values())}")
    if granularity not in (3600, 86400):
        raise ValueError(f"Unsupported granularity '{granularity}'. Must be 3600 (1H) or 86400 (1D).")

    url = f"https://{COINBASE_HOST}/products/{product_id}/candles"
    headers = {"User-Agent": "CryptoRiskEnginePro/2.1", "Accept": "application/json"}
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching {product_id} candles (granularity={granularity}s, attempt={attempt}/{MAX_RETRIES})")
            resp = requests.get(
                url,
                params={"granularity": int(granularity)},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", BACKOFF_BASE_SECONDS * attempt))
                logger.warning(f"Rate limit (HTTP 429) hit on {product_id}. Sleeping {retry_after:.1f}s before retry.")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            raw = resp.json()
            if not isinstance(raw, list) or len(raw) < MIN_REQUIRED_CANDLES:
                raise RuntimeError(f"Coinbase returned insufficient candle history ({len(raw) if isinstance(raw, list) else 0} rows) for {product_id}.")

            df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            df = df[df["close"] > 0].reset_index(drop=True)
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df["pct_return"] = df["close"].pct_change()
            df = df.dropna(subset=["log_return", "pct_return"]).reset_index(drop=True)

            tf_label = "Daily" if granularity == 86400 else "Hourly"
            status_msg = f"LIVE COINBASE EXCHANGE API — {product_id} ({len(df)} verified {tf_label} OHLCV candles)"
            logger.info(f"API fetch succeeded: {status_msg}")
            return df, status_msg

        except requests.Timeout as exc:
            last_error = f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s ({exc})"
            logger.warning(f"Attempt {attempt} timeout for {product_id}: {last_error}")
        except requests.ConnectionError as exc:
            last_error = f"Network connection error ({exc})"
            logger.warning(f"Attempt {attempt} connection error for {product_id}: {last_error}")
        except requests.HTTPError as exc:
            last_error = f"HTTP status error ({exc})"
            logger.warning(f"Attempt {attempt} HTTP error for {product_id}: {last_error}")
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON payload from Coinbase for {product_id}: {exc}") from exc

        if attempt < MAX_RETRIES:
            sleep_sec = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            time.sleep(sleep_sec)

    logger.error(f"Exhausted all {MAX_RETRIES} retries for {product_id}: {last_error}")
    raise RuntimeError(f"Unable to fetch live market data for {product_id} after {MAX_RETRIES} attempts: {last_error}")
