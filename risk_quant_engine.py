import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from config import (
    ANNUAL_DAYS_CRYPTO,
    CONFIDENCE_LEVEL_95,
    CONFIDENCE_LEVEL_99,
    EWMA_DECAY_ALPHA,
    MIN_REQUIRED_CANDLES,
    STOP_LOSS_VAR_MULTIPLIER,
    RiskSummary,
    get_logger,
)

logger = get_logger("CryptoRiskEngine.Quant")

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


def fit_garch_volatility(returns_pct: pd.Series, annual_factor: int = ANNUAL_DAYS_CRYPTO) -> tuple[pd.Series, float, str]:
    """
    Fits a GARCH(1,1) conditional volatility model via Maximum Likelihood Estimation,
    falling back to RiskMetrics EWMA if the optimizer fails to converge.

    Parameters
    ----------
    returns_pct : pd.Series
        Decimal periodic return series (e.g., 0.025 for +2.5%).
    annual_factor : int
        Periods per year used for annualization (365 for daily, 8760 for hourly).

    Returns
    -------
    tuple[pd.Series, float, str]
        Conditional periodic volatility series, next-period annualized volatility (%), and model name.
    """
    r_scaled = returns_pct.astype(float) * 100.0
    if HAS_ARCH and len(r_scaled) >= MIN_REQUIRED_CANDLES:
        try:
            am = arch_model(r_scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            cond_vol_periodic = (res.conditional_volatility / 100.0).clip(lower=1e-5)
            fcast = res.forecast(horizon=1, reindex=False)
            next_var = float(fcast.variance.values[-1, 0])
            next_periodic_vol = max(np.sqrt(max(next_var, 1e-10)) / 100.0, 1e-5)
            ann_vol = float(next_periodic_vol * np.sqrt(annual_factor) * 100.0)
            logger.info(f"GARCH(1,1) MLE converged cleanly | Forecast Annualized Vol: {ann_vol:.2f}%")
            return cond_vol_periodic, ann_vol, "GARCH(1,1) MLE (arch)"
        except Exception as exc:
            logger.warning(f"GARCH(1,1) MLE fell back to RiskMetrics EWMA: {exc}")

    ewma_var = (returns_pct.astype(float) ** 2).ewm(alpha=EWMA_DECAY_ALPHA, adjust=False).mean().clip(lower=1e-10)
    cond_vol_periodic = np.sqrt(ewma_var)
    next_periodic_vol = float(cond_vol_periodic.iloc[-1])
    ann_vol = float(next_periodic_vol * np.sqrt(annual_factor) * 100.0)
    logger.info(f"RiskMetrics EWMA computed | Forecast Annualized Vol: {ann_vol:.2f}%")
    return cond_vol_periodic, ann_vol, "RiskMetrics EWMA (lambda=0.94)"


def compute_risk_metrics(
    df: pd.DataFrame,
    account_equity_usd: float = 10000.0,
    target_ann_vol_pct: float = 25.0,
    max_risk_per_trade_pct: float = 2.0,
    annual_factor: int = ANNUAL_DAYS_CRYPTO,
) -> tuple[pd.DataFrame, RiskSummary]:
    """
    Computes conditional volatility, Parametric/Historical/Cornish-Fisher VaR,
    Expected Shortfall (CVaR), Max Drawdown, Sharpe Ratio, and Dynamic Position Sizing.

    Parameters
    ----------
    df : pd.DataFrame
        Validated OHLCV DataFrame containing 'close' and 'pct_return' columns.
    account_equity_usd : float
        Total trading capital in USD (must be > 0).
    target_ann_vol_pct : float
        Target annualized portfolio volatility ceiling in percent (must be > 0).
    max_risk_per_trade_pct : float
        Maximum 1-period 95% VaR loss budget per position in percent (must be > 0).
    annual_factor : int
        Annualization factor (365 for daily candles, 8760 for hourly candles).

    Returns
    -------
    tuple[pd.DataFrame, RiskSummary]
        Enriched time-series DataFrame and frozen RiskSummary dataclass instance.

    Raises
    ------
    ValueError
        If input DataFrame is empty or numeric parameters are non-positive.
    """
    if df is None or df.empty or "pct_return" not in df.columns or "close" not in df.columns:
        raise ValueError("Input DataFrame must contain non-empty 'close' and 'pct_return' columns.")
    if account_equity_usd <= 0 or target_ann_vol_pct <= 0 or max_risk_per_trade_pct <= 0 or annual_factor <= 0:
        raise ValueError("Equity, target volatility, risk budget, and annualization factor must all be > 0.")

    work = df.copy()
    rets = work["pct_return"].astype(float)
    cond_vol, forecast_ann_vol, model_label = fit_garch_volatility(rets, annual_factor=annual_factor)
    work["cond_daily_vol"] = cond_vol.values
    work["cond_ann_vol_pct"] = work["cond_daily_vol"] * np.sqrt(annual_factor) * 100.0

    z_95 = norm.ppf(CONFIDENCE_LEVEL_95)
    z_99 = norm.ppf(CONFIDENCE_LEVEL_99)
    work["var_95_pct"] = np.abs(z_95 * work["cond_daily_vol"]) * 100.0
    work["var_99_pct"] = np.abs(z_99 * work["cond_daily_vol"]) * 100.0

    hist_var_95 = float(abs(np.percentile(rets, 5)) * 100.0)
    hist_var_99 = float(abs(np.percentile(rets, 1)) * 100.0)
    tail_5 = rets[rets <= np.percentile(rets, 5)]
    cvar_95 = float(abs(tail_5.mean()) * 100.0) if len(tail_5) > 0 else hist_var_95

    s = float(skew(rets))
    k = float(kurtosis(rets))
    z_cf = (
        z_95
        + (1.0 / 6.0) * (z_95**2 - 1.0) * s
        + (1.0 / 24.0) * (z_95**3 - 3.0 * z_95) * k
        - (1.0 / 36.0) * (2.0 * z_95**3 - 5.0 * z_95) * (s**2)
    )
    next_periodic_vol = float(work["cond_daily_vol"].iloc[-1])
    cf_var_95 = float(abs(z_cf) * next_periodic_vol * 100.0)
    param_var_95 = float(abs(z_95) * next_periodic_vol * 100.0)

    cum_curve = (1.0 + rets).cumprod()
    peak = cum_curve.cummax()
    dd = (cum_curve - peak) / peak
    work["drawdown_pct"] = dd * 100.0
    max_dd_pct = float(abs(dd.min()) * 100.0)

    mean_ann_ret = float(rets.mean() * annual_factor * 100.0)
    realized_ann_vol = float(rets.std() * np.sqrt(annual_factor) * 100.0)
    sharpe = (mean_ann_ret / realized_ann_vol) if realized_ann_vol > 0 else 0.0

    latest_price = float(work["close"].iloc[-1])
    if latest_price <= 0:
        raise ValueError("Latest asset closing price must be strictly positive.")

    vol_scalar = min(1.0, target_ann_vol_pct / max(forecast_ann_vol, 1.0))
    vol_target_usd = account_equity_usd * vol_scalar
    max_loss_usd = account_equity_usd * (max_risk_per_trade_pct / 100.0)
    var_budget_usd = max_loss_usd / max(param_var_95 / 100.0, 1e-4)

    recommended_usd = min(account_equity_usd, vol_target_usd, var_budget_usd)
    recommended_units = recommended_usd / latest_price
    stop_distance_pct = param_var_95 * STOP_LOSS_VAR_MULTIPLIER
    suggested_stop_px = latest_price * (1.0 - stop_distance_pct / 100.0)

    # Bear-Market Cash Shield: 0% crypto allocation when Spot < 50-Period EMA
    ema50 = float(work["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    if latest_price < ema50:
        recommended_usd = 0.0
        recommended_units = 0.0
        model_label = f"{model_label} | BEAR SHIELD (Spot < EMA50 ${ema50:,.0f} -> 100% Cash/USDC)"

    summary = RiskSummary(
        model_label=model_label,
        latest_price=round(latest_price, 2),
        forecast_ann_vol_pct=round(forecast_ann_vol, 2),
        realized_ann_vol_pct=round(realized_ann_vol, 2),
        param_var_95_pct=round(param_var_95, 2),
        hist_var_95_pct=round(hist_var_95, 2),
        cf_var_95_pct=round(cf_var_95, 2),
        hist_var_99_pct=round(hist_var_99, 2),
        cvar_95_pct=round(cvar_95, 2),
        max_dd_pct=round(max_dd_pct, 2),
        sharpe_ratio=round(sharpe, 2),
        skewness=round(s, 2),
        excess_kurtosis=round(k, 2),
        recommended_usd=round(recommended_usd, 2),
        recommended_pct=round((recommended_usd / account_equity_usd) * 100.0, 1),
        recommended_units=round(recommended_units, 6),
        suggested_stop_px=round(suggested_stop_px, 2),
        stop_distance_pct=round(stop_distance_pct, 2),
    )
    logger.info(
        f"Risk computation finished | Spot=${summary.latest_price:,.2f} | "
        f"ParamVaR95={summary.param_var_95_pct:.2f}% | RecSize=${summary.recommended_usd:,.2f}"
    )
    return work, summary
