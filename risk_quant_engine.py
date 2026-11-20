import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


def fit_garch_volatility(returns_pct: pd.Series, annual_factor: int = 365) -> tuple[pd.Series, float, str]:
    """
    Fits a real GARCH(1,1) conditional volatility model on percentage returns.
    Falls back cleanly to RiskMetrics EWMA (lambda=0.94) only if optimizer does not converge.
    """
    r_scaled = returns_pct.astype(float) * 100.0
    if HAS_ARCH and len(r_scaled) >= 40:
        try:
            am = arch_model(r_scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            cond_vol_daily = (res.conditional_volatility / 100.0).clip(lower=1e-5)
            fcast = res.forecast(horizon=1, reindex=False)
            next_var = float(fcast.variance.values[-1, 0])
            next_daily_vol = max(np.sqrt(max(next_var, 1e-10)) / 100.0, 1e-5)
            ann_vol = float(next_daily_vol * np.sqrt(annual_factor) * 100.0)
            return cond_vol_daily, ann_vol, "GARCH(1,1) MLE (arch)"
        except Exception:
            pass

    ewma_var = (returns_pct.astype(float) ** 2).ewm(alpha=0.06, adjust=False).mean().clip(lower=1e-10)
    cond_vol_daily = np.sqrt(ewma_var)
    next_daily_vol = float(cond_vol_daily.iloc[-1])
    ann_vol = float(next_daily_vol * np.sqrt(annual_factor) * 100.0)
    return cond_vol_daily, ann_vol, "RiskMetrics EWMA (lambda=0.94)"
def compute_risk_metrics(
    df: pd.DataFrame,
    account_equity_usd: float = 10000.0,
    target_ann_vol_pct: float = 25.0,
    max_risk_per_trade_pct: float = 2.0,
    annual_factor: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """
    Computes 1-Day 95% & 99% VaR (Parametric, Historical, Cornish-Fisher),
    Expected Shortfall (CVaR), Max Drawdown, Sharpe Ratio, and Dynamic Position Sizing.
    """
    work = df.copy()
    rets = work["pct_return"].astype(float)
    cond_vol, forecast_ann_vol, model_label = fit_garch_volatility(rets, annual_factor=annual_factor)
    work["cond_daily_vol"] = cond_vol.values
    work["cond_ann_vol_pct"] = work["cond_daily_vol"] * np.sqrt(annual_factor) * 100.0

    # Dynamic 1-Day 95% & 99% VaR series based on conditional volatility
    z_95 = norm.ppf(0.05)
    z_99 = norm.ppf(0.01)
    work["var_95_pct"] = np.abs(z_95 * work["cond_daily_vol"]) * 100.0
    work["var_99_pct"] = np.abs(z_99 * work["cond_daily_vol"]) * 100.0

    # Historical & Cornish-Fisher fat-tail adjustments
    hist_var_95 = float(abs(np.percentile(rets, 5)) * 100.0)
    hist_var_99 = float(abs(np.percentile(rets, 1)) * 100.0)
    tail_5 = rets[rets <= np.percentile(rets, 5)]
    cvar_95 = float(abs(tail_5.mean()) * 100.0) if len(tail_5) > 0 else hist_var_95

    s = float(skew(rets))
    k = float(kurtosis(rets))
    z_cf = z_95 + (1.0 / 6.0) * (z_95**2 - 1.0) * s + (1.0 / 24.0) * (z_95**3 - 3.0 * z_95) * k - (1.0 / 36.0) * (2.0 * z_95**3 - 5.0 * z_95) * (s**2)
    next_daily_vol = float(work["cond_daily_vol"].iloc[-1])
    cf_var_95 = float(abs(z_cf) * next_daily_vol * 100.0)
    param_var_95 = float(abs(z_95) * next_daily_vol * 100.0)

    # Drawdown & Annualized Sharpe
    cum_curve = (1.0 + rets).cumprod()
    peak = cum_curve.cummax()
    dd = (cum_curve - peak) / peak
    work["drawdown_pct"] = dd * 100.0
    max_dd_pct = float(abs(dd.min()) * 100.0)

    mean_ann_ret = float(rets.mean() * annual_factor * 100.0)
    realized_ann_vol = float(rets.std() * np.sqrt(annual_factor) * 100.0)
    sharpe = (mean_ann_ret / realized_ann_vol) if realized_ann_vol > 0 else 0.0

    # Dynamic Volatility-Targeted & VaR-Capped Position Sizing
    latest_price = float(work["close"].iloc[-1])
    vol_scalar = min(1.0, target_ann_vol_pct / max(forecast_ann_vol, 1.0))
    vol_target_usd = account_equity_usd * vol_scalar

    max_loss_usd = account_equity_usd * (max_risk_per_trade_pct / 100.0)
    var_budget_usd = max_loss_usd / max(param_var_95 / 100.0, 1e-4)
    recommended_usd = min(account_equity_usd, vol_target_usd, var_budget_usd)
    recommended_units = recommended_usd / latest_price if latest_price > 0 else 0.0
    stop_distance_pct = param_var_95 * 1.5
    suggested_stop_px = latest_price * (1.0 - stop_distance_pct / 100.0)

    summary = {
        "model_label": model_label,
        "latest_price": latest_price,
        "forecast_ann_vol_pct": round(forecast_ann_vol, 2),
        "realized_ann_vol_pct": round(realized_ann_vol, 2),
        "param_var_95_pct": round(param_var_95, 2),
        "hist_var_95_pct": round(hist_var_95, 2),
        "cf_var_95_pct": round(cf_var_95, 2),
        "hist_var_99_pct": round(hist_var_99, 2),
        "cvar_95_pct": round(cvar_95, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "skewness": round(s, 2),
        "excess_kurtosis": round(k, 2),
        "recommended_usd": round(recommended_usd, 2),
        "recommended_pct": round((recommended_usd / account_equity_usd * 100.0) if account_equity_usd > 0 else 0.0, 1),
        "recommended_units": round(recommended_units, 6),
        "suggested_stop_px": round(suggested_stop_px, 2),
        "stop_distance_pct": round(stop_distance_pct, 2),
    }
    return work, summary
