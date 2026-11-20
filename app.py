import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_data import SUPPORTED_PAIRS, fetch_coinbase_candles
from risk_quant_engine import compute_risk_metrics

st.set_page_config(
    page_title="Crypto Volatility Forecasting & Dynamic Risk Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_risk_excel(df: pd.DataFrame, summary: dict, pair_label: str, source_msg: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_rows = [
            {"Metric": "Generated (UTC)", "Value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")},
            {"Metric": "Market Pair", "Value": pair_label},
            {"Metric": "Verified Feed", "Value": source_msg},
            {"Metric": "Volatility Model", "Value": summary["model_label"]},
            {"Metric": "Latest Spot Close ($)", "Value": f"${summary['latest_price']:,.2f}"},
            {"Metric": "Forecast Annualized Volatility (%)", "Value": f"{summary['forecast_ann_vol_pct']:.2f}%"},
            {"Metric": "1-Day 95% Parametric VaR (%)", "Value": f"{summary['param_var_95_pct']:.2f}%"},
            {"Metric": "1-Day 95% Historical VaR (%)", "Value": f"{summary['hist_var_95_pct']:.2f}%"},
            {"Metric": "1-Day 95% Cornish-Fisher VaR (%)", "Value": f"{summary['cf_var_95_pct']:.2f}%"},
            {"Metric": "1-Day 95% Expected Shortfall CVaR (%)", "Value": f"{summary['cvar_95_pct']:.2f}%"},
            {"Metric": "Maximum Historical Drawdown (%)", "Value": f"{summary['max_dd_pct']:.2f}%"},
            {"Metric": "Recommended Position Size ($)", "Value": f"${summary['recommended_usd']:,.2f} ({summary['recommended_pct']:.1f}%)"},
            {"Metric": "Recommended Units", "Value": summary["recommended_units"]},
            {"Metric": "Volatility-Adjusted Stop Price ($)", "Value": f"${summary['suggested_stop_px']:,.2f} (-{summary['stop_distance_pct']:.2f}%)"},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="01_Risk_Summary", index=False)
        export_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "pct_return", "cond_ann_vol_pct", "var_95_pct", "drawdown_pct"] if c in df.columns]
        out_df = df[export_cols].copy()
        if "date" in out_df.columns:
            out_df["date"] = out_df["date"].astype(str)
        out_df.to_excel(writer, sheet_name="02_GARCH_VaR_Timeseries", index=False)
    return buf.getvalue()
@st.cache_data(ttl=900)
def cached_candles(product_id: str, granularity: int):
    return fetch_coinbase_candles(product_id=product_id, granularity=granularity)


st.title("⚡ Institutional Crypto Volatility Forecasting & Dynamic Risk Engine")
st.caption("100% Real Exchange Data • GARCH(1,1) Conditional Volatility • Parametric, Historical & Cornish-Fisher VaR • Dynamic Position Sizing")

st.sidebar.header("⚙️ Portfolio & Risk Parameters")
pair_label = st.sidebar.selectbox("Select Spot Market Pair", list(SUPPORTED_PAIRS.keys()))
product_id = SUPPORTED_PAIRS[pair_label]
tf_choice = st.sidebar.radio("Candle Interval", ["Daily (1D — 350 Days)", "Hourly (1H — 350 Hours)"])
granularity = 86400 if "Daily" in tf_choice else 3600
ann_factor = 365 if granularity == 86400 else 365 * 24

account_equity = st.sidebar.number_input("Trading Capital / Account Equity ($)", 500.0, 50000000.0, 10000.0, 500.0)
target_vol = st.sidebar.slider("Target Annualized Portfolio Volatility (%)", 10.0, 80.0, 25.0, 2.5)
max_risk_pct = st.sidebar.slider("Max 1-Day 95% VaR Risk Budget per Position (%)", 0.5, 5.0, 2.0, 0.25)

if st.sidebar.button("🔄 Refresh Live Market Feed"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner(f"Fetching verified OHLCV market data for {product_id} from Coinbase Exchange..."):
        raw_df, feed_status = cached_candles(product_id, granularity)
except Exception as exc:
    st.error(f"Live market feed error: {exc}")
    st.stop()

st.success(f"✅ **Verified Market Feed:** {feed_status}")

scored_df, summary = compute_risk_metrics(
    raw_df,
    account_equity_usd=float(account_equity),
    target_ann_vol_pct=float(target_vol),
    max_risk_per_trade_pct=float(max_risk_pct),
    annual_factor=ann_factor,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{product_id} Spot Close", f"${summary['latest_price']:,.2f}", delta=f"Max DD: -{summary['max_dd_pct']:.2f}%", delta_color="inverse")
k2.metric("GARCH(1,1) Forecast Vol", f"{summary['forecast_ann_vol_pct']:.2f}%", delta=f"Realized: {summary['realized_ann_vol_pct']:.2f}%")
k3.metric("1-Period 95% VaR (Param)", f"{summary['param_var_95_pct']:.2f}%", delta=f"CVaR (ES): {summary['cvar_95_pct']:.2f}%", delta_color="inverse")
k4.metric("Recommended Position Size", f"${summary['recommended_usd']:,.2f}", delta=f"{summary['recommended_pct']:.1f}% of Capital")

st.markdown("---")
t1, t2, t3 = st.tabs([
    "📈 GARCH(1,1) Volatility Regime",
    "🛡️ Tail Risk (VaR & Expected Shortfall)",
    "🎯 Dynamic Position Sizing & Excel Export",
])

with t1:
    st.subheader(f"Conditional Annualized Volatility ({summary['model_label']})")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=scored_df["date"], y=scored_df["cond_ann_vol_pct"], name="Conditional Vol (%)", line=dict(color="#00c853", width=2)))
    fig_vol.add_hline(y=float(target_vol), line_dash="dash", line_color="orange", annotation_text=f"Target Vol ({target_vol:.0f}%)")
    fig_vol.update_layout(yaxis_title="Annualized Volatility (%)", height=390)
    st.plotly_chart(fig_vol, use_container_width=True)

    fig_px = px.line(scored_df, x="date", y="close", title=f"{product_id} Historical Closing Price ($)")
    fig_px.update_layout(height=340)
    st.plotly_chart(fig_px, use_container_width=True)

with t2:
    st.subheader("Parametric, Historical & Cornish-Fisher Fat-Tail VaR Comparison")
    var_table = pd.DataFrame([
        {"Risk Metric": "1-Period 95% Parametric VaR (GARCH)", "LossPct": f"{summary['param_var_95_pct']:.2f}%", "Dollar_Impact_Full_Capital": f"${account_equity * summary['param_var_95_pct'] / 100:,.2f}"},
        {"Risk Metric": "1-Period 95% Historical Simulation VaR", "LossPct": f"{summary['hist_var_95_pct']:.2f}%", "Dollar_Impact_Full_Capital": f"${account_equity * summary['hist_var_95_pct'] / 100:,.2f}"},
        {"Risk Metric": "1-Period 95% Cornish-Fisher Fat-Tail VaR", "LossPct": f"{summary['cf_var_95_pct']:.2f}%", "Dollar_Impact_Full_Capital": f"${account_equity * summary['cf_var_95_pct'] / 100:,.2f}"},
        {"Risk Metric": "1-Period 95% Expected Shortfall (CVaR)", "LossPct": f"{summary['cvar_95_pct']:.2f}%", "Dollar_Impact_Full_Capital": f"${account_equity * summary['cvar_95_pct'] / 100:,.2f}"},
        {"Risk Metric": "1-Period 99% Historical VaR", "LossPct": f"{summary['hist_var_99_pct']:.2f}%", "Dollar_Impact_Full_Capital": f"${account_equity * summary['hist_var_99_pct'] / 100:,.2f}"},
    ])
    st.dataframe(var_table, use_container_width=True)

    fig_hist = px.histogram(scored_df, x="pct_return", nbins=45, title=f"{product_id} Return Distribution (Skew: {summary['skewness']:.2f} | Kurtosis: {summary['excess_kurtosis']:.2f})")
    fig_hist.update_layout(height=360)
    st.plotly_chart(fig_hist, use_container_width=True)

with t3:
    st.subheader("Volatility-Targeted & VaR-Constrained Allocation Ticket")
    p1, p2, p3 = st.columns(3)
    p1.metric("Optimal Dollar Allocation", f"${summary['recommended_usd']:,.2f}", delta=f"{summary['recommended_pct']:.1f}% of Equity")
    p2.metric("Optimal Asset Quantity", f"{summary['recommended_units']:.6f} {product_id.split('-')[0]}")
    p3.metric("1.5x VaR Stop-Loss Level", f"${summary['suggested_stop_px']:,.2f}", delta=f"-{summary['stop_distance_pct']:.2f}% below spot", delta_color="inverse")

    excel_bytes = build_risk_excel(scored_df, summary, pair_label, feed_status)
    st.download_button(
        label="📊 Download Multi-Sheet Excel Risk Report (.xlsx)",
        data=excel_bytes,
        file_name=f"Crypto_Risk_Report_{product_id}.xlsx",
    )
