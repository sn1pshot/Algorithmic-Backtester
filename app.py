from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_csv, download_nifty, validate_ohlcv
from src.strategies import add_indicators, generate_signals
from src.backtester import run_backtest
from src.metrics import performance_metrics, trade_metrics, yearly_returns

st.set_page_config(page_title="NIFTY 50 Strategy Lab", page_icon="📈", layout="wide")
st.title("NIFTY 50 Algorithmic Trading Backtester")
st.caption("Research tool for historical simulations — not investment advice.")

with st.sidebar:
    st.header("Configuration")
    test_year = st.selectbox("Out-of-sample test year", [2022, 2023, 2024], index=2)
    selected_strategies = st.multiselect(
        "Strategies",
        ["50/200-day MA crossover", "RSI mean reversion"],
        default=["50/200-day MA crossover", "RSI mean reversion"],
    )
    initial_capital = st.number_input("Initial capital (₹)", min_value=1000, value=1_000_000, step=50_000)
    cost_pct = st.number_input(
        "Transaction cost per 100% position turnover (%)",
        min_value=0.0, max_value=5.0, value=0.10, step=0.05,
        help="Applied to absolute daily position changes. Includes your combined assumed costs/slippage; replace with a realistic cost model.",
    )
    rsi_period = st.selectbox("RSI lookback", [7, 14, 21], index=1)
    rsi_entry = st.selectbox("RSI entry threshold", [20, 25, 30], index=2)
    rsi_exit = st.selectbox("RSI exit threshold", [45, 50, 60], index=1)
    st.divider()
    st.caption("Default strategy parameters should be fixed before inspecting the selected test year.")

st.subheader("1. Load daily NIFTY 50 data")
st.write("Use a CSV containing at least `Date` and `Close` columns, or attempt a download from Yahoo Finance (`^NSEI`). The project does not bundle market data.")
data_source = st.radio("Data source", ["Download from Yahoo Finance", "Upload CSV"], horizontal=True)

raw = None
if data_source == "Download from Yahoo Finance":
    if st.button("Download NIFTY 50 history (2000–2024)", type="primary"):
        with st.spinner("Downloading data..."):
            try:
                raw = download_nifty("2000-01-01", "2025-01-01")
                st.session_state["market_data"] = raw
                st.success(f"Downloaded {len(raw):,} rows.")
            except Exception as exc:
                st.error(f"Download failed: {exc}. Try uploading a CSV instead.")
    if "market_data" in st.session_state:
        raw = st.session_state["market_data"]
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = load_csv(uploaded)
            st.session_state["market_data"] = raw
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if raw is None:
    st.info("Load data to run the backtest. Expected columns: Date and Close; Open/High/Low/Volume are optional.")
    st.stop()

try:
    data = validate_ohlcv(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

min_date, max_date = data.index.min().date(), data.index.max().date()
st.write(f"**Loaded:** {len(data):,} trading rows · {min_date} to {max_date}")
if pd.Timestamp(min_date) > pd.Timestamp("2000-01-01") or pd.Timestamp(max_date) < pd.Timestamp("2024-12-31"):
    st.warning("The dataset may not cover the full requested 2000–2024 period. Results only use available observations.")

test_start = pd.Timestamp(f"{test_year}-01-01")
test_end = pd.Timestamp(f"{test_year+1}-01-01")
if data.index.min() > test_start or data.index.max() < pd.Timestamp(f"{test_year}-12-31"):
    st.error(f"Data does not cover all of {test_year}. Choose another year or upload a dataset that includes it.")
    st.stop()

# Use prior history for indicator warm-up, but score only the selected out-of-sample year.
warmup_start = pd.Timestamp("2000-01-01")
work = data.loc[warmup_start:test_end - pd.Timedelta(days=1)].copy()
if work.empty:
    st.error("No data available for the selected period.")
    st.stop()

indicators = add_indicators(work, rsi_period=rsi_period)
results = {}
for strategy_name in selected_strategies:
    signal = generate_signals(
        indicators, strategy_name,
        rsi_period=rsi_period, rsi_entry=rsi_entry, rsi_exit=rsi_exit
    )
    bt = run_backtest(
        indicators, signal, initial_capital=float(initial_capital),
        cost_rate=float(cost_pct) / 100.0
    )
    results[strategy_name] = bt

# Benchmark is fully invested from the first test-year session.
benchmark_position = pd.Series(1.0, index=indicators.index)
benchmark_bt = run_backtest(
    indicators, benchmark_position, initial_capital=float(initial_capital),
    cost_rate=0.0, charge_initial_entry=False
)
results["NIFTY 50 buy-and-hold"] = benchmark_bt

test_index = data.loc[test_start:test_end - pd.Timedelta(days=1)].index
if len(test_index) == 0:
    st.error("No test-year observations found.")
    st.stop()

# Metrics computed strictly on test-year equity/returns; preserve start capital at the first test session.
summary_rows = []
for name, bt in results.items():
    test_bt = bt.loc[test_index].copy()
    # Normalize the test-year equity to initial capital, preserving daily strategy returns.
    daily_returns = test_bt["net_return"].fillna(0.0)
    equity = float(initial_capital) * (1.0 + daily_returns).cumprod()
    summary = performance_metrics(daily_returns, equity, initial_capital)
    trades = trade_metrics(test_bt["position"], daily_returns)
    summary_rows.append({"Strategy": name, **summary, **trades})
summary = pd.DataFrame(summary_rows).set_index("Strategy")

st.subheader(f"2. Performance summary — {test_year}")
display_summary = summary.copy()
percent_cols = ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown", "Best Day", "Worst Day"]
for col in percent_cols:
    if col in display_summary:
        display_summary[col] = display_summary[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Win Rate", "Profit Factor"]:
    if col in display_summary:
        display_summary[col] = display_summary[col].map(
            lambda x: (f"{x:.2%}" if col == "Win Rate" else f"{x:.2f}") if pd.notna(x) else "—"
        )
st.dataframe(display_summary, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Equity curve", "Drawdown", "Monthly returns", "Trade log / data"])
with tab1:
    fig = go.Figure()
    for name, bt in results.items():
        series = bt.loc[test_index, "net_return"].fillna(0.0)
        curve = float(initial_capital) * (1 + series).cumprod()
        fig.add_trace(go.Scatter(x=curve.index, y=curve, mode="lines", name=name))
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio value (₹)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = go.Figure()
    for name, bt in results.items():
        series = bt.loc[test_index, "net_return"].fillna(0.0)
        curve = (1 + series).cumprod()
        dd = curve / curve.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=name))
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    monthly_frames = []
    for name, bt in results.items():
        monthly = (1 + bt.loc[test_index, "net_return"].fillna(0.0)).resample("ME").prod() - 1
        monthly_frames.append(pd.DataFrame({"Month": monthly.index.strftime("%Y-%m"), "Strategy": name, "Return": monthly.values}))
    monthly_df = pd.concat(monthly_frames, ignore_index=True)
    if not monthly_df.empty:
        pivot = monthly_df.pivot(index="Strategy", columns="Month", values="Return")
        st.dataframe(pivot.style.format("{:.2%}"), use_container_width=True)
        fig = px.imshow(pivot, text_auto=".1%", aspect="auto", color_continuous_midpoint=0,
                        labels=dict(color="Monthly return"))
        st.plotly_chart(fig, use_container_width=True)
with tab4:
    strategy_for_log = st.selectbox("Inspect strategy", list(results.keys()))
    chosen = results[strategy_for_log].loc[test_index].copy()
    st.dataframe(chosen.tail(100), use_container_width=True)
    csv = chosen.to_csv(index=True).encode("utf-8")
    st.download_button("Download selected strategy results CSV", csv, f"{strategy_for_log.replace('/', '-')}_{test_year}.csv", "text/csv")
    summary_csv = summary.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics CSV", summary_csv, f"nifty_metrics_{test_year}.csv", "text/csv")

st.subheader("3. Interpretation notes")
st.markdown("""
- **CAGR** is annualized growth over the scored test-year window; with one calendar year it is close to the total return, adjusted for the exact elapsed time.
- **Sharpe / Sortino** use daily returns and a zero risk-free rate in this starter implementation. Set a defensible risk-free series for a more rigorous study.
- **Trade count** is estimated from changes between flat and invested states; it is not an exchange execution ledger.
- The simulation assumes fractional exposure to the index and does not model futures rolls, lot sizes, dividends/total-return index differences, margin, market impact, or taxes in detail.
- Signals use closing prices and are shifted to the next session to avoid trading on the same close that generated the signal.
""")
