# NIFTY 50 Algorithmic Trading Backtester

A small, explainable Python project for evaluating two long/flat strategies on daily NIFTY 50 index data:

1. **50/200-day moving-average crossover**
2. **RSI mean reversion**

The Streamlit dashboard lets you choose a test year (2022, 2023, or 2024), initial capital, RSI parameters, and estimated transaction costs. It reports return, CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown, Calmar, trade count, win rate, profit factor, and average holding days. It plots equity and drawdown curves and a monthly-return heatmap.

> This is an educational research tool, not investment advice. It is not a broker-connected system and does not produce executable orders.

## Quick start

Requires Python 3.10+.

```bash
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If activation is blocked in PowerShell, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Data

The dashboard can attempt to download `^NSEI` through Yahoo Finance. Network availability and historical coverage are not guaranteed. If download fails, upload a CSV with at least:

```csv
Date,Close
2000-01-03,1592.20
2000-01-04,...
```

Open, High, Low, and Volume are optional. Use a reliable source and validate its coverage and values. The project intentionally does **not** bundle market data.

For the requested evaluation years, data should cover 2000 through the end of 2024. A CSV exported from a vendor should have one row per trading session, sorted or sortable by date, with no duplicate dates. If your source has a multi-row header or nonstandard column names, normalize it first.

## Methodology

- **Development history:** daily observations from 2000 through 2021 are available for indicator warm-up and research. This starter uses fixed, transparent defaults rather than automatically optimizing parameters.
- **Test window:** select 2022, 2023, or 2024. Only that calendar year is included in the displayed score.
- **MA crossover:** long/flat; target exposure is 1 when SMA(50) > SMA(200), otherwise 0.
- **RSI mean reversion:** long/flat; enter when RSI falls below the selected entry threshold and exit when RSI rises above the exit threshold.
- **Execution timing:** signals computed from a closing price are shifted to the next session's position, avoiding same-close look-ahead.
- **Returns:** daily close-to-close index returns multiplied by the prior signal's position, minus an assumed cost times absolute position turnover.
- **Benchmark:** NIFTY 50 buy-and-hold, with no assumed entry cost.
- **Risk-free rate:** Sharpe and Sortino are calculated with a zero risk-free rate in this starter. For formal research, use a date-aligned risk-free series.
- **CAGR:** calculated using the number of scored daily observations divided by 252 as the elapsed-year approximation.

## Important limitations

1. The NIFTY 50 price index is not itself directly tradable. This simulates index exposure and does not fully model an ETF, futures contract, futures rolls, dividends, lot sizes, margin, taxes, liquidity, or market impact.
2. The cost slider is a simplified assumption applied to absolute position turnover, not a detailed Indian-market brokerage/tax calculator.
3. RSI/SMA definitions and data vendors can differ slightly. The RSI uses exponentially smoothed gains/losses as a Wilder-style approximation.
4. The app uses prior history for indicator warm-up, but the backtest accounting begins over the loaded series. The scored test year is re-normalized to the chosen initial capital.
5. Avoid choosing parameters after repeatedly inspecting test-year results. Use a validation/walk-forward process inside 2000–2021 and freeze choices before evaluating 2022–2024.
6. A benchmark comparison should use the same test dates. Consider using the NIFTY 50 Total Return Index (TRI) when comparing against a dividend-inclusive investment.
7. Trade statistics count continuous long exposures as trades and are approximate, not an exchange execution ledger.

## Project structure

```text
nifty-backtester/
├── app.py
├── data/
├── src/
│   ├── data_loader.py
│   ├── strategies.py
│   ├── backtester.py
│   └── metrics.py
├── tests/
│   └── test_backtester.py
└── requirements.txt
```

## Run tests

```bash
pytest -q
```

## Suggested next steps

- Add a validated local NIFTY 50 CSV so results are reproducible.
- Add unit tests against known hand-calculated return paths.
- Add walk-forward evaluation and parameter stability analysis.
- Add benchmark metrics such as beta, alpha, tracking error, and information ratio.
- Save run configuration alongside output CSVs for reproducibility.
