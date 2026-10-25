from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def performance_metrics(daily_returns: pd.Series, equity: pd.Series, initial_capital: float) -> dict:
    r = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    eq = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty or eq.empty:
        return {k: np.nan for k in ["Total Return", "CAGR", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Best Day", "Worst Day"]}
    total_return = float(eq.iloc[-1] / initial_capital - 1)
    elapsed_years = max(len(r) / TRADING_DAYS, 1 / TRADING_DAYS)
    cagr = float((eq.iloc[-1] / initial_capital) ** (1 / elapsed_years) - 1) if eq.iloc[-1] > 0 else -1.0
    volatility = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    daily_std = r.std(ddof=1)
    sharpe = float(np.sqrt(TRADING_DAYS) * r.mean() / daily_std) if pd.notna(daily_std) and daily_std > 0 else np.nan
    downside = r[r < 0]
    downside_std = float(np.sqrt((downside ** 2).mean())) if not downside.empty else 0.0
    sortino = float(np.sqrt(TRADING_DAYS) * r.mean() / downside_std) if downside_std > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Best Day": float(r.max()),
        "Worst Day": float(r.min()),
    }

def trade_metrics(position: pd.Series, daily_returns: pd.Series) -> dict:
    pos = pd.Series(position).fillna(0).astype(float)
    r = pd.Series(daily_returns).reindex(pos.index).fillna(0.0)
    entries = (pos > 0) & (pos.shift(1).fillna(0) <= 0)
    exits = (pos <= 0) & (pos.shift(1).fillna(0) > 0)
    # A trade is a continuous period with exposure. Approximate trade return from daily net returns.
    trade_returns = []
    active = False
    acc = 1.0
    for p, ret in zip(pos, r):
        if p > 0:
            if not active:
                active, acc = True, 1.0
            acc *= 1 + float(ret)
        elif active:
            trade_returns.append(acc - 1)
            active = False
    if active:
        trade_returns.append(acc - 1)
    wins = [x for x in trade_returns if x > 0]
    losses = [x for x in trade_returns if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "Trades": int(entries.sum()),
        "Win Rate": float(len(wins) / len(trade_returns)) if trade_returns else np.nan,
        "Average Trade Return": float(np.mean(trade_returns)) if trade_returns else np.nan,
        "Profit Factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "Average Holding Days": float(pos.sum() / max(int(entries.sum()), 1)) if entries.sum() else np.nan,
    }

def yearly_returns(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).resample("YE").prod() - 1
