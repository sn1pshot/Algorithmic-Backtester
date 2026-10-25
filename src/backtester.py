from __future__ import annotations
import numpy as np
import pandas as pd

def run_backtest(
    data: pd.DataFrame,
    target_position: pd.Series,
    initial_capital: float = 1_000_000,
    cost_rate: float = 0.001,
    charge_initial_entry: bool = True,
) -> pd.DataFrame:
    """
    Daily close-to-next-close portfolio simulation.
    Signals observed at close t become exposure for session t+1.
    cost_rate is the fraction charged per unit of absolute position turnover.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if cost_rate < 0:
        raise ValueError("cost_rate cannot be negative.")
    out = data[["Close"]].copy()
    signal = target_position.reindex(out.index).fillna(0.0).clip(0, 1)
    out["signal"] = signal
    out["position"] = signal.shift(1).fillna(0.0)
    if charge_initial_entry and len(out):
        # Do not charge an initial entry before the first observed signal.
        pass
    out["market_return"] = out["Close"].pct_change().fillna(0.0)
    out["gross_return"] = out["position"] * out["market_return"]
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    if not charge_initial_entry and len(out):
        out.loc[out.index[0], "turnover"] = 0.0
    out["transaction_cost"] = out["turnover"] * cost_rate
    out["net_return"] = out["gross_return"] - out["transaction_cost"]
    out["equity"] = float(initial_capital) * (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out
