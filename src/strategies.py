from __future__ import annotations
import numpy as np
import pandas as pd

def add_indicators(data: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    out = data.copy()
    close = out["Close"].astype(float)
    out["SMA_50"] = close.rolling(50, min_periods=50).mean()
    out["SMA_200"] = close.rolling(200, min_periods=200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder-style exponentially smoothed RSI.
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"RSI_{rsi_period}"] = 100 - (100 / (1 + rs))
    out.loc[(avg_loss == 0) & (avg_gain > 0), f"RSI_{rsi_period}"] = 100.0
    out.loc[(avg_loss == 0) & (avg_gain == 0), f"RSI_{rsi_period}"] = 50.0
    return out

def generate_signals(
    data: pd.DataFrame,
    strategy: str,
    rsi_period: int = 14,
    rsi_entry: int = 30,
    rsi_exit: int = 50,
) -> pd.Series:
    """Return target exposure (0 or 1) at each close; engine shifts it to next session."""
    close = data["Close"]
    if strategy == "50/200-day MA crossover":
        fast = data["SMA_50"]
        slow = data["SMA_200"]
        target = (fast > slow).astype(float)
        target[fast.isna() | slow.isna()] = 0.0
        return target.rename("signal")
    if strategy == "RSI mean reversion":
        rsi = data[f"RSI_{rsi_period}"]
        position = 0.0
        values = []
        for value in rsi:
            if not np.isfinite(value):
                values.append(position)
                continue
            if position == 0 and value < rsi_entry:
                position = 1.0
            elif position == 1 and value > rsi_exit:
                position = 0.0
            values.append(position)
        return pd.Series(values, index=data.index, name="signal", dtype=float)
    raise ValueError(f"Unknown strategy: {strategy}")
