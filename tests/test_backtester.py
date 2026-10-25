import numpy as np
import pandas as pd
from src.backtester import run_backtest
from src.metrics import performance_metrics, trade_metrics
from src.strategies import add_indicators, generate_signals

def sample_data():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    return pd.DataFrame({"Close": close})

def test_backtest_shifts_signal_one_session():
    data = pd.DataFrame({"Close": [100.0, 110.0, 121.0]}, index=pd.date_range("2024-01-01", periods=3))
    signal = pd.Series([1.0, 1.0, 0.0], index=data.index)
    result = run_backtest(data, signal, initial_capital=1000, cost_rate=0.0)
    assert result["position"].tolist() == [0.0, 1.0, 1.0]
    assert np.isclose(result["net_return"].iloc[1], 0.1)
    assert np.isclose(result["net_return"].iloc[2], 0.1)

def test_indicators_and_ma_signal_exist():
    data = add_indicators(sample_data())
    signal = generate_signals(data, "50/200-day MA crossover")
    assert len(signal) == len(data)
    assert set(signal.unique()).issubset({0.0, 1.0})

def test_metrics_have_expected_keys():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    equity = 1000 * (1 + returns).cumprod()
    metrics = performance_metrics(returns, equity, 1000)
    assert "CAGR" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
    trades = trade_metrics(pd.Series([0, 1, 1, 0]), returns)
    assert trades["Trades"] == 1
