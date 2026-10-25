from __future__ import annotations
import pandas as pd

def load_csv(file_or_path) -> pd.DataFrame:
    """Load daily market data from CSV. Accepts Date/Datetime and Close/Adj Close."""
    df = pd.read_csv(file_or_path)
    date_col = next((c for c in df.columns if c.strip().lower() in {"date", "datetime", "timestamp"}), None)
    if date_col is None:
        raise ValueError("CSV must include a Date or Datetime column.")
    close_col = next((c for c in df.columns if c.strip().lower() in {"close", "adj close", "adj_close", "price"}), None)
    if close_col is None:
        raise ValueError("CSV must include a Close or Adj Close column.")
    rename = {date_col: "Date", close_col: "Close"}
    for candidate in ["Open", "High", "Low", "Volume"]:
        found = next((c for c in df.columns if c.strip().lower() == candidate.lower()), None)
        if found:
            rename[found] = candidate
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    return df

def download_nifty(start="2000-01-01", end="2025-01-01") -> pd.DataFrame:
    """Download NIFTY 50 daily history from Yahoo Finance using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
    frame = yf.download("^NSEI", start=start, end=end, auto_adjust=False, progress=False)
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no rows. Upload a CSV from a trusted data source.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "Date"
    if "Adj Close" in frame.columns and "Close" not in frame.columns:
        frame["Close"] = frame["Adj Close"]
    return frame

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a daily close series."""
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date")
    if "Close" not in out.columns:
        raise ValueError("Data must contain a Close column.")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    if len(out) < 250:
        raise ValueError("At least 250 valid daily observations are required.")
    if out.empty:
        raise ValueError("No valid positive Close prices found.")
    return out
