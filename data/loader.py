"""Market-data loader: SPY spot path, short-rate path, realized volatility.

yfinance is a live source, so every pull is cached to `data/raw/`. A rerun uses
the cache and reproduces every number in the README exactly; `--live` forces a
re-pull.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG

_D = CONFIG["data"]
_M = CONFIG["model"]

_SPOT_FILE = "spy_spot_history.csv"
_RATE_FILE = "short_rate_history.csv"
_META_FILE = "snapshot_meta.json"


@dataclass
class MarketData:
    """Daily market state over the PnL window, plus pre-window burn-in."""

    spot: pd.Series          # full history incl. burn-in, index = date
    rate_level: pd.Series    # short-rate proxy in percent, full history
    realized_vol: pd.Series  # EWMA annualised realized vol, full history
    window: pd.DatetimeIndex  # valuation dates inside the PnL window
    source: str              # "cache" | "live"

    @property
    def window_spot(self) -> pd.Series:
        return self.spot.reindex(self.window)


def _raw_dir(root: Path) -> Path:
    d = root / _D["snapshot_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pull_yf(symbol: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    idx = h.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(h["Close"].to_numpy(float), index=idx.normalize(), name=symbol)
    return s[~s.index.duplicated(keep="last")].sort_index()


def ewma_realized_vol(spot: pd.Series, lam: float | None = None) -> pd.Series:
    """RiskMetrics EWMA of squared log returns, annualised at 252 days.

    EWMA rather than a rolling window: a gap day must move the vol level on the
    day it happens, not 21 days later and then again when it drops out. The
    rolling alternative injects a spurious second vol move at the drop-out date
    that no attribution term can explain.
    """
    lam = _M["rv_lambda"] if lam is None else lam
    r = np.log(spot).diff().dropna()
    var = r.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
    rv = np.sqrt(var * 252.0)
    rv.iloc[: _M["rv_min_periods"]] = np.nan   # burn-in
    return rv.rename("realized_vol")


def gjr_realized_vol(spot: pd.Series, lam: float | None = None,
                     gamma: float | None = None) -> pd.Series:
    """Leverage-asymmetric EWMA (GJR-style) of squared log returns.

        var_t = lam * var_{t-1} + (1 - lam) * (1 + gamma * sign_down_t) * r_t^2

    where sign_down_t is +1 on a down day and -1 on an up day. Symmetric
    returns give an average weight of 1, so this variant shares an
    unconditional vol level with `ewma_realized_vol` and differs only in the
    ASYMMETRY of the response.

    Motivation: the symmetric EWMA is blind to the sign of the return, so a
    +2.9% rally raises the modelled vol exactly as a -2.9% selloff would.
    Equity vol does the opposite. Since vanna PnL is vanna * dS * d(sigma),
    the symmetric variant mis-signs the vanna contribution on up-gap days.
    """
    lam = _M["rv_lambda"] if lam is None else lam
    gamma = _M["rv_gamma"] if gamma is None else gamma
    r = np.log(spot).diff().dropna()
    weight = 1.0 + gamma * np.where(r.to_numpy() < 0.0, 1.0, -1.0)
    shock = pd.Series(weight * r.to_numpy() ** 2, index=r.index)
    var = shock.ewm(alpha=1.0 - lam, adjust=False).mean()
    rv = np.sqrt(var.clip(lower=1e-12) * 252.0)
    rv.iloc[: _M["rv_min_periods"]] = np.nan
    return rv.rename("realized_vol_gjr")


def realized_vol(spot: pd.Series, dynamics: str = "ewma") -> pd.Series:
    """Dispatch on the vol-dynamics variant name."""
    if dynamics == "ewma":
        return ewma_realized_vol(spot)
    if dynamics == "gjr":
        return gjr_realized_vol(spot)
    raise ValueError(f"unknown vol dynamics {dynamics!r}")


def load_market_data(root: Path, live: bool = False) -> MarketData:
    """Load (or pull and cache) the spot and short-rate histories."""
    raw = _raw_dir(root)
    spot_p, rate_p, meta_p = raw / _SPOT_FILE, raw / _RATE_FILE, raw / _META_FILE

    if not live and spot_p.exists() and rate_p.exists():
        spot = pd.read_csv(spot_p, index_col=0, parse_dates=True).iloc[:, 0]
        rate = pd.read_csv(rate_p, index_col=0, parse_dates=True).iloc[:, 0]
        source = "cache"
    else:
        end = (pd.Timestamp(_D["window_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        spot = _pull_yf(_D["underlying"], _D["history_start"], end)
        try:
            rate = _pull_yf(_D["rate_symbol"], _D["history_start"], end)
        except Exception:                      # noqa: BLE001
            rate = pd.Series(dtype=float, name=_D["rate_symbol"])
        spot.to_csv(spot_p)
        rate.to_csv(rate_p)
        meta_p.write_text(json.dumps({
            "underlying": _D["underlying"],
            "rate_symbol": _D["rate_symbol"],
            "history_start": _D["history_start"],
            "window": [_D["window_start"], _D["window_end"]],
            "pulled_utc": pd.Timestamp.utcnow().isoformat(),
            "n_spot_rows": int(len(spot)),
            "n_rate_rows": int(len(rate)),
        }, indent=2))
        source = "live"

    spot.name, rate.name = "spot", "rate_pct"
    window = spot.loc[_D["window_start"]:_D["window_end"]].index
    return MarketData(spot=spot, rate_level=rate,
                      realized_vol=ewma_realized_vol(spot),
                      window=pd.DatetimeIndex(window), source=source)


def gap_days(spot: pd.Series, window: pd.DatetimeIndex,
             threshold: float | None = None) -> pd.DataFrame:
    """Simple-return moves inside the window exceeding `threshold` in absolute value."""
    threshold = _D["gap_threshold"] if threshold is None else threshold
    w = spot.reindex(window)
    ret = w.pct_change()
    log_ret = np.log(w).diff()
    df = pd.DataFrame({"spot": w, "ret": ret, "log_ret": log_ret}).dropna()
    out = df[df["ret"].abs() > threshold].copy()
    out["prev_spot"] = w.shift(1).reindex(out.index)
    return out[["prev_spot", "spot", "ret", "log_ret"]]
