"""Discount-curve adapter.

Primary source is Project 2's bootstrapped SOFR OIS curve
(`sofr-curve/output/curve.csv`). If that file is missing or unparseable the
adapter degrades to a flat constant rate and records which source was used;
every downstream output carries a `rate_source` stamp so a reader can tell
whether a run used the real curve.

The curve is a single snapshot (Project 2's valuation date). Two things follow:

  * Term structure  -- taken from the curve, interpolated in year fraction on
    the continuously-compounded zero rate.
  * Daily level     -- the curve alone gives no history, so the level is moved
    each day by the change in the 13-week bill yield relative to its value on
    the curve's own valuation date. On that date the overlay is exactly zero
    and r(T) reproduces the bootstrapped curve.

Without the overlay Delta-r would be identically zero across the window and the
rho attribution term would be vacuous.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG

_UP = CONFIG["upstream"]


@dataclass
class DiscountCurve:
    """Zero curve plus a daily parallel-level overlay."""

    year_fractions: np.ndarray      # pillar year fractions, ascending
    zero_rates: np.ndarray          # continuously-compounded zero rates
    source: str                     # "sofr-curve" | "flat-fallback"
    asof: pd.Timestamp | None       # curve valuation date
    note: str                       # human-readable provenance line

    def base_zero(self, T: np.ndarray | float) -> np.ndarray | float:
        """Zero rate at maturity T, flat-extrapolated outside the pillar range."""
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        z = np.interp(T_arr, self.year_fractions, self.zero_rates)
        return z if np.ndim(T) else float(z[0])

    def zero(self, T: np.ndarray | float, level_shift: float = 0.0):
        """Zero rate including the day's parallel level overlay."""
        return self.base_zero(T) + level_shift

    def discount(self, T: np.ndarray | float, level_shift: float = 0.0):
        """Discount factor exp(-r(T) T)."""
        T_arr = np.asarray(T, dtype=float)
        return np.exp(-self.zero(T_arr, level_shift) * T_arr)


def load_curve(path: str | Path | None = None) -> DiscountCurve:
    """Load Project 2's curve; fall back to a flat rate on any failure."""
    path = Path(path if path is not None else _UP["sofr_curve"])
    flat = _UP["flat_rate_fallback"]
    try:
        df = pd.read_csv(path)
        need = {"year_fraction", "zero_rate_cc"}
        if not need.issubset(df.columns):
            raise ValueError(f"curve.csv missing columns {need - set(df.columns)}")
        df = df[["year_fraction", "zero_rate_cc"]].dropna()
        df = df.sort_values("year_fraction").drop_duplicates("year_fraction")
        if len(df) < 2:
            raise ValueError("curve.csv has fewer than 2 usable pillars")

        asof = None
        try:
            raw = pd.read_csv(path)
            if "segment_start" in raw.columns:
                asof = pd.to_datetime(raw["segment_start"]).min()
        except Exception:
            asof = None

        yf = df["year_fraction"].to_numpy(float)
        zr = df["zero_rate_cc"].to_numpy(float)
        note = (f"SOFR OIS curve from {path} -- {len(yf)} pillars, "
                f"{yf.min():.3f}y to {yf.max():.2f}y, "
                f"zero {zr.min():.4%} to {zr.max():.4%}"
                + (f", curve as-of {asof.date()}" if asof is not None else ""))
        return DiscountCurve(yf, zr, "sofr-curve", asof, note)

    except Exception as exc:  # noqa: BLE001 -- any failure degrades to flat
        note = (f"FLAT FALLBACK r = {flat:.4%} -- could not use {path} "
                f"({type(exc).__name__}: {exc})")
        return DiscountCurve(np.array([0.0, 100.0]),
                             np.array([flat, flat]),
                             "flat-fallback", None, note)


def build_level_overlay(rate_series: pd.Series | None,
                        anchor: pd.Timestamp | None) -> pd.Series | None:
    """Daily parallel shift (decimal) of the zero curve.

    `rate_series` is a percent-quoted short-rate history (e.g. ^IRX close).
    The shift is zero on `anchor`, the curve's own valuation date. Returns None
    if no history is available, in which case the curve is held static.
    """
    if rate_series is None or len(rate_series) == 0:
        return None
    s = rate_series.dropna().sort_index() / 100.0
    if anchor is None:
        anchor_val = float(s.iloc[0])
    else:
        idx = s.index[s.index <= anchor]
        anchor_val = float(s.loc[idx[-1]]) if len(idx) else float(s.iloc[0])
    return s - anchor_val
