"""Portfolio definition and daily valuation.

The book is deliberately Greek-diverse (spec: a delta-one book gives a
trivially clean attribution and demonstrates nothing):

  * long straddle      -- long gamma, long vega, short theta
  * short strangle     -- short gamma and short vega, opposite sign to the above
  * call spread        -- gamma changes sign across the spot range
  * calendar spread    -- long/short vega in DIFFERENT expiry buckets, which is
                          the only structure that makes bucketed vega differ
                          from a single portfolio vega number

Strikes are round levels spanning the realised spot path (622 -> 695 -> 632 ->
773) rather than all struck at the entry spot. Struck all-at-entry, a 24%
rally leaves every position deep in the money and the book decays into delta-
one by mid-window, which would destroy exactly the gamma/vanna exposure the
attribution is meant to stress. All expiries fall after the window end, so no
position expires inside the window and no roll/expiry mechanic is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG
from .pricing import Greeks, bs_greeks, bs_price, fd_vanna, fd_volga
from .rates import DiscountCurve
from .surface import VolSurface

_P = CONFIG["portfolio"]
_D = CONFIG["data"]
_M = CONFIG["model"]
_B = CONFIG["buckets"]

MULT = _P["contract_multiplier"]
DAYCOUNT = _D["day_count"]
Q = _M["dividend_yield"]

# position_id, structure, option_type, strike, expiry, quantity (contracts)
_POSITIONS: list[tuple[str, str, str, float, str, int]] = [
    ("STRAD_C625",   "long_straddle",  "C", 625.0, "2026-09-18",  100),
    ("STRAD_P625",   "long_straddle",  "P", 625.0, "2026-09-18",  100),
    ("STRNG_P560",   "short_strangle", "P", 560.0, "2026-12-18",  -80),
    ("STRNG_C690",   "short_strangle", "C", 690.0, "2026-12-18",  -80),
    ("CSPRD_C700",   "call_spread",    "C", 700.0, "2027-03-19",  150),
    ("CSPRD_C760",   "call_spread",    "C", 760.0, "2027-03-19", -150),
    ("CAL_C650_FRT", "calendar",       "C", 650.0, "2026-10-16", -120),
    ("CAL_C650_BCK", "calendar",       "C", 650.0, "2027-06-17",  120),
]


@dataclass
class MarketState:
    """Everything needed to value the book on one date."""

    date: pd.Timestamp
    spot: float
    rate_shift: float        # parallel overlay on the zero curve, decimals
    vol_factor: float        # lambda_t = RV_t / RV_calibration
    realized_vol: float


def default_portfolio() -> pd.DataFrame:
    df = pd.DataFrame(_POSITIONS, columns=["position_id", "structure",
                                           "option_type", "strike", "expiry",
                                           "quantity"])
    df.insert(1, "underlying", _D["underlying"])
    df["entry_date"] = _P["entry_date"]
    df["multiplier"] = MULT
    return df


def load_portfolio(path: str | Path | None = None) -> pd.DataFrame:
    """Read data/portfolio.csv if present, else return (and let run.py write) the default."""
    if path is not None and Path(path).exists():
        df = pd.read_csv(path)
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")
        return df
    return default_portfolio()


def bucket_of(T: float) -> str:
    """Vega bucket label for a remaining maturity in years."""
    edges, labels = _B["edges"], _B["labels"]
    idx = int(np.searchsorted(edges, T, side="right") - 1)
    return labels[int(np.clip(idx, 0, len(labels) - 1))]


def year_fraction(valuation: pd.Timestamp, expiry: str) -> float:
    return float((pd.Timestamp(expiry) - pd.Timestamp(valuation)).days) / DAYCOUNT


def value_position(row: pd.Series, state: MarketState, curve: DiscountCurve,
                   surface: VolSurface, vol_shift: float = 0.0,
                   spot_override: float | None = None) -> dict[str, Any]:
    """Price one position and compute its scaled Greeks.

    `vol_shift` is a parallel vol bump in decimals applied to this option's
    implied vol -- the mechanism the bucketed-vega engine uses. `spot_override`
    reprices at a different spot without touching anything else.
    """
    S = state.spot if spot_override is None else float(spot_override)
    K = float(row["strike"])
    opt = str(row["option_type"])
    T = year_fraction(state.date, row["expiry"])
    scale = float(row["quantity"]) * float(row.get("multiplier", MULT))

    r = float(curve.zero(T, state.rate_shift))
    F = S * np.exp((r - Q) * T)
    sigma = float(surface.vol(K, F, T, state.vol_factor, vol_shift=vol_shift))

    g = bs_greeks(S, K, T, r, Q, sigma, opt)

    # Spec: cross-Greeks by central finite difference (1% spot, 1 vol point).
    vanna = fd_vanna(S, K, T, r, Q, sigma, opt)
    volga = fd_volga(S, K, T, r, Q, sigma, opt)

    # SABR delta = BS delta + vega * d(sigma)/dS along the smile.
    dsig_dF = surface.dvol_dF(K, F, T, state.vol_factor)
    dF_dS = float(np.exp((r - Q) * T))
    delta_sabr = g.delta + g.vega * dsig_dF * dF_dS

    # Bartlett's delta additionally propagates the alpha move implied by the
    # SABR correlation: d(alpha) = rho * nu * F^(-beta) * dF.
    alpha, beta, rho, nu, atm = surface.params(T, F, state.vol_factor)
    dsig_dalpha = surface.dvol_dalpha(K, F, T, state.vol_factor)
    dalpha_dF = rho * nu / (F ** beta)
    dsig_dF_bart = dsig_dF + dsig_dalpha * dalpha_dF
    delta_bartlett = g.delta + g.vega * dsig_dF_bart * dF_dS

    return {
        "position_id": row["position_id"], "structure": row["structure"],
        "option_type": opt, "strike": K, "expiry": row["expiry"],
        "quantity": int(row["quantity"]), "T": T, "bucket": bucket_of(T),
        "spot": S, "forward": F, "rate": r, "iv": sigma, "atm_vol": atm,
        "alpha": alpha, "rho_sabr": rho, "nu_sabr": nu,
        "price": g.price, "value": g.price * scale,
        "delta": g.delta * scale, "gamma": g.gamma * scale,
        "vega": g.vega * scale, "theta": g.theta * scale, "rho": g.rho * scale,
        "vanna": vanna * scale, "volga": volga * scale,
        "vanna_analytic": g.vanna * scale, "volga_analytic": g.volga * scale,
        "delta_sabr": delta_sabr * scale, "delta_bartlett": delta_bartlett * scale,
        "dsig_dS": dsig_dF * dF_dS, "dsig_dS_bartlett": dsig_dF_bart * dF_dS,
    }


def price_portfolio(positions: pd.DataFrame, state: MarketState,
                    curve: DiscountCurve, surface: VolSurface,
                    bucket_shifts: dict[str, float] | None = None) -> float:
    """Book value only -- no Greeks. The hot path for bump-and-reprice."""
    total = 0.0
    for _, row in positions.iterrows():
        K, opt = float(row["strike"]), str(row["option_type"])
        T = year_fraction(state.date, row["expiry"])
        shift = float(bucket_shifts.get(bucket_of(T), 0.0)) if bucket_shifts else 0.0
        r = float(curve.zero(T, state.rate_shift))
        F = state.spot * np.exp((r - Q) * T)
        sigma = float(surface.vol(K, F, T, state.vol_factor, vol_shift=shift))
        total += (bs_price(state.spot, K, T, r, Q, sigma, opt)
                  * float(row["quantity"]) * float(row.get("multiplier", MULT)))
    return total


def value_portfolio(positions: pd.DataFrame, state: MarketState,
                    curve: DiscountCurve, surface: VolSurface,
                    bucket_shifts: dict[str, float] | None = None,
                    spot_override: float | None = None) -> pd.DataFrame:
    """Value every position. `bucket_shifts` maps bucket label -> vol bump."""
    rows = []
    for _, row in positions.iterrows():
        shift = 0.0
        if bucket_shifts:
            T = year_fraction(state.date, row["expiry"])
            shift = float(bucket_shifts.get(bucket_of(T), 0.0))
        rows.append(value_position(row, state, curve, surface,
                                   vol_shift=shift, spot_override=spot_override))
    return pd.DataFrame(rows)


AGG_COLS = ["value", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga",
            "delta_sabr", "delta_bartlett", "vanna_analytic", "volga_analytic"]


def aggregate(legs: pd.DataFrame) -> pd.Series:
    """Book-level totals."""
    return legs[AGG_COLS].sum()


def bucket_vega(legs: pd.DataFrame) -> pd.Series:
    """Analytic vega split by expiry bucket (the fast reference; the bump-and-
    reprice version lives in bucketing.py)."""
    v = legs.groupby("bucket")["vega"].sum()
    return v.reindex(_B["labels"]).fillna(0.0)
