"""Vega bucketing by bump-and-reprice.

A single portfolio vega number is not actionable. This book makes the point
concretely: at 2026-04-07 its bucket vegas are roughly +30k / -42k / +32k per
vol point across 3M-6M / 6M-1Y / 1Y+, which nearly cancel. The single number
says the book is close to vega-flat; the ladder says it holds a large calendar
position that a non-parallel vol move will hit hard.

Method (per spec): bump the vol surface by one vol point inside one expiry
bucket, hold the other buckets fixed, reprice the whole book, and divide by the
bump. This is a bump-and-reprice vega, not a sum of analytic leg vegas -- the
two differ by the second-order vol convexity (volga) inside the bucket, and
that difference is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG
from .portfolio import MarketState, price_portfolio
from .rates import DiscountCurve
from .surface import VolSurface

_B = CONFIG["buckets"]
_BUMP = CONFIG["greeks"]["surface_vol_bump"]
LABELS: list[str] = _B["labels"]


def bucket_vegas(positions: pd.DataFrame, state: MarketState,
                 curve: DiscountCurve, surface: VolSurface,
                 bump: float | None = None) -> dict[str, float]:
    """Central-difference bucket vegas, per 1.00 of vol.

    Also returns:
      `total_parallel` -- all buckets bumped together, the correct total vega;
      `sum_buckets`    -- the arithmetic sum of the individual bucket vegas;
      `interaction`    -- their difference, which measures how much the buckets
                          interact through the surface. Additive bucketing is
                          exact only if the repricing is linear in the bumps.
    """
    bump = _BUMP if bump is None else bump
    out: dict[str, float] = {}
    for lab in LABELS:
        up = price_portfolio(positions, state, curve, surface, {lab: +bump})
        dn = price_portfolio(positions, state, curve, surface, {lab: -bump})
        out[lab] = (up - dn) / (2.0 * bump)

    all_up = price_portfolio(positions, state, curve, surface,
                             {lab: +bump for lab in LABELS})
    all_dn = price_portfolio(positions, state, curve, surface,
                             {lab: -bump for lab in LABELS})
    out["total_parallel"] = (all_up - all_dn) / (2.0 * bump)
    out["sum_buckets"] = float(sum(out[lab] for lab in LABELS))
    out["interaction"] = out["total_parallel"] - out["sum_buckets"]
    return out


def bucket_vega_path(positions: pd.DataFrame, states: dict, curve: DiscountCurve,
                     surface: VolSurface, dates=None) -> pd.DataFrame:
    """Bucket vegas for every date in the window."""
    dates = list(states.keys()) if dates is None else list(dates)
    rows = []
    for d in dates:
        bv = bucket_vegas(positions, states[d], curve, surface)
        rows.append({"date": d, "spot": states[d].spot,
                     "vol_factor": states[d].vol_factor, **bv})
    df = pd.DataFrame(rows).set_index("date")
    return df


def bucket_atm_moves(surface: VolSurface, lam0: float, lam1: float
                     ) -> dict[str, float]:
    """ATM vol change per bucket, evaluated at each bucket's representative maturity.

    These are the d(sigma) inputs the bucketed attribution multiplies its bucket
    vegas by -- the observable a desk would read off its own vol marks.
    """
    tenors = CONFIG["experiments"]["bucket_tenors"]
    return {lab: surface.atm_vol(float(tenors[lab]), lam1)
                 - surface.atm_vol(float(tenors[lab]), lam0)
            for lab in LABELS}


def single_atm_move(surface: VolSurface, lam0: float, lam1: float) -> float:
    """The one-number vol move: change in ATM vol at the benchmark tenor."""
    T = float(CONFIG["experiments"]["single_vega_tenor"])
    return surface.atm_vol(T, lam1) - surface.atm_vol(T, lam0)
