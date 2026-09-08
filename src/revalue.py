"""Full-repricing adapter for the risk module.

`var.full_reval_var` and everything in `stress.py` take one callback:

    revalue(ds_rel, dvol_abs, dr_abs) -> P&L in dollars

This module builds it out of the machinery that already values the book, so the
risk numbers and the attribution numbers come from the same pricer:

    portfolio.price_portfolio  -- the eight legs, valued leg by leg
    surface.VolSurface         -- SABR vols, alpha re-solved per (T, F, lambda)
    pricing.bs_price           -- Black-Scholes on each leg
    rates.DiscountCurve        -- the bootstrapped SOFR curve plus level overlay

The shock is applied to the STATE, not to the value:

    S     -> S * (1 + ds_rel)
    sigma -> sigma + dvol_abs      on every leg's own vol mark
    r(T)  -> r(T) + dr_abs         parallel across the curve

and the book is then repriced from scratch. Nothing is Taylor-expanded. That is
the entire point of the module: at stress sizes the Greeks have themselves moved,
so a Greek-based stress number and a repriced one disagree, and the disagreement
is the finding rather than an error to be tuned away.

Three things this adapter does NOT do, stated here so they are not mistaken for
oversights:

  * The shock is instantaneous. `t` does not advance, so no theta accrues, and a
    scenario's `horizon_days` is metadata describing how long the move took in
    the world -- it does not enter the repricing.
  * The vol shock is a parallel level shift, which is the model's single vol
    risk factor. Section 4.1 of docs/MODEL_VALIDATION.md quantifies why that is
    the binding limitation on this book.
  * `surface.vol` clips the shocked vol into [vol_floor, vol_cap] from CONFIG.
    A large negative vol shock therefore saturates at the 3% floor rather than
    going through it, which makes deep vol-down cells less severe than an
    unclipped model would report.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .portfolio import MarketState, price_portfolio
from .rates import DiscountCurve
from .surface import VolSurface

_B = CONFIG["buckets"]


class _CachedSurface(VolSurface):
    """VolSurface with a memo on `params`.

    `params` re-solves the ATM alpha cubic through `np.roots` on every call, and
    the stress grid and reverse stress test hit the same (T, F, lambda) triples
    tens of thousands of times -- for a fixed rate shock the forward depends only
    on the spot shock, so one column of the grid shares a single alpha across
    every vol shock in it.

    The cache is keyed on the exact float triple and `params` is a pure function
    of it, so this changes runtime and nothing else. No result moves by a bit.
    """

    def __init__(self, base: VolSurface) -> None:
        super().__init__(base.ts, base.rv_anchor, base.kappa)
        self._memo: dict[tuple[float, float, float], tuple] = {}

    def params(self, T: float, F: float, lam: float):
        key = (float(T), float(F), float(lam))
        hit = self._memo.get(key)
        if hit is None:
            hit = super().params(T, F, lam)
            self._memo[key] = hit
        return hit


class Revaluer:
    """Callable `(ds_rel, dvol_abs, dr_abs) -> P&L` by full repricing.

    The base state is one dated snapshot of the book. Every shock is measured
    against that snapshot's value, so `revalue(0, 0, 0)` is identically zero.
    """

    def __init__(self, positions: pd.DataFrame, base_state: MarketState,
                 curve: DiscountCurve, surface: VolSurface) -> None:
        self.positions = positions.reset_index(drop=True)
        self.base_state = base_state
        self.curve = curve
        self.surface = surface if isinstance(surface, _CachedSurface) \
            else _CachedSurface(surface)
        self.base_value = float(price_portfolio(self.positions, base_state,
                                                curve, self.surface))
        self.n_calls = 0

    # ------------------------------------------------------------------ core
    def shocked_state(self, ds_rel: float, dr_abs: float) -> MarketState:
        """The base state with spot and the curve level moved."""
        return replace(self.base_state,
                       spot=self.base_state.spot * (1.0 + float(ds_rel)),
                       rate_shift=self.base_state.rate_shift + float(dr_abs))

    def book_value(self, ds_rel: float, dvol_abs: float, dr_abs: float) -> float:
        """Book value under the shocked state."""
        # A parallel vol bump is expressed as the same shift in every expiry
        # bucket. `price_portfolio` looks each leg's bucket up and adds the
        # shift to that leg's own SABR vol, so covering all four labels applies
        # the bump to all eight legs and to nothing else.
        shifts = {label: float(dvol_abs) for label in _B["labels"]}
        self.n_calls += 1
        return float(price_portfolio(self.positions,
                                     self.shocked_state(ds_rel, dr_abs),
                                     self.curve, self.surface, shifts))

    def __call__(self, ds_rel: float, dvol_abs: float, dr_abs: float) -> float:
        return self.book_value(ds_rel, dvol_abs, dr_abs) - self.base_value

    # ------------------------------------------------------------- reporting
    def taylor(self, ds_rel: float, dvol_abs: float, dr_abs: float,
               greeks: pd.Series) -> float:
        """Second-order Greek approximation of the same shock.

        Not used to produce any risk number -- it exists so `run_risk.py` can
        report the gap against `__call__` and show how far the Taylor expansion
        has drifted at stress sizes.
        """
        dS = self.base_state.spot * float(ds_rel)
        return float(greeks["delta"] * dS
                     + 0.5 * greeks["gamma"] * dS ** 2
                     + greeks["vega"] * float(dvol_abs)
                     + greeks["rho"] * float(dr_abs)
                     + greeks["vanna"] * dS * float(dvol_abs)
                     + 0.5 * greeks["volga"] * float(dvol_abs) ** 2)


def build_revaluer(positions: pd.DataFrame, states: dict, curve: DiscountCurve,
                   surface: VolSurface,
                   asof: pd.Timestamp | str | None = None) -> Revaluer:
    """Revaluer anchored on one valuation date.

    `asof` defaults to the last date in `states` -- the book as it stands at the
    end of the attribution window, which is the book a risk report is about.
    """
    dates = sorted(states)
    if asof is None:
        day = dates[-1]
    else:
        day = pd.Timestamp(asof)
        if day not in states:
            prior = [d for d in dates if d <= day]
            if not prior:
                raise ValueError(f"no market state on or before {day.date()}")
            day = prior[-1]
    return Revaluer(positions, states[day], curve, surface)


def sanity_check(rev: Revaluer) -> dict[str, float]:
    """Invariants a caller can assert on. Mirrors tests/test_sanity.py."""
    zero = rev(0.0, 0.0, 0.0)
    up = rev(+0.01, 0.0, 0.0)
    dn = rev(-0.01, 0.0, 0.0)
    return {
        "zero_shock_pnl": zero,
        "base_value": rev.base_value,
        "up_1pct": up,
        "down_1pct": dn,
        # (up + down) / 2 is one half of gamma * dS^2 plus higher order, so a
        # non-zero value here is the convexity the linear Greeks would miss.
        "convexity_1pct": 0.5 * (up + dn),
    }


__all__ = ["Revaluer", "build_revaluer", "sanity_check"]
