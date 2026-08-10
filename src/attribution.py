"""Taylor decomposition of daily PnL into Greek contributions.

Ground truth is full revaluation:

    PnL_actual = V(S1, sigma1, r1, t1) - V(S0, sigma0, r0, t0)

Explanation is a second-order Taylor expansion with every Greek evaluated at
t0. All state variables are per-leg: each option has its own implied vol from
the SABR surface and its own discount rate off the SOFR curve, and both change
partly because the market moved and partly because the option got one day
shorter. Both effects are genuine changes in that leg's state, so both belong
in d(sigma) and dr; the T-decay at FIXED sigma and r is what theta covers, so
there is no double count.

Three axes are varied, one at a time, so each experiment is single-factor:

  order      first  (delta, vega, theta, rho)
             second (adds gamma, vanna, volga)
  delta      bs | sabr | bartlett
  vega_mode  single | bucketed | exact

DELTA VARIANTS AND THE DOUBLE-COUNT. A SABR delta already contains the vol move
the smile produces when spot moves:

    delta_SABR = delta_BS + vega * d(sigma)/dS

so charging the vega term with the FULL realised d(sigma) on top of it would
count the spot-induced slide twice. The SABR variants therefore use

    d(sigma)_eff = d(sigma) - (d(sigma)/dS) * dS

Once that is done the two variants agree to first order in dS and differ only
through the curvature of sigma(S) -- i.e. only on large moves. That
near-degeneracy is the honest result and is reported as such; the sharper test
of a delta is hedge-P&L variance (`hedge_effectiveness`), which is the test
Bartlett's delta was introduced to win.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .bucketing import LABELS, bucket_atm_moves, bucket_vegas, single_atm_move
from .config import CONFIG
from .portfolio import MarketState, value_portfolio
from .rates import DiscountCurve
from .surface import VolSurface

DAYCOUNT = CONFIG["data"]["day_count"]

Order = Literal["first", "second"]
DeltaFlavor = Literal["bs", "sabr", "bartlett"]
VegaMode = Literal["single", "bucketed", "exact"]

FIRST_ORDER = ["delta", "vega", "theta", "rho"]
SECOND_ORDER = FIRST_ORDER + ["gamma", "vanna", "volga"]

_DELTA_COL = {"bs": "delta", "sabr": "delta_sabr", "bartlett": "delta_bartlett"}
_SLIDE_COL = {"bs": None, "sabr": "dsig_dS", "bartlett": "dsig_dS_bartlett"}


@dataclass(frozen=True)
class Spec:
    order: Order = "second"
    delta: DeltaFlavor = "bs"
    vega_mode: VegaMode = "exact"

    @property
    def name(self) -> str:
        return f"{self.order}|{self.delta}|{self.vega_mode}"

    @property
    def components(self) -> list[str]:
        return FIRST_ORDER if self.order == "first" else SECOND_ORDER


class Engine:
    """Caches per-date valuations so every Spec reuses one set of revaluations."""

    def __init__(self, positions: pd.DataFrame, states: dict, curve: DiscountCurve,
                 surface: VolSurface, dates: Iterable | None = None):
        self.positions = positions
        self.states: dict = states
        self.curve = curve
        self.surface = surface
        self.dates = list(states.keys()) if dates is None else list(dates)
        self._legs: dict = {}
        self._bvega: dict = {}

    def legs(self, d) -> pd.DataFrame:
        if d not in self._legs:
            self._legs[d] = value_portfolio(self.positions, self.states[d],
                                            self.curve, self.surface)
        return self._legs[d]

    def bucket_vegas(self, d) -> dict[str, float]:
        if d not in self._bvega:
            self._bvega[d] = bucket_vegas(self.positions, self.states[d],
                                          self.curve, self.surface)
        return self._bvega[d]

    def book_value(self, d) -> float:
        return float(self.legs(d)["value"].sum())

    # ------------------------------------------------------------------ core
    def run(self, spec: Spec) -> pd.DataFrame:
        rows = []
        for d0, d1 in zip(self.dates[:-1], self.dates[1:]):
            rows.append(self._one_day(d0, d1, spec))
        df = pd.DataFrame(rows).set_index("date")

        comps = spec.components
        df["explained"] = df[[f"pnl_{c}" for c in comps]].sum(axis=1)
        df["unexplained"] = df["actual"] - df["explained"]
        denom = df["actual"].abs().replace(0.0, np.nan)
        df["unexplained_pct_of_actual"] = df["unexplained"] / denom
        df.attrs["spec"] = spec.name
        return df

    def _one_day(self, d0, d1, spec: Spec) -> dict:
        l0, l1 = self.legs(d0), self.legs(d1)
        s0, s1 = self.states[d0], self.states[d1]

        dS = s1.spot - s0.spot
        dt = float((d1 - d0).days) / DAYCOUNT
        actual = float(l1["value"].sum() - l0["value"].sum())

        dsig_leg = (l1["iv"] - l0["iv"]).to_numpy(float)
        dr_leg = (l1["rate"] - l0["rate"]).to_numpy(float)

        # --- delta, and the vol move that survives it ------------------------
        delta_book = float(l0[_DELTA_COL[spec.delta]].sum())
        pnl_delta = delta_book * dS
        slide_col = _SLIDE_COL[spec.delta]
        if slide_col is None:
            dsig_eff = dsig_leg
        else:
            dsig_eff = dsig_leg - l0[slide_col].to_numpy(float) * dS

        # --- vega, at the requested granularity ------------------------------
        if spec.vega_mode == "exact":
            pnl_vega = float(np.sum(l0["vega"].to_numpy(float) * dsig_eff))
            dsig_used = dsig_eff
        elif spec.vega_mode == "bucketed":
            bv = self.bucket_vegas(d0)
            moves = bucket_atm_moves(self.surface, s0.vol_factor, s1.vol_factor)
            pnl_vega = float(sum(bv[b] * moves[b] for b in LABELS))
            dsig_used = l0["bucket"].map(moves).to_numpy(float)
        else:  # single
            bv = self.bucket_vegas(d0)
            move = single_atm_move(self.surface, s0.vol_factor, s1.vol_factor)
            pnl_vega = float(bv["total_parallel"] * move)
            dsig_used = np.full(len(l0), move)

        row = {
            "date": d1, "spot_0": s0.spot, "spot_1": s1.spot, "dS": dS,
            "dS_pct": dS / s0.spot, "dt": dt,
            "vol_factor_0": s0.vol_factor, "vol_factor_1": s1.vol_factor,
            # Weighted by |vega|, not signed vega: the book crosses vega-flat
            # nine times in this window, and a signed-vega denominator explodes
            # there (one day printed a -84 vol-point "move").
            "dsig_vega_weighted": float(
                np.sum(np.abs(l0["vega"].to_numpy(float)) * dsig_leg)
                / max(float(np.abs(l0["vega"].to_numpy(float)).sum()), 1e-12)),
            "dsig_atm_3m": single_atm_move(self.surface, s0.vol_factor, s1.vol_factor),
            "dr": float(np.mean(dr_leg)),
            "value_0": float(l0["value"].sum()), "value_1": float(l1["value"].sum()),
            "actual": actual,
            "book_delta": float(l0["delta"].sum()),
            "book_gamma": float(l0["gamma"].sum()),
            "book_vega": float(l0["vega"].sum()),
            "book_vanna": float(l0["vanna"].sum()),
            "book_volga": float(l0["volga"].sum()),
            "pnl_delta": pnl_delta,
            "pnl_gamma": 0.5 * float(l0["gamma"].sum()) * dS * dS,
            "pnl_vega": pnl_vega,
            "pnl_theta": float(l0["theta"].sum()) * dt,
            "pnl_rho": float(np.sum(l0["rho"].to_numpy(float) * dr_leg)),
            "pnl_vanna": float(np.sum(l0["vanna"].to_numpy(float) * dsig_used)) * dS,
            "pnl_volga": 0.5 * float(np.sum(l0["volga"].to_numpy(float)
                                            * dsig_used ** 2)),
        }
        return row


# --------------------------------------------------------------------------
# Summaries and experiment drivers
# --------------------------------------------------------------------------

def summarise(df: pd.DataFrame, label: str) -> dict:
    """Headline residual statistics for one attribution run."""
    unexp, act = df["unexplained"], df["actual"]
    abs_act_total = float(act.abs().sum())
    ss_res = float(((act - df["explained"]) ** 2).sum())
    ss_tot = float(((act - act.mean()) ** 2).sum())
    return {
        "variant": label,
        "n_days": int(len(df)),
        "total_actual": float(act.sum()),
        "total_explained": float(df["explained"].sum()),
        "mean_abs_unexplained": float(unexp.abs().mean()),
        "std_unexplained": float(unexp.std(ddof=1)),
        "max_abs_unexplained": float(unexp.abs().max()),
        "worst_day": str(unexp.abs().idxmax().date()),
        "mean_abs_unexp_pct_of_abs_pnl": float(unexp.abs().sum() / abs_act_total)
        if abs_act_total else np.nan,
        "r2_explained_vs_actual": 1.0 - ss_res / ss_tot if ss_tot else np.nan,
        "corr_explained_actual": float(df["explained"].corr(act)),
    }


def improvement(base: dict, better: dict, key: str = "mean_abs_unexplained") -> float:
    """Percentage reduction in `key` going from `base` to `better`."""
    b = base[key]
    return float((b - better[key]) / b) if b else np.nan


def hedge_effectiveness(engine: Engine) -> pd.DataFrame:
    """Variance of daily PnL after delta-hedging with each delta variant.

    hedged_t = PnL_actual - delta(t0) * dS. This is the test that discriminates
    between deltas: an attribution can always absorb a bad delta into its vega
    term, but a hedge cannot.
    """
    rows = []
    for d0, d1 in zip(engine.dates[:-1], engine.dates[1:]):
        l0 = engine.legs(d0)
        dS = engine.states[d1].spot - engine.states[d0].spot
        actual = engine.book_value(d1) - engine.book_value(d0)
        rows.append({"date": d1, "actual": actual, "dS": dS,
                     **{f"hedged_{k}": actual - float(l0[c].sum()) * dS
                        for k, c in _DELTA_COL.items()}})
    df = pd.DataFrame(rows).set_index("date")
    out = []
    for k in _DELTA_COL:
        h = df[f"hedged_{k}"]
        out.append({"delta_variant": k, "std_hedged_pnl": float(h.std(ddof=1)),
                    "mean_abs_hedged_pnl": float(h.abs().mean()),
                    "var_ratio_vs_unhedged": float(h.var(ddof=1)
                                                   / df["actual"].var(ddof=1))})
    res = pd.DataFrame(out)
    base = float(res.loc[res["delta_variant"] == "bs", "std_hedged_pnl"].iloc[0])
    res["std_reduction_vs_bs"] = (base - res["std_hedged_pnl"]) / base
    res["std_unhedged"] = float(df["actual"].std(ddof=1))
    return res.set_index("delta_variant"), df
