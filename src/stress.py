"""
Stress testing for the option book.

Three families, because they answer different questions:

  historical    -- replay a dated episode's joint (spot, vol, rate) move.
                   Answers "what if that happened again".
  hypothetical  -- a grid of spot x vol shocks with no historical precedent.
                   Answers "where is this book actually fragile".
  reverse       -- solve for the smallest shock that produces a stated loss.
                   Answers "what would it take to lose X", which is the only
                   one of the three a desk can act on directly.

Every scenario is applied by FULL REPRICING, not by Taylor expansion. The
whole point of a stress test is to move far enough that the Greeks themselves
have changed, so a Greek-based stress number is close to meaningless at the
sizes that matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


# --------------------------------------------------------------------------
# scenario definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    ds_rel: float        # relative spot move, e.g. -0.20 for -20%
    dvol_abs: float      # absolute vol points, e.g. +0.25 for +25 vol points
    dr_abs: float        # absolute rate move, e.g. -0.0150 for -150bp
    horizon_days: int
    source: str


# Historical episodes. Moves are the widely reported index-level figures for
# each episode, rounded; they are the shape of the event, not a tick-accurate
# replay. Anyone rerunning this with vendor data should expect differences at
# the margin and should not read the second digit as meaningful.
HISTORICAL = [
    Scenario(
        "Black Monday 1987",
        ds_rel=-0.204, dvol_abs=+0.50, dr_abs=-0.0100, horizon_days=1,
        source="S&P 500 one-day close-to-close, 19 Oct 1987",
    ),
    Scenario(
        "Lehman / GFC 2008",
        ds_rel=-0.30, dvol_abs=+0.45, dr_abs=-0.0150, horizon_days=20,
        source="S&P 500 and VIX, Sep-Nov 2008",
    ),
    Scenario(
        "Volmageddon 2018",
        ds_rel=-0.041, dvol_abs=+0.20, dr_abs=-0.0010, horizon_days=1,
        source="S&P 500 and VIX, 5 Feb 2018",
    ),
    Scenario(
        "COVID crash 2020",
        ds_rel=-0.34, dvol_abs=+0.60, dr_abs=-0.0150, horizon_days=23,
        source="S&P 500 and VIX, 19 Feb - 23 Mar 2020",
    ),
    Scenario(
        "2022 rates repricing",
        ds_rel=-0.25, dvol_abs=+0.15, dr_abs=+0.0400, horizon_days=252,
        source="S&P 500 and 2Y UST, calendar 2022",
    ),
    Scenario(
        "Aug 2024 carry unwind",
        ds_rel=-0.03, dvol_abs=+0.38, dr_abs=-0.0020, horizon_days=1,
        source="S&P 500 and VIX, 5 Aug 2024",
    ),
]

# Hypothetical scenarios. These exist because the historical set has a
# structural blind spot: every episode above pairs a spot fall with a vol
# rise. A book that is long vol and short spot looks safe against all six and
# can still be badly exposed to a slow grind up with vol collapsing.
HYPOTHETICAL = [
    Scenario("Spot flat, vol -30%", 0.00, -0.30, 0.0, 1, "hypothetical"),
    Scenario("Spot +10%, vol -10 pts", +0.10, -0.10, 0.0, 5, "hypothetical"),
    Scenario("Spot +15%, vol +15 pts", +0.15, +0.15, 0.0, 5,
             "hypothetical: correlation breakdown, up-crash"),
    Scenario("Spot -10%, vol unchanged", -0.10, 0.00, 0.0, 5,
             "hypothetical: sell-off with no vol bid"),
    Scenario("Rates +200bp, spot flat", 0.00, 0.00, +0.0200, 20, "hypothetical"),
]


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

def run_scenarios(
    revalue: Callable[[float, float, float], float],
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Apply each scenario by full repricing.

    `revalue(ds_rel, dvol_abs, dr_abs) -> P&L in dollars`.
    """
    rows = []
    for s in scenarios:
        pnl = float(revalue(s.ds_rel, s.dvol_abs, s.dr_abs))
        rows.append({
            "scenario": s.name,
            "dS_rel": f"{s.ds_rel:+.1%}",
            "dVol_pts": f"{100 * s.dvol_abs:+.0f}",
            "dRate_bp": f"{10000 * s.dr_abs:+.0f}",
            "horizon_d": s.horizon_days,
            "P&L": round(pnl, 2),
            "source": s.source,
        })
    df = pd.DataFrame(rows).sort_values("P&L").reset_index(drop=True)
    return df


def stress_grid(
    revalue: Callable[[float, float, float], float],
    spot_shocks: np.ndarray | None = None,
    vol_shocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full spot x vol repricing grid.

    A single worst-case number tells you the depth of the hole but not where
    it is. The grid shows the shape -- and on a book with sign-flipping gamma
    the worst cell is frequently interior, not at a corner, which is exactly
    the case a corner-only scenario set misses.
    """
    if spot_shocks is None:
        spot_shocks = np.arange(-0.30, 0.3001, 0.05)
    if vol_shocks is None:
        vol_shocks = np.arange(-0.20, 0.4001, 0.05)

    grid = np.zeros((len(vol_shocks), len(spot_shocks)))
    for i, dv in enumerate(vol_shocks):
        for j, dsr in enumerate(spot_shocks):
            grid[i, j] = revalue(float(dsr), float(dv), 0.0)

    return pd.DataFrame(
        grid,
        index=[f"vol{100 * v:+.0f}" for v in vol_shocks],
        columns=[f"spot{100 * s:+.0f}%" for s in spot_shocks],
    )


def reverse_stress(
    revalue: Callable[[float, float, float], float],
    target_loss: float,
    spot_range: tuple[float, float] = (-0.40, 0.40),
    vol_range: tuple[float, float] = (-0.30, 0.60),
    n: int = 121,
) -> dict:
    """Smallest shock, by Euclidean distance in (spot%, vol-point) space,
    that produces at least `target_loss`.

    Distance metric is stated because it is a choice, not a fact: a 10% spot
    move and 10 vol points are treated as equally far, which is defensible for
    a short-dated equity book and is not defensible in general. Change the
    weights and the answer moves.
    """
    ss = np.linspace(spot_range[0], spot_range[1], n)
    vv = np.linspace(vol_range[0], vol_range[1], n)

    best = None
    for dv in vv:
        for dsr in ss:
            pnl = revalue(float(dsr), float(dv), 0.0)
            if pnl <= -abs(target_loss):
                dist = np.hypot(dsr * 100.0, dv * 100.0)
                if best is None or dist < best["distance"]:
                    best = {
                        "target_loss": -abs(target_loss),
                        "dS_rel": float(dsr),
                        "dVol_abs": float(dv),
                        "P&L": float(pnl),
                        "distance": float(dist),
                    }

    if best is None:
        return {
            "target_loss": -abs(target_loss),
            "reachable": False,
            "note": "no shock inside the search box produces this loss",
        }
    best["reachable"] = True
    return best
