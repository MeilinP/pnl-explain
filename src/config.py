"""Central configuration for the PnL-explain engine.

Every convention, bump size, bound and window boundary lives here so no numeric
constant is buried in the source. Modules import CONFIG rather than declaring
their own literals.
"""
from __future__ import annotations

from typing import Any

CONFIG: dict[str, Any] = {
    # ---------------------------------------------------------------- data
    "data": {
        "underlying": "SPY",
        # PnL window. 2025-08-01 .. 2026-08-07 contains five |dS/S| > 2% days
        # (see output/gap_days.csv) and the Mar-2026 drawdown episode.
        "window_start": "2025-08-01",
        "window_end": "2026-08-07",
        # Realized-vol burn-in: the EWMA needs history before window_start or
        # the first weeks of the vol path are dominated by the seed value.
        "history_start": "2024-06-01",
        "snapshot_dir": "data/raw",
        "rate_symbol": "^IRX",     # 13-week bill, daily level moves for Delta-r
        "day_count": 365.25,       # calendar-time convention for option T
        "gap_threshold": 0.02,     # |dS/S| defining a "gap day"
    },

    # ------------------------------------------------------ upstream inputs
    # Project 1 and Project 2 outputs. Both are consumed through adapters in
    # src/rates.py and src/surface.py that degrade to flat inputs on failure.
    "upstream": {
        "sabr_params": "../sabr-calibration/output/calibrated_params.csv",
        "sofr_curve": "../sofr-curve/output/curve.csv",
        # Fallback used only if sofr_curve cannot be read/parsed.
        "flat_rate_fallback": 0.035,
        "flat_vol_fallback": 0.16,
    },

    # --------------------------------------------------------------- model
    "model": {
        "beta": 0.5,               # inherited from Project 1's calibration
        "atm_tolerance": 1e-12,
        "alpha_min": 1e-8,
        # Continuous dividend yield: trailing-12m SPY distributions / spot at
        # the calibration date (7.525 / 773.26). Held constant so that no
        # unattributed dividend term leaks into the residual.
        "dividend_yield": 0.00973,
        # --- surface evolution (documented approximation, see README) ---
        # Only one calibration date exists, so the surface is evolved by
        # scaling the ATM term structure with realized vol while holding the
        # rho / nu term-structure shape fixed.
        "rv_lambda": 0.94,         # RiskMetrics EWMA decay for realized vol
        "rv_min_periods": 60,      # trading days of burn-in before use
        # GJR-style leverage asymmetry. A down day's squared return enters with
        # weight (1 + gamma), an up day's with (1 - gamma); with returns roughly
        # symmetric the average weight stays 1, so the two variants share an
        # unconditional vol level and differ only in their response asymmetry.
        # gamma = 0.5 gives down moves 3x the impact of up moves.
        "rv_gamma": 0.50,
        # Variance mean reversion damping the ATM shock along T:
        #     w(T) = (1 - exp(-kappa T)) / (kappa T)
        # kappa = 0 recovers a uniform (parallel) scaling of the ATM curve.
        "rv_kappa": 4.0,
        "vol_floor": 0.03,
        "vol_cap": 1.50,
    },

    # ------------------------------------------------------------ experiments
    "experiments": {
        # Sensitivity sweeps. The baseline is always listed first.
        "kappa_grid": [4.0, 2.0, 8.0],
        "vol_dynamics": ["ewma", "gjr"],
        # Representative ATM maturity for the single-vega attribution: the one
        # number a desk quotes when it says "vol moved a point".
        "single_vega_tenor": 0.25,
        # Bucket-representative maturities (midpoints; 1Y+ uses a nominal 1.5y).
        "bucket_tenors": {"0-3M": 0.125, "3M-6M": 0.375, "6M-1Y": 0.75, "1Y+": 1.5},
        "worst_days": 10,
    },

    # ------------------------------------------------------------ portfolio
    "portfolio": {
        "contract_multiplier": 100,
        "entry_date": "2025-08-01",
    },

    # -------------------------------------------------------------- greeks
    "greeks": {
        # Spec: central finite differences for cross-Greeks.
        "spot_bump_rel": 0.01,     # 1% of spot
        "vol_bump": 0.01,          # 1 vol point
        "rate_bump": 1e-4,         # 1bp, for rho by FD cross-check
        "surface_vol_bump": 0.01,  # bucket bump size, 1 vol point
    },

    # -------------------------------------------------------- vega buckets
    # Boundaries in years. An option is assigned to the bucket containing its
    # remaining maturity on the valuation date, so exposure migrates down the
    # ladder as the window advances.
    "buckets": {
        "edges": [0.0, 0.25, 0.50, 1.00, 1e9],
        "labels": ["0-3M", "3M-6M", "6M-1Y", "1Y+"],
    },

    # ---------------------------------------------------------- diagnostics
    "frtb": {
        "mean_ratio_limit": 0.10,      # |mean(unexplained)| / std(actual)
        "variance_ratio_limit": 0.20,  # var(unexplained) / var(actual)
        "spearman_floor": 0.80,        # reported alongside; Basel's amber/green
    },

    # -------------------------------------------------------------- output
    "output": {
        "dir": "output",
        "dpi": 140,
        "bp": 1e4,
    },
}
