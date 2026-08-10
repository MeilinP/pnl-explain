"""Historical SABR volatility surface.

Project 1 calibrated SPY on a single date (2026-08-07). This module turns that
one calibration into a daily surface over the PnL window.

APPROXIMATION (stated in the README, repeated here because it is the single
most important modelling choice in the project):

  * rho(T) and nu(T) -- the skew and smile-convexity term structures -- are
    taken from the calibrated fit and held FIXED through time, interpolated in
    maturity onto whatever T each option has on a given date. Only one
    calibration exists, so there is no information from which to evolve their
    shape.

  * The ATM vol term structure is anchored on the calibrated ATM curve and
    scaled each day by the ratio of EWMA realized vol to its value on the
    calibration date:

        lambda_t = RV_t / RV_calibration
        sigma_ATM(t, T) = sigma_ATM_cal(T) * [1 + (lambda_t - 1) * w(T)]

    The implied-over-realized volatility risk premium cancels in the ratio, so
    only the *change* in the vol regime is transmitted -- not the level.

  * w(T) = (1 - exp(-kappa T)) / (kappa T) damps the shock along the maturity
    axis. This is the term-structure response of a mean-reverting instantaneous
    variance: a vol shock hits the front of the curve harder than the back.
    Setting kappa = 0 recovers a uniform (parallel) scaling of the ATM curve.
    Uniform scaling was rejected because it makes every expiry's vol move a
    fixed multiple of every other's, which would render the bucketed-vega
    experiment degenerate by construction.

  * alpha is re-solved from the ATM cubic at every (date, expiry). The surface
    therefore reproduces its own ATM level exactly, and on the calibration date
    lambda = 1 and the surface reproduces Project 1's calibrated vols exactly.

What this does NOT capture: skew steepening in a selloff (rho is frozen), and
vol-of-vol regime shifts (nu is frozen). Both would require a time series of
calibrations. The consequence shows up in the residual diagnostics as
unexplained PnL that correlates with |dS| beyond what gamma and vanna absorb.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .hagan import sabr_vol, solve_alpha

_UP = CONFIG["upstream"]
_M = CONFIG["model"]
_D = CONFIG["data"]


@dataclass
class SABRTermStructure:
    """Calibrated parameter term structure from Project 1."""

    T: np.ndarray          # calibrated maturities, ascending
    atm_vol: np.ndarray
    rho: np.ndarray
    nu: np.ndarray
    beta: float
    source: str            # "sabr-calibration" | "flat-fallback"
    note: str

    def interp(self, T: float) -> tuple[float, float, float]:
        """(atm_vol, rho, nu) at maturity T.

        ATM vol is interpolated in TOTAL VARIANCE (sigma^2 T) rather than in
        vol: linear-in-variance is the interpolation that cannot create a
        calendar-arbitrage between two arbitrage-free pillars. rho and nu are
        interpolated linearly, flat-extrapolated outside the pillar range.
        """
        Tc = float(np.clip(T, self.T[0], self.T[-1]))
        w = np.interp(Tc, self.T, self.atm_vol ** 2 * self.T)
        atm = float(np.sqrt(max(w, 1e-12) / max(Tc, 1e-12)))
        rho = float(np.interp(Tc, self.T, self.rho))
        nu = float(np.interp(Tc, self.T, self.nu))
        return atm, rho, nu


def load_sabr_params(path: str | Path | None = None) -> SABRTermStructure:
    """Load Project 1's calibrated parameters; degrade to a flat surface on failure."""
    path = Path(path if path is not None else _UP["sabr_params"])
    try:
        df = pd.read_csv(path)
        need = {"T", "atm_vol", "rho", "nu", "beta"}
        if not need.issubset(df.columns):
            raise ValueError(f"missing columns {need - set(df.columns)}")
        df = df[["T", "atm_vol", "rho", "nu", "beta"]].dropna().sort_values("T")
        if len(df) < 2:
            raise ValueError("fewer than 2 calibrated expiries")
        note = (f"SABR term structure from {path} -- {len(df)} expiries, "
                f"T {df['T'].min():.3f}y to {df['T'].max():.3f}y, "
                f"ATM {df['atm_vol'].min():.2%}-{df['atm_vol'].max():.2%}, "
                f"rho {df['rho'].min():.3f}..{df['rho'].max():.3f}, "
                f"nu {df['nu'].min():.3f}..{df['nu'].max():.3f}, beta={df['beta'].iloc[0]}")
        return SABRTermStructure(
            T=df["T"].to_numpy(float), atm_vol=df["atm_vol"].to_numpy(float),
            rho=df["rho"].to_numpy(float), nu=df["nu"].to_numpy(float),
            beta=float(df["beta"].iloc[0]), source="sabr-calibration", note=note)
    except Exception as exc:                                    # noqa: BLE001
        v = _UP["flat_vol_fallback"]
        note = (f"FLAT FALLBACK sigma = {v:.2%} -- could not use {path} "
                f"({type(exc).__name__}: {exc})")
        return SABRTermStructure(T=np.array([0.01, 30.0]),
                                 atm_vol=np.array([v, v]),
                                 rho=np.zeros(2), nu=np.full(2, 1e-4),
                                 beta=_M["beta"], source="flat-fallback", note=note)


def term_weight(T: np.ndarray | float, kappa: float | None = None):
    """w(T) = (1 - exp(-kappa T)) / (kappa T); w -> 1 as kappa T -> 0."""
    kappa = _M["rv_kappa"] if kappa is None else kappa
    x = np.asarray(kappa * np.asarray(T, dtype=float))
    return np.where(np.abs(x) < 1e-8, 1.0, (1.0 - np.exp(-x)) / np.where(x == 0, 1.0, x))


class VolSurface:
    """Daily SABR surface driven by a realized-vol level factor."""

    def __init__(self, ts: SABRTermStructure, rv_anchor: float,
                 kappa: float | None = None):
        self.ts = ts
        self.rv_anchor = float(rv_anchor)
        self.kappa = _M["rv_kappa"] if kappa is None else float(kappa)

    def level_factor(self, realized_vol: float) -> float:
        """lambda_t = RV_t / RV_calibration."""
        if not np.isfinite(realized_vol) or self.rv_anchor <= 0:
            return 1.0
        return float(realized_vol) / self.rv_anchor

    def atm_vol(self, T: float, lam: float) -> float:
        """Scaled, term-damped ATM vol at maturity T."""
        base, _, _ = self.ts.interp(T)
        scaled = base * (1.0 + (lam - 1.0) * float(term_weight(T, self.kappa)))
        return float(np.clip(scaled, _M["vol_floor"], _M["vol_cap"]))

    def params(self, T: float, F: float, lam: float
               ) -> tuple[float, float, float, float, float]:
        """(alpha, beta, rho, nu, atm_vol) for one (T, F, lambda) state."""
        _, rho, nu = self.ts.interp(T)
        atm = self.atm_vol(T, lam)
        alpha = solve_alpha(F, T, atm, self.ts.beta, rho, nu)
        return alpha, self.ts.beta, rho, nu, atm

    def vol(self, K: np.ndarray | float, F: float, T: float, lam: float,
            vol_shift: float = 0.0) -> np.ndarray | float:
        """Implied vol at strike(s) K.

        `vol_shift` adds a parallel bump in vol points, used by the bucketed
        vega engine (bump one expiry bucket, hold the rest fixed).
        """
        if T <= 0:
            return 0.0
        alpha, beta, rho, nu, _ = self.params(T, F, lam)
        v = sabr_vol(K, F, T, alpha, beta, rho, nu)
        v = np.asarray(v, dtype=float) + vol_shift
        v = np.clip(v, _M["vol_floor"], _M["vol_cap"])
        return v if np.ndim(K) else float(v)

    def dvol_dF(self, K: float, F: float, T: float, lam: float,
                rel_bump: float | None = None) -> float:
        """d(sigma)/dF at fixed (alpha, rho, nu) -- the smile's forward slide.

        Central difference. This is the term that turns a Black-Scholes delta
        into a SABR delta.
        """
        h = F * (CONFIG["greeks"]["spot_bump_rel"] if rel_bump is None else rel_bump)
        alpha, beta, rho, nu, _ = self.params(T, F, lam)
        up = sabr_vol(K, F + h, T, alpha, beta, rho, nu)
        dn = sabr_vol(K, F - h, T, alpha, beta, rho, nu)
        return float((up - dn) / (2.0 * h))

    def dvol_dalpha(self, K: float, F: float, T: float, lam: float,
                    rel_bump: float = 1e-4) -> float:
        """d(sigma)/d(alpha) at fixed F -- needed for Bartlett's delta."""
        alpha, beta, rho, nu, _ = self.params(T, F, lam)
        h = alpha * rel_bump
        up = sabr_vol(K, F, T, alpha + h, beta, rho, nu)
        dn = sabr_vol(K, F, T, alpha - h, beta, rho, nu)
        return float((up - dn) / (2.0 * h))


def build_surface(ts: SABRTermStructure, realized_vol: pd.Series,
                  calibration_date: str | pd.Timestamp) -> tuple[VolSurface, float]:
    """Anchor the surface so lambda = 1 on the calibration date."""
    rv = realized_vol.dropna()
    cal = pd.Timestamp(calibration_date)
    idx = rv.index[rv.index <= cal]
    if len(idx) == 0:
        raise ValueError(f"no realized-vol observation on or before {cal.date()}")
    anchor = float(rv.loc[idx[-1]])
    return VolSurface(ts, anchor), anchor
