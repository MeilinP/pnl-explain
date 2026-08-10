"""Hagan et al. (2002) lognormal SABR approximation and the ATM alpha solve.

Vendored from Project 1 (`sabr-calibration/src/hagan.py`, `calibrate.py`) so
this project runs standalone. The formulas are identical; only the CONFIG
import differs. Project 1 remains the authority for the calibration itself --
here we consume its fitted (rho, nu) and re-solve alpha whenever the forward,
maturity or ATM level changes.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG

_ATM_TOL = CONFIG["model"]["atm_tolerance"]
_ALPHA_MIN = CONFIG["model"]["alpha_min"]


def _common_C(rho: float, nu: float) -> float:
    """The (2 - 3 rho^2) nu^2 / 24 term, shared by the ATM and non-ATM branches."""
    return (2.0 - 3.0 * rho * rho) * nu * nu / 24.0


def sabr_atm_vol(F: float, T: float, alpha: float, beta: float,
                 rho: float, nu: float) -> float:
    """Hagan ATM limit (K = F)."""
    one_mb = 1.0 - beta
    A = one_mb * one_mb * alpha * alpha / (24.0 * F ** (2.0 - 2.0 * beta))
    B = rho * beta * alpha * nu / (4.0 * F ** one_mb)
    C = _common_C(rho, nu)
    return alpha / F ** one_mb * (1.0 + (A + B + C) * T)


def sabr_vol(K: np.ndarray | float, F: float, T: float, alpha: float,
             beta: float, rho: float, nu: float) -> np.ndarray | float:
    """Black lognormal implied vol under SABR, vectorised over K.

    Strikes within `atm_tolerance` of the forward take the explicit ATM branch;
    the general formula divides by x(z), which vanishes as K -> F.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    out = np.empty_like(K_arr)

    one_mb = 1.0 - beta
    C = _common_C(rho, nu)

    atm_mask = np.abs(K_arr - F) < _ATM_TOL
    if atm_mask.any():
        out[atm_mask] = sabr_atm_vol(F, T, alpha, beta, rho, nu)

    gen = ~atm_mask
    if gen.any():
        Kg = K_arr[gen]
        FK = F * Kg
        FK_half = FK ** (one_mb / 2.0)
        log_FK = np.log(F / Kg)

        z = (nu / alpha) * FK_half * log_FK
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
        x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))

        A = one_mb * one_mb * alpha * alpha / (24.0 * FK ** one_mb)
        B = rho * beta * nu * alpha / (4.0 * FK_half)
        D = one_mb * one_mb * log_FK ** 2 / 24.0
        E = one_mb ** 4 * log_FK ** 4 / 1920.0

        numerator = alpha * z * (1.0 + (A + B + C) * T)
        denominator = FK_half * x_z * (1.0 + D + E)
        vol = numerator / denominator

        degenerate = ~np.isfinite(vol)
        if degenerate.any():
            vol[degenerate] = sabr_atm_vol(F, T, alpha, beta, rho, nu)
        out[gen] = vol

    return out if np.ndim(K) else float(out[0])


def solve_alpha(F: float, T: float, atm_vol: float, beta: float,
                rho: float, nu: float) -> float:
    """Smallest strictly positive real root of the ATM cubic in alpha.

        [(1-b)^2 T / (24 F^(2-2b))] a^3
      + [rho b nu T / (4 F^(1-b))]  a^2
      + [1 + (2-3rho^2) nu^2 T/24]  a
      - sigma_atm F^(1-b)          = 0

    Guarantees the surface reproduces its ATM level exactly at every date.
    """
    one_mb = 1.0 - beta
    c3 = one_mb * one_mb * T / (24.0 * F ** (2.0 - 2.0 * beta))
    c2 = rho * beta * nu * T / (4.0 * F ** one_mb)
    c1 = 1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * T / 24.0
    c0 = -atm_vol * F ** one_mb

    roots = np.roots([c3, c2, c1, c0])
    real = roots[np.abs(roots.imag) < 1e-10].real
    positive = real[real > _ALPHA_MIN]
    if positive.size == 0:
        return float(atm_vol * F ** one_mb)
    return float(positive.min())
