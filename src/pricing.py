"""Black-Scholes pricing and Greeks for European options on a dividend-paying
underlying.

Conventions used throughout, so the attribution arithmetic is unambiguous:

  * `vega`  is dV/d(sigma) per 1.00 of vol (100 vol points). Reported per vol
    point where a human reads it, but the raw Greek is per 1.00 so that
    PnL_vega = vega * d(sigma) works with sigma in decimals.
  * `theta` is dV/dt with t running FORWARD in calendar time (negative for a
    long option), per year. PnL_theta = theta * dt with dt in years.
  * `rho`   is dV/dr per 1.00 of rate. PnL_rho = rho * dr with dr in decimals.
  * All Greeks are per share. Position scaling by quantity * multiplier happens
    in portfolio.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from .config import CONFIG

_G = CONFIG["greeks"]

OptionType = Literal["C", "P"]


@dataclass
class Greeks:
    """Per-share value and risk sensitivities."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float      # d2V/dS d(sigma)
    volga: float      # d2V/d(sigma)^2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def names() -> list[str]:
        return [f.name for f in fields(Greeks)]


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float
           ) -> tuple[float, float]:
    vs = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vs
    return float(d1), float(d1 - vs)


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             opt: OptionType) -> float:
    """Black-Scholes price. T <= 0 or sigma <= 0 collapses to the intrinsic value."""
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = (S - K) if opt == "C" else (K - S)
        return float(max(intrinsic, 0.0))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if opt == "C":
        return float(S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2))
    return float(K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float,
              opt: OptionType) -> Greeks:
    """Analytic Black-Scholes Greeks.

    vanna and volga have closed forms and are computed here as the reference
    values; the spec's finite-difference versions live in greeks.py and are
    cross-checked against these (see tests/test_sanity.py).
    """
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max((S - K) if opt == "C" else (K - S), 0.0)
        itm = float(S > K) if opt == "C" else -float(S < K)
        return Greeks(intrinsic, itm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrtT = np.sqrt(T)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = float(norm.pdf(d1))

    gamma = df_q * pdf / (S * sigma * sqrtT)
    vega = S * df_q * pdf * sqrtT
    vanna = -df_q * pdf * d2 / sigma
    volga = vega * d1 * d2 / sigma

    if opt == "C":
        delta = df_q * float(norm.cdf(d1))
        rho = K * T * df_r * float(norm.cdf(d2))
        theta = (-S * df_q * pdf * sigma / (2.0 * sqrtT)
                 - r * K * df_r * float(norm.cdf(d2))
                 + q * S * df_q * float(norm.cdf(d1)))
    else:
        delta = -df_q * float(norm.cdf(-d1))
        rho = -K * T * df_r * float(norm.cdf(-d2))
        theta = (-S * df_q * pdf * sigma / (2.0 * sqrtT)
                 + r * K * df_r * float(norm.cdf(-d2))
                 - q * S * df_q * float(norm.cdf(-d1)))

    return Greeks(bs_price(S, K, T, r, q, sigma, opt), delta, gamma, vega,
                  float(theta), rho, vanna, volga)


# --------------------------------------------------------------------------
# Finite-difference cross-Greeks (the spec's prescribed method)
# --------------------------------------------------------------------------

def fd_vanna(S: float, K: float, T: float, r: float, q: float, sigma: float,
             opt: OptionType, hS: float | None = None,
             hv: float | None = None) -> float:
    """d2V/dS d(sigma) by a central cross difference.

    Bumps: 1% of spot, 1 vol point (config `greeks.spot_bump_rel` / `vol_bump`).
    """
    hS = S * _G["spot_bump_rel"] if hS is None else hS
    hv = _G["vol_bump"] if hv is None else hv
    pp = bs_price(S + hS, K, T, r, q, sigma + hv, opt)
    pm = bs_price(S + hS, K, T, r, q, sigma - hv, opt)
    mp = bs_price(S - hS, K, T, r, q, sigma + hv, opt)
    mm = bs_price(S - hS, K, T, r, q, sigma - hv, opt)
    return float((pp - pm - mp + mm) / (4.0 * hS * hv))


def fd_volga(S: float, K: float, T: float, r: float, q: float, sigma: float,
             opt: OptionType, hv: float | None = None) -> float:
    """d2V/d(sigma)^2 by a central second difference in vol."""
    hv = _G["vol_bump"] if hv is None else hv
    up = bs_price(S, K, T, r, q, sigma + hv, opt)
    mid = bs_price(S, K, T, r, q, sigma, opt)
    dn = bs_price(S, K, T, r, q, sigma - hv, opt)
    return float((up - 2.0 * mid + dn) / (hv * hv))


def fd_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float,
             opt: OptionType, hS: float | None = None) -> float:
    """d2V/dS^2 by a central second difference -- cross-check on the analytic gamma."""
    hS = S * _G["spot_bump_rel"] if hS is None else hS
    up = bs_price(S + hS, K, T, r, q, sigma, opt)
    mid = bs_price(S, K, T, r, q, sigma, opt)
    dn = bs_price(S - hS, K, T, r, q, sigma, opt)
    return float((up - 2.0 * mid + dn) / (hS * hS))
