"""
Value-at-Risk and Expected Shortfall for the option book.

Four estimators on the same book and the same window, so the disagreement
between them is the output rather than a nuisance:

  historical   -- empirical quantile of the realised daily P&L series
  parametric   -- normal quantile from Greek-implied P&L variance (delta-normal,
                  optionally with the gamma correction)
  filtered     -- historical simulation on EWMA-devolatilised returns, rescaled
                  to today's vol (Barone-Adesi filtered historical simulation)
  full-reval   -- full repricing of every leg under each historical scenario

The first three are approximations of the fourth. On an option book with sign-
flipping gamma the gap is not a rounding error, and reporting only one number
hides it.

Inputs come from the attribution pipeline already in this repo:
    exposures_daily.csv   -- daily Greeks per date (delta, gamma, vega, theta, rho)
    daily_attribution.csv -- realised daily P&L, dS, dsigma, dr

Conventions match src/config.py: P&L in dollars, sigma in absolute vol points,
r in absolute rate points, one-day horizon unless stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Sequence


# --------------------------------------------------------------------------
# results container
# --------------------------------------------------------------------------

@dataclass
class RiskResult:
    method: str
    alpha: float                 # tail probability, e.g. 0.01 for 99%
    horizon_days: int
    var: float                   # positive number = loss
    es: float                    # positive number = loss
    n_obs: int
    n_effective: float           # obs actually informing the tail
    note: str = ""

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "confidence": f"{100 * (1 - self.alpha):g}%",
            "horizon_days": self.horizon_days,
            "VaR": round(self.var, 2),
            "ES": round(self.es, 2),
            "n_obs": self.n_obs,
            "n_eff_tail": round(self.n_effective, 1),
            "note": self.note,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _tail(losses: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Return (VaR, ES, n_effective) from a loss sample. Losses positive."""
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        raise ValueError("empty loss sample")

    # Order-statistic VaR. No interpolation: on a 255-day sample the
    # interpolated 99% quantile sits between two points that are themselves
    # 1-in-255 events, and the interpolation invents precision the sample
    # does not have.
    k = int(np.ceil((1.0 - alpha) * losses.size)) - 1
    k = min(max(k, 0), losses.size - 1)
    ordered = np.sort(losses)
    var = ordered[k]

    tail = ordered[k:]
    es = float(tail.mean())
    return float(var), es, float(tail.size)


def _normal_quantile(alpha: float) -> float:
    """Inverse standard normal CDF at 1 - alpha, via erfinv."""
    from scipy.special import erfinv
    return float(np.sqrt(2.0) * erfinv(2.0 * (1.0 - alpha) - 1.0))


def _scale_horizon(x: float, horizon_days: int) -> float:
    """Square-root-of-time scaling.

    Valid only under iid returns and a locally linear book. Both fail here:
    option Greeks change as spot moves, so a 10-day move is not 10 one-day
    moves applied to the same delta. Reported anyway because regulators ask
    for it, and flagged in the note field so the reader knows what it assumes.
    """
    return x * np.sqrt(horizon_days)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def historical_var(
    pnl: Sequence[float],
    alpha: float = 0.01,
    horizon_days: int = 1,
) -> RiskResult:
    """Empirical quantile of realised daily P&L."""
    losses = -np.asarray(pnl, dtype=float)
    var, es, n_eff = _tail(losses, alpha)

    note = ""
    if n_eff < 5:
        note = f"only {n_eff:.0f} obs in the tail; ES is not a stable estimate"
    if horizon_days > 1:
        var, es = _scale_horizon(var, horizon_days), _scale_horizon(es, horizon_days)
        note = (note + "; " if note else "") + "sqrt-time scaled, assumes iid"

    return RiskResult("historical", alpha, horizon_days, var, es,
                      len(losses), n_eff, note)


def parametric_var(
    exposures: pd.DataFrame,
    ds: Sequence[float],
    dsigma: Sequence[float],
    alpha: float = 0.01,
    horizon_days: int = 1,
    include_gamma: bool = True,
) -> RiskResult:
    """Delta-normal VaR, optionally with the gamma correction.

    P&L ~= delta*dS + 0.5*gamma*dS^2 + vega*dsigma

    The gamma term is what makes this interesting: on a book whose gamma
    changes sign, the quadratic term is not a small positive add-on, and a
    pure delta-normal number can sit on either side of the truth.
    """
    ds = np.asarray(ds, dtype=float)
    dsigma = np.asarray(dsigma, dtype=float)

    delta = exposures["delta"].to_numpy(dtype=float)
    vega = exposures["vega"].to_numpy(dtype=float)
    gamma = exposures["gamma"].to_numpy(dtype=float) if include_gamma else None

    n = min(len(ds), len(delta))
    pnl = delta[:n] * ds[:n] + vega[:n] * dsigma[:n]
    if gamma is not None:
        pnl = pnl + 0.5 * gamma[:n] * ds[:n] ** 2

    sigma = float(np.std(pnl, ddof=1))
    z = _normal_quantile(alpha)
    var = z * sigma
    # Normal ES closed form: sigma * phi(z) / alpha
    phi = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
    es = sigma * phi / alpha

    note = "normal assumption; " + (
        "delta + gamma + vega" if include_gamma else "delta + vega only"
    )
    if horizon_days > 1:
        var, es = _scale_horizon(var, horizon_days), _scale_horizon(es, horizon_days)
        note += "; sqrt-time scaled"

    return RiskResult("parametric", alpha, horizon_days, var, es, n, float(n), note)


def filtered_historical_var(
    pnl: Sequence[float],
    alpha: float = 0.01,
    horizon_days: int = 1,
    lam: float = 0.94,
) -> RiskResult:
    """Filtered historical simulation (Barone-Adesi).

    Devolatilise the P&L series by its EWMA vol, then rescale the standardised
    residuals to today's vol level. Plain historical simulation treats a
    quiet-regime day and a crisis day as equally informative about tomorrow;
    this does not.
    """
    x = np.asarray(pnl, dtype=float)
    if x.size < 20:
        raise ValueError("need at least 20 observations to filter")

    var_ewma = np.empty_like(x)
    var_ewma[0] = float(np.var(x[: min(20, x.size)], ddof=1))
    for t in range(1, x.size):
        var_ewma[t] = lam * var_ewma[t - 1] + (1.0 - lam) * x[t - 1] ** 2
    vol = np.sqrt(var_ewma)
    vol[vol <= 0] = np.nan

    z = x / vol
    z = z[np.isfinite(z)]
    scaled = z * vol[-1]

    losses = -scaled
    var, es, n_eff = _tail(losses, alpha)

    note = f"EWMA lambda={lam}; current vol {vol[-1]:,.0f}"
    if horizon_days > 1:
        var, es = _scale_horizon(var, horizon_days), _scale_horizon(es, horizon_days)
        note += "; sqrt-time scaled"

    return RiskResult("filtered_hs", alpha, horizon_days, var, es,
                      int(z.size), n_eff, note)


def full_reval_var(
    revalue: Callable[[float, float, float], float],
    ds_rel: Sequence[float],
    dsigma: Sequence[float],
    dr: Sequence[float],
    alpha: float = 0.01,
    horizon_days: int = 1,
) -> RiskResult:
    """Historical simulation with full repricing.

    `revalue(ds_rel, dsigma, dr) -> P&L` must reprice every leg from scratch
    under the shocked state, not Taylor-expand it. This is the benchmark the
    other three estimators are approximating.
    """
    ds_rel = np.asarray(ds_rel, dtype=float)
    dsigma = np.asarray(dsigma, dtype=float)
    dr = np.asarray(dr, dtype=float)

    n = min(len(ds_rel), len(dsigma), len(dr))
    pnl = np.array([revalue(ds_rel[i], dsigma[i], dr[i]) for i in range(n)])

    losses = -pnl
    var, es, n_eff = _tail(losses, alpha)

    note = "full repricing under historical shocks"
    if horizon_days > 1:
        var, es = _scale_horizon(var, horizon_days), _scale_horizon(es, horizon_days)
        note += "; sqrt-time scaled"

    return RiskResult("full_reval", alpha, horizon_days, var, es, n, n_eff, note)


# --------------------------------------------------------------------------
# backtest -- Kupiec POF and Christoffersen independence
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    n: int
    n_exceptions: int
    expected: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    basel_zone: str
    verdict: str = field(default="")


def backtest_var(
    pnl: Sequence[float],
    var_series: Sequence[float],
    alpha: float = 0.01,
) -> BacktestResult:
    """Two-part VaR backtest.

    Kupiec POF tests whether the *number* of exceptions matches alpha.
    Christoffersen tests whether they are *independent* -- a model can have
    exactly the right count and still be broken if every exception lands in
    the same week, which is the failure mode that actually costs money.
    """
    from scipy.stats import chi2

    losses = -np.asarray(pnl, dtype=float)
    v = np.asarray(var_series, dtype=float)
    n = min(losses.size, v.size)
    hits = (losses[:n] > v[:n]).astype(int)

    x = int(hits.sum())
    p = alpha
    expected = n * p

    # Kupiec proportion-of-failures
    if 0 < x < n:
        pi = x / n
        lr_pof = -2.0 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - ((n - x) * np.log(1 - pi) + x * np.log(pi))
        )
    else:
        lr_pof = 0.0
    p_pof = float(1.0 - chi2.cdf(lr_pof, df=1))

    # Christoffersen independence
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    lr_ind = 0.0
    if (n00 + n01) > 0 and (n10 + n11) > 0 and (n01 + n11) > 0:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)
        if 0 < pi01 < 1 and 0 < pi11 < 1 and 0 < pi_all < 1:
            ll_uncond = (n00 + n10) * np.log(1 - pi_all) + (n01 + n11) * np.log(pi_all)
            ll_cond = (
                n00 * np.log(1 - pi01) + n01 * np.log(pi01)
                + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
            )
            lr_ind = -2.0 * (ll_uncond - ll_cond)
    p_ind = float(1.0 - chi2.cdf(lr_ind, df=1))

    # Basel traffic light, scaled from its 250-day / 99% definition
    scaled = x * 250.0 / max(n, 1)
    zone = "green" if scaled <= 4 else ("yellow" if scaled <= 9 else "red")

    verdict = []
    if p_pof < 0.05:
        verdict.append("exception count rejects the model")
    if p_ind < 0.05:
        verdict.append("exceptions are clustered")
    if not verdict:
        verdict.append("no rejection on either test")
    if n < 250:
        verdict.append(
            f"n={n} is below the 250-day Basel window; the count test has "
            "low power here and a green zone is weak evidence"
        )

    return BacktestResult(
        n=n, n_exceptions=x, expected=expected,
        kupiec_lr=float(lr_pof), kupiec_p=p_pof,
        christoffersen_lr=float(lr_ind), christoffersen_p=p_ind,
        basel_zone=zone, verdict="; ".join(verdict),
    )
