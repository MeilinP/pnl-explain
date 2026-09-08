"""Sanity checks for the Phase-1 machinery. Run: python tests/test_sanity.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_market_data                      # noqa: E402
from src.config import CONFIG                                 # noqa: E402
from src.pricing import bs_greeks, bs_price, fd_gamma, fd_vanna, fd_volga  # noqa: E402
from src.rates import load_curve                              # noqa: E402
from src.surface import build_surface, load_sabr_params       # noqa: E402

CAL_DATE = "2026-08-07"
_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _fail += 1


print("put-call parity")
S, K, T, r, q, v = 700.0, 690.0, 0.75, 0.035, 0.0097, 0.18
lhs = bs_price(S, K, T, r, q, v, "C") - bs_price(S, K, T, r, q, v, "P")
rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
check("C - P = S e^-qT - K e^-rT", abs(lhs - rhs) < 1e-9, f"diff {lhs - rhs:.2e}")

print("\nanalytic vs finite-difference Greeks")
g = bs_greeks(S, K, T, r, q, v, "C")
for nm, ana, fd, tol in (("gamma", g.gamma, fd_gamma(S, K, T, r, q, v, "C"), 5e-3),
                         ("vanna", g.vanna, fd_vanna(S, K, T, r, q, v, "C"), 5e-3),
                         ("volga", g.volga, fd_volga(S, K, T, r, q, v, "C"), 5e-3)):
    rel = abs(fd - ana) / max(abs(ana), 1e-12)
    check(f"{nm} rel err < {tol:.0e}", rel < tol,
          f"analytic {ana:.6g} fd {fd:.6g} rel {rel:.2e}")

print("\nupstream adapters")
curve = load_curve(ROOT / CONFIG["upstream"]["sofr_curve"])
check("SOFR curve loads from Project 2", curve.source == "sofr-curve", curve.source)
bad = load_curve(ROOT / "does_not_exist.csv")
check("curve falls back to flat on failure", bad.source == "flat-fallback",
      f"r(5y)={bad.base_zero(5.0):.4%}")
check("flat fallback is genuinely flat",
      abs(bad.base_zero(1.0) - bad.base_zero(30.0)) < 1e-15)

ts = load_sabr_params(ROOT / CONFIG["upstream"]["sabr_params"])
check("SABR params load from Project 1", ts.source == "sabr-calibration", ts.source)
check("SABR falls back to flat on failure",
      load_sabr_params(ROOT / "nope.csv").source == "flat-fallback")

print("\nsurface anchoring (lambda = 1 on the calibration date)")
md = load_market_data(ROOT, live=False)
surface, anchor = build_surface(ts, md.realized_vol, CAL_DATE)
rv_cal = md.realized_vol.dropna().loc[:CAL_DATE].iloc[-1]
check("anchor equals realized vol on calibration date", abs(anchor - rv_cal) < 1e-12)
errs = [abs(surface.atm_vol(float(t), 1.0) - float(a))
        for t, a in zip(ts.T, ts.atm_vol)]
check("surface reproduces Project 1 ATM vols at lambda=1",
      max(errs) < 1e-12, f"max abs err {max(errs):.2e}")

print("\nSABR ATM reproduction (alpha cubic)")
from src.hagan import sabr_atm_vol                             # noqa: E402
worst = 0.0
for T_ in (0.115, 0.5, 1.0, 2.357):
    for F_ in (620.0, 700.0, 780.0):
        for lam in (0.5, 1.0, 1.31):
            a, b, rho, nu, atm = surface.params(T_, F_, lam)
            worst = max(worst, abs(sabr_atm_vol(F_, T_, a, b, rho, nu) - atm))
check("ATM vol reproduced by solved alpha", worst < 1e-10, f"max abs err {worst:.2e}")

print("\nportfolio")
from src.portfolio import bucket_of, default_portfolio, year_fraction  # noqa: E402
pf = default_portfolio()
end = pd.Timestamp(CONFIG["data"]["window_end"])
check("no position expires inside the window",
      all(pd.Timestamp(e) > end for e in pf["expiry"]))
check("book has 4 distinct structures", pf["structure"].nunique() == 4,
      ", ".join(sorted(pf["structure"].unique())))
check("bucket boundaries map correctly",
      [bucket_of(t) for t in (0.1, 0.3, 0.8, 1.5)] == ["0-3M", "3M-6M", "6M-1Y", "1Y+"])
check("calendar legs sit in different buckets at window end",
      bucket_of(year_fraction(end, "2026-10-16")) != bucket_of(year_fraction(end, "2027-06-17")))

# ==========================================================================
# Risk module -- invariants of the VaR/ES estimators and the stress framework
# ==========================================================================

from src.portfolio import MarketState, load_portfolio                 # noqa: E402
from src.rates import build_level_overlay                             # noqa: E402
from src.revalue import build_revaluer                                # noqa: E402
from src.stress import stress_grid                                    # noqa: E402
from src.var import (backtest_var, filtered_historical_var,           # noqa: E402
                     full_reval_var, historical_var, parametric_var)

ALPHAS = [0.05, 0.025, 0.01]      # 95%, 97.5%, 99% -- ascending confidence
OUT = ROOT / CONFIG["output"]["dir"]

# One market state at the end of the window: the book a risk report is about.
_win = md.window
_rv = md.realized_vol.reindex(_win).ffill()
_lam = _rv / anchor
# Same rate overlay run_risk.py uses, so the numbers printed here are the ones
# in the report rather than a second, slightly different set.
_ov = build_level_overlay(md.rate_level, curve.asof)
_ov = _ov.reindex(_win).ffill() if _ov is not None else None
_states = {d: MarketState(date=d, spot=float(md.spot.loc[d]),
                          rate_shift=float(_ov.loc[d]) if _ov is not None else 0.0,
                          vol_factor=float(_lam.loc[d]),
                          realized_vol=float(_rv.loc[d])) for d in _win}
_positions = load_portfolio(ROOT / "data" / "portfolio.csv")
rev = build_revaluer(_positions, _states, curve, surface)

print("\nrepricing adapter")
z = rev(0.0, 0.0, 0.0)
check("revalue(0, 0, 0) = 0", abs(z) < 1e-9, f"{z:.3e} on a book of {rev.base_value:,.0f}")
check("a shocked reval actually moves the book",
      abs(rev(-0.10, 0.05, 0.0)) > 1.0, f"{rev(-0.10, 0.05, 0.0):,.0f} at -10% spot / +5 vol pts")

# Realised P&L. The attribution outputs when they exist -- these are invariants
# of the estimators, so a deterministic synthetic series serves when they do not.
_att_p = OUT / "daily_attribution.csv"
if _att_p.exists():
    _att = pd.read_csv(_att_p, index_col=0, parse_dates=True)
    PNL = _att["actual"].to_numpy(float)
    DS, DSIG, DR = (_att["dS_pct"].to_numpy(float),
                    _att["dsig_vega_weighted"].to_numpy(float),
                    _att["dr"].to_numpy(float))
    _src = f"daily_attribution.csv, {len(PNL)} moves"
else:
    _rng = np.random.default_rng(11)
    PNL = _rng.standard_t(4, 255) * 25_000.0
    DS, DSIG, DR = PNL * 0.0, PNL * 0.0, PNL * 0.0
    _src = "synthetic (run.py has not been run)"

print(f"\nVaR estimators  [{_src}]")

# Historical VaR is defined as an order statistic, with no interpolation. This
# pins that definition: a future switch to np.quantile would change every
# reported number and would otherwise pass silently.
for a in ALPHAS:
    got = historical_var(PNL, alpha=a).var
    ordered = np.sort(-PNL)
    k = min(max(int(np.ceil((1.0 - a) * ordered.size)) - 1, 0), ordered.size - 1)
    check(f"historical VaR at {100 * (1 - a):g}% is order statistic k={k}",
          got == ordered[k], f"{got:,.2f} vs {ordered[k]:,.2f}")

# parametric needs a Greek frame aligned to the moves; reuse the exposure path.
_exp_p = OUT / "exposures_daily.csv"
if _exp_p.exists():
    exposures_frame = pd.read_csv(_exp_p, index_col=0, parse_dates=True).iloc[:-1]
    DS_ABS = _att["dS"].to_numpy(float)
else:
    exposures_frame = pd.DataFrame({"delta": np.full(len(PNL), 4_314.0),
                                    "gamma": np.full(len(PNL), -29.9),
                                    "vega": np.full(len(PNL), -133_800.0)})
    DS_ABS = PNL * 0.0

_estimators = {
    "historical": lambda a: historical_var(PNL, alpha=a),
    "parametric": lambda a: parametric_var(exposures_frame, DS_ABS, DSIG, alpha=a),
    "filtered_hs": lambda a: filtered_historical_var(PNL, alpha=a),
    "full_reval": lambda a: full_reval_var(rev, DS, DSIG, DR, alpha=a),
}

_results = {name: {a: fn(a) for a in ALPHAS} for name, fn in _estimators.items()}

for name, byalpha in _results.items():
    vars_ = [byalpha[a].var for a in ALPHAS]          # ALPHAS is ascending conf
    check(f"{name}: VaR monotone increasing in confidence",
          all(v2 >= v1 - 1e-9 for v1, v2 in zip(vars_, vars_[1:])),
          " <= ".join(f"{v:,.0f}" for v in vars_))

for name, byalpha in _results.items():
    worst = min(byalpha[a].es - byalpha[a].var for a in ALPHAS)
    check(f"{name}: ES >= VaR at every confidence", worst >= -1e-9,
          f"min(ES - VaR) = {worst:,.2f}")

print("\nbacktest")
# Kupiec's LR is a likelihood ratio against the null that the exception rate is
# alpha. When the observed rate IS alpha the two likelihoods coincide, so the
# statistic is exactly zero -- constructed here rather than approached.
_n, _a = 100, 0.05
_losses = np.array([20.0] * 5 + [5.0] * 95)
_bt = backtest_var(-_losses, np.full(_n, 10.0), alpha=_a)
check("Kupiec LR = 0 when exceptions equal expectation",
      _bt.n_exceptions == 5 and abs(_bt.expected - 5.0) < 1e-12
      and abs(_bt.kupiec_lr) < 1e-12,
      f"x={_bt.n_exceptions}, expected={_bt.expected:.2f}, LR={_bt.kupiec_lr:.3e}")

print("\nstress grid")
_spot = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
_vol = np.array([-0.05, 0.0, 0.05])
_grid = stress_grid(rev, _spot, _vol)
_i0, _j0 = int(np.argmin(np.abs(_vol))), int(np.argmin(np.abs(_spot)))
_cell = float(_grid.to_numpy(float)[_i0, _j0])
check("stress grid (0, 0) cell = 0", abs(_cell) < 1e-9, f"{_cell:.3e}")
check("stress grid has no missing cells", bool(np.isfinite(_grid.to_numpy(float)).all()))

print(f"\n{'ALL CHECKS PASSED' if _fail == 0 else f'{_fail} CHECK(S) FAILED'}")
sys.exit(1 if _fail else 0)
