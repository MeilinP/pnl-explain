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

print(f"\n{'ALL CHECKS PASSED' if _fail == 0 else f'{_fail} CHECK(S) FAILED'}")
sys.exit(1 if _fail else 0)
