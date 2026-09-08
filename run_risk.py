"""Project 3 (risk module) -- VaR, ES and stress testing for the option book.

    python run_risk.py                  # full risk run (VaR + backtest + stress)
    python run_risk.py --stage var      # estimators, backtest and sensitivity only
    python run_risk.py --live           # re-pull yfinance instead of using the cache

Consumes the attribution pipeline's outputs (`output/daily_attribution.csv` and
`output/exposures_daily.csv`), so `run.py` must have been run first. The
full-revaluation estimator and every stress scenario reprice the book through
`src/revalue.py` rather than Taylor-expanding it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load_market_data, realized_vol                # noqa: E402
from src.config import CONFIG                                         # noqa: E402
from src.diagnostics import GRID, INK, INK_MUTED, SURFACE             # noqa: E402
from src.portfolio import (MarketState, aggregate, load_portfolio,    # noqa: E402
                           value_portfolio)
from src.rates import build_level_overlay, load_curve                 # noqa: E402
from src.revalue import build_revaluer                                # noqa: E402
from src.stress import (HISTORICAL, HYPOTHETICAL, reverse_stress,     # noqa: E402
                        run_scenarios, stress_grid)
from src.surface import build_surface, load_sabr_params               # noqa: E402
from src.var import (backtest_var, filtered_historical_var,           # noqa: E402
                     full_reval_var, historical_var, parametric_var)

OUT = ROOT / CONFIG["output"]["dir"]
DATA = ROOT / "data"
CAL_DATE = "2026-08-07"          # Project 1's calibration date; lambda = 1 here
BASE_KAPPA = float(CONFIG["experiments"]["kappa_grid"][0])

# ---------------------------------------------------------------- risk config
# Kept here rather than in src/config.py so the attribution engine's config is
# untouched by the risk module.
CONFIDENCES = [0.95, 0.975, 0.99]          # reported confidence levels
ALPHAS = [round(1.0 - c, 4) for c in CONFIDENCES]
HORIZONS = [1, 10]                          # trading days
BACKTEST_WINDOW = 125                       # rolling-window length, trading days
LAMBDA_GRID = [0.90, 0.94, 0.97]            # EWMA decay sweep; 0.94 is baseline
BASE_LAMBDA = 0.94
REVERSE_TARGETS = [100_000.0, 250_000.0, 500_000.0, 1_000_000.0]


def _shock_axis(lo: float, hi: float, step: float) -> np.ndarray:
    """Shock grid with the arange epsilon cleaned off.

    `np.arange(-0.20, 0.4001, 0.05)` lands on -1.4e-17 instead of 0, which
    formats as a "-0" label and makes the zero cell impossible to address by
    name. Rounded and snapped so the grid has an exact, findable origin.
    """
    a = np.round(np.arange(lo, hi + step / 2.0, step), 10)
    a[np.abs(a) < 1e-12] = 0.0
    return a


SPOT_SHOCKS = _shock_axis(-0.30, 0.30, 0.05)
VOL_SHOCKS = _shock_axis(-0.20, 0.40, 0.05)

# The four estimators in a FIXED reporting order, which is also the fixed colour
# order in the figure. Never re-sorted by magnitude -- colour follows the
# estimator, not its rank.
METHODS = ["historical", "parametric", "filtered_hs", "full_reval"]
METHOD_COLORS = {"historical": "#2a78d6", "parametric": "#eb6834",
                 "filtered_hs": "#1baf7a", "full_reval": "#eda100"}


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _money(x: float) -> str:
    return f"${x:,.0f}"


# ==========================================================================
# Stage 0 -- inputs
# ==========================================================================

def load_inputs(live: bool = False) -> dict:
    """Attribution outputs, the market state path, and the repricing callback."""
    att_p, exp_p = OUT / "daily_attribution.csv", OUT / "exposures_daily.csv"
    for p in (att_p, exp_p):
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(ROOT)} -- run `python run.py` first")

    att = pd.read_csv(att_p, index_col=0, parse_dates=True)
    exp = pd.read_csv(exp_p, index_col=0, parse_dates=True)

    # Row i of daily_attribution is the move from date i to date i+1, and its
    # book_* columns are the Greeks at the START of that move. So the exposure
    # rows that pair with it are exposures_daily minus its last date. Asserted
    # rather than assumed -- a silent one-day shift here would bias every
    # parametric number and would not show up as an error anywhere else.
    exp_start = exp.iloc[:-1]
    if len(exp_start) != len(att):
        raise SystemExit(f"alignment: {len(exp_start)} exposure rows vs {len(att)} moves")
    drift = float(np.max(np.abs(exp_start["delta"].to_numpy() - att["book_delta"].to_numpy())))
    if drift > 1e-6:
        raise SystemExit(f"alignment: exposures/attribution book delta differs by {drift:.3e}")

    md = load_market_data(ROOT, live=live)
    win = md.window
    curve = load_curve(ROOT / CONFIG["upstream"]["sofr_curve"])
    ts = load_sabr_params(ROOT / CONFIG["upstream"]["sabr_params"])
    overlay = build_level_overlay(md.rate_level, curve.asof)

    rv_full = realized_vol(md.spot, "ewma")
    surface, anchor = build_surface(ts, rv_full, CAL_DATE)
    surface.kappa = BASE_KAPPA
    rv = rv_full.reindex(win).ffill()
    lam = rv / anchor
    ov = overlay.reindex(win).ffill() if overlay is not None else None
    states = {d: MarketState(date=d, spot=float(md.spot.loc[d]),
                             rate_shift=float(ov.loc[d]) if ov is not None else 0.0,
                             vol_factor=float(lam.loc[d]),
                             realized_vol=float(rv.loc[d])) for d in win}

    positions = load_portfolio(DATA / "portfolio.csv")
    rev = build_revaluer(positions, states, curve, surface)
    legs = value_portfolio(positions, rev.base_state, curve, surface)
    greeks = aggregate(legs)

    _hr("1. INPUTS")
    print(f"attribution     : {att_p.name}  ({len(att)} daily moves, "
          f"{att.index[0].date()} -> {att.index[-1].date()})")
    print(f"exposures       : {exp_p.name}  ({len(exp)} valuation dates)")
    print(f"discounting     : [{curve.source}] {curve.note}")
    print(f"vol surface     : [{ts.source}] {ts.note}")
    print(f"\nrisk snapshot   : {rev.base_state.date.date()}  "
          f"spot {rev.base_state.spot:.2f}  lambda {rev.base_state.vol_factor:.3f}  "
          f"rate overlay {rev.base_state.rate_shift:+.2%}")
    print(f"book value      : {_money(rev.base_value)}")
    print(f"book greeks     : delta {greeks['delta']:>10,.0f} sh   "
          f"gamma {greeks['gamma']:>8,.1f}   vega/pt {greeks['vega'] / 100:>10,.0f}")
    print(f"                  theta/day {greeks['theta'] / 365.25:>9,.0f}   "
          f"rho/bp {greeks['rho'] / 1e4:>8,.0f}   "
          f"vanna/pt {greeks['vanna'] / 100:>8,.1f}")
    chk = rev(0.0, 0.0, 0.0)
    print(f"revalue(0,0,0)  : {chk:.6e}   (must be 0 -- the repricing identity)")

    return {"att": att, "exp": exp, "exp_start": exp_start, "rev": rev,
            "greeks": greeks, "positions": positions, "curve": curve,
            "surface": surface, "states": states, "md": md}


# ==========================================================================
# Stage 1 -- the four estimators
# ==========================================================================

def estimator_table(ctx: dict, lam: float) -> pd.DataFrame:
    """All four estimators at every confidence and horizon, for one EWMA lambda."""
    att, exp_start, rev = ctx["att"], ctx["exp_start"], ctx["rev"]
    pnl = att["actual"].to_numpy(float)
    ds = att["dS"].to_numpy(float)
    ds_rel = att["dS_pct"].to_numpy(float)
    dsig = att["dsig_vega_weighted"].to_numpy(float)
    dr = att["dr"].to_numpy(float)

    rows = []
    for alpha, h in [(a, h) for a in ALPHAS for h in HORIZONS]:
        for res in (
            historical_var(pnl, alpha=alpha, horizon_days=h),
            parametric_var(exp_start, ds, dsig, alpha=alpha, horizon_days=h),
            filtered_historical_var(pnl, alpha=alpha, horizon_days=h, lam=lam),
            full_reval_var(rev, ds_rel, dsig, dr, alpha=alpha, horizon_days=h),
        ):
            row = res.as_row()
            row["ewma_lambda"] = lam
            rows.append(row)

    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    df["_m"] = df["method"].map(order)
    df = df.sort_values(["horizon_days", "confidence", "_m"]).drop(columns="_m")
    return df.reset_index(drop=True)


def stage_var(ctx: dict) -> dict:
    _hr("2. VaR AND EXPECTED SHORTFALL  (four estimators, same book, same window)")
    base = estimator_table(ctx, BASE_LAMBDA)

    for h in HORIZONS:
        sub = base[base["horizon_days"] == h]
        piv_var = sub.pivot(index="method", columns="confidence", values="VaR")
        piv_es = sub.pivot(index="method", columns="confidence", values="ES")
        piv_var, piv_es = piv_var.reindex(METHODS), piv_es.reindex(METHODS)
        print(f"\n  horizon {h} day{'s' if h > 1 else ''}"
              + ("   [sqrt-time scaled -- see MODEL_VALIDATION.md 4.2]" if h > 1 else ""))
        print("    VaR:")
        print(piv_var.to_string(float_format=lambda x: f"{x:>14,.0f}"))
        print("    ES:")
        print(piv_es.to_string(float_format=lambda x: f"{x:>14,.0f}"))

    d1 = base[base["horizon_days"] == 1]
    gaps = {}
    for conf in dict.fromkeys(d1["confidence"]):
        sub = d1[d1["confidence"] == conf]
        par = float(sub.loc[sub["method"] == "parametric", "VaR"].iloc[0])
        fr = float(sub.loc[sub["method"] == "full_reval", "VaR"].iloc[0])
        gaps[conf] = (par, fr, (fr - par) / abs(par) if par else np.nan)

    print("\n  parametric vs full-revaluation, 1-day:")
    for conf, (par, fr, g) in gaps.items():
        print(f"    {conf:>6}   parametric {_money(par):>10}   "
              f"full-reval {_money(fr):>10}   gap {g:+7.1%}")
    worst = max(gaps.values(), key=lambda t: abs(t[2]))
    wg = worst[2]
    print(f"  LARGEST PARAMETRIC vs FULL-REVALUATION GAP: {wg:+.1%}")
    # The escalation threshold in the monitoring plan (MODEL_VALIDATION.md 6) is
    # a 50% divergence. The verdict is read off the number rather than asserted,
    # because on this book at this snapshot the number can come out either way.
    if abs(wg) >= 0.50:
        print("  => MATERIAL. Above the 50% escalation trigger in the monitoring plan:")
        print("     the delta-gamma-vega expansion is not a usable substitute for")
        print("     repricing on this book, and that is the case against a")
        print("     delta-normal framework here.")
    elif abs(wg) >= 0.10:
        print("  => sizeable but below the 50% escalation trigger; the expansion is")
        print("     biased, not broken, at one-day shock sizes.")
    else:
        print("  => NOT material at one-day shock sizes. The delta-gamma-vega")
        print("     expansion tracks the repriced number closely here, because the")
        print("     realised daily moves are small enough for the Taylor expansion to")
        print("     hold. This does NOT extend to stress sizes: section 4 measures the")
        print("     same expansion against the same repricing at scenario magnitudes,")
        print("     where it misses by a wide margin. A small gap at 1-day VaR and a")
        print("     large gap under stress is one finding, not two contradictory ones.")

    # ES >= VaR is an identity of the estimators, not a finding. Printed as a
    # visible check so a future change that breaks it is caught in the log.
    bad = base[base["ES"] < base["VaR"] - 1e-9]
    print(f"\n  ES >= VaR holds on all {len(base)} (method, confidence, horizon) "
          f"cells: {'YES' if bad.empty else 'NO -- ' + str(len(bad)) + ' violations'}")

    tails = base[(base["horizon_days"] == 1) & (base["confidence"] == "99%")]
    print(f"  tail observations behind the 99% ES: "
          + ", ".join(f"{r.method} {r.n_eff_tail:g}" for r in tails.itertuples()))
    print("  With 255 daily moves, a 99% ES is an average over two or three points.")
    print("  It is reported because it is required, and it is an order-of-magnitude")
    print("  statement only (MODEL_VALIDATION.md 3).")

    return {"base": base}


# ==========================================================================
# Stage 2 -- backtest
# ==========================================================================

def stage_backtest(ctx: dict) -> dict:
    att = ctx["att"]
    pnl = att["actual"].to_numpy(float)
    n = len(pnl)
    if n <= BACKTEST_WINDOW + 5:
        raise SystemExit(f"need more than {BACKTEST_WINDOW} moves to backtest")

    _hr(f"3. BACKTEST -- HISTORICAL VaR, {BACKTEST_WINDOW}-DAY ROLLING WINDOW")
    lines: list[str] = []
    lines.append("VaR BACKTEST -- historical simulation, "
                 f"{BACKTEST_WINDOW}-day rolling estimation window")
    lines.append(f"book: 8-leg SPY option book, snapshot {ctx['rev'].base_state.date.date()}")
    lines.append(f"sample: {att.index[0].date()} .. {att.index[-1].date()}  "
                 f"({n} daily moves; {n - BACKTEST_WINDOW} tested)")
    lines.append("")

    results = {}
    for alpha in ALPHAS:
        # VaR for day t is estimated from the BACKTEST_WINDOW moves strictly
        # before t, so no observation informs the forecast that is used to test
        # it. Any overlap here would flatter the model.
        var_path, test_pnl, test_idx = [], [], []
        for t in range(BACKTEST_WINDOW, n):
            var_path.append(historical_var(pnl[t - BACKTEST_WINDOW:t], alpha=alpha).var)
            test_pnl.append(pnl[t])
            test_idx.append(att.index[t])
        bt = backtest_var(test_pnl, var_path, alpha=alpha)
        results[alpha] = {"bt": bt, "var": np.array(var_path),
                          "pnl": np.array(test_pnl),
                          "index": pd.DatetimeIndex(test_idx)}

        conf = f"{100 * (1 - alpha):.1f}%"
        block = [
            f"--- confidence {conf}  (alpha = {alpha}) ---",
            f"  observations tested        : {bt.n}",
            f"  exceptions observed        : {bt.n_exceptions}",
            f"  exceptions expected        : {bt.expected:.2f}",
            f"  Kupiec POF   LR = {bt.kupiec_lr:8.4f}   p = {bt.kupiec_p:.4f}",
            f"  Christoffersen indep. LR = {bt.christoffersen_lr:8.4f}   p = {bt.christoffersen_p:.4f}",
            f"  Basel traffic light        : {bt.basel_zone.upper()}"
            f"   (scaled from the 250-day definition)",
            f"  verdict                    : {bt.verdict}",
            "",
        ]
        lines.extend(block)
        print("\n" + "\n".join(block).rstrip())

    lines.extend([
        "READING THESE NUMBERS",
        "  Kupiec tests the COUNT of exceptions; Christoffersen tests their",
        "  INDEPENDENCE. A model can pass the first and fail the second -- the right",
        "  number of exceptions, all inside one week -- and that is the failure mode",
        "  that costs money. Count-only backtesting cannot see it.",
        "",
        f"  The tested sample is {n - BACKTEST_WINDOW} days, below the 250-day Basel",
        "  window. At that length the Kupiec test has low power against moderate",
        "  misspecification, so a green zone here is weak evidence of adequacy and",
        "  not a pass. This is stated rather than left to be inferred.",
        "",
        "  The rolling window is also the model's memory: a 125-day window forgets a",
        "  shock 125 days after it happens, and the VaR path steps down on that date",
        "  for no reason present in the market.",
    ])
    (OUT / "var_backtest.txt").write_text("\n".join(lines) + "\n")
    print("\n  " + "\n  ".join(lines[-8:]))
    return results


# ==========================================================================
# Stage 3 -- stress
# ==========================================================================

def stage_stress(ctx: dict) -> dict:
    rev, greeks = ctx["rev"], ctx["greeks"]

    _hr("4. STRESS SCENARIOS -- FULL REPRICING")
    hist = run_scenarios(rev, HISTORICAL)
    hyp = run_scenarios(rev, HYPOTHETICAL)
    hist.insert(0, "set", "HISTORICAL")
    hyp.insert(0, "set", "HYPOTHETICAL")
    scen = pd.concat([hist, hyp], ignore_index=True)

    # Taylor comparison, reported alongside but never used as a risk number.
    # Shock sizes come from the Scenario objects, not from re-parsing the
    # formatted strings in the frame -- those are rounded for display.
    by_name = {s.name: s for s in HISTORICAL + HYPOTHETICAL}
    scen["PnL_taylor"] = [
        rev.taylor(by_name[nm].ds_rel, by_name[nm].dvol_abs, by_name[nm].dr_abs, greeks)
        for nm in scen["scenario"]
    ]
    scen["taylor_error"] = scen["PnL_taylor"] - scen["P&L"]
    scen["taylor_error_pct"] = scen["taylor_error"] / scen["P&L"].abs().replace(0, np.nan)
    scen["pct_of_book"] = scen["P&L"] / rev.base_value

    for label in ("HISTORICAL", "HYPOTHETICAL"):
        sub = scen[scen["set"] == label]
        print(f"\n  {label}")
        print(sub[["scenario", "dS_rel", "dVol_pts", "dRate_bp", "horizon_d",
                   "P&L", "pct_of_book", "PnL_taylor", "taylor_error_pct"]
                  ].to_string(index=False, formatters={
                      "P&L": "{:>14,.0f}".format,
                      "PnL_taylor": "{:>14,.0f}".format,
                      "pct_of_book": "{:>+8.1%}".format,
                      "taylor_error_pct": "{:>+9.1%}".format}))

    worst = scen.loc[scen["P&L"].idxmin()]
    print(f"\n  worst scenario: {worst['scenario']}  {_money(worst['P&L'])}  "
          f"({worst['pct_of_book']:+.1%} of book value)")
    med_err = float(scen["taylor_error_pct"].abs().median())
    print(f"  median |Taylor error| across the eleven scenarios: {med_err:.1%}")
    print("  That is why stress is repriced rather than Taylor-expanded: at these")
    print("  sizes the Greeks used by the expansion have themselves moved.")
    print("  Scenario horizons are metadata describing how long each move took in")
    print("  the world. The shock is applied instantaneously, so no theta accrues.")
    scen.to_csv(OUT / "stress_scenarios.csv", index=False)

    # ------------------------------------------------------------------ grid
    _hr("5. STRESS GRID -- SPOT x VOL, FULL REPRICING")
    grid = stress_grid(rev, SPOT_SHOCKS, VOL_SHOCKS)
    print(grid.div(1e3).to_string(float_format=lambda x: f"{x:>10,.0f}")
          + "\n  (thousands of dollars)")
    z = grid.to_numpy(float)
    iv, js = np.unravel_index(int(np.argmin(z)), z.shape)
    ivx, jsx = np.unravel_index(int(np.argmax(z)), z.shape)
    interior = (0 < iv < z.shape[0] - 1) and (0 < js < z.shape[1] - 1)
    print(f"\n  worst cell: {grid.columns[js]} / {grid.index[iv]}  {_money(z.min())}  "
          f"({z.min() / rev.base_value:+.1%} of book)")
    print(f"  best cell : {grid.columns[jsx]} / {grid.index[ivx]}  {_money(z.max())}")
    print(f"  WORST CELL IS INTERIOR (not a corner): {'YES' if interior else 'NO'}")
    if interior:
        print("  => a corner-only scenario set would miss it.")
    else:
        print("  => on this book the worst cell sits on the boundary of the search box,")
        print("     so the grid's shape, not just its corner, is what to read.")
    i0 = int(np.argmin(np.abs(VOL_SHOCKS)))
    j0 = int(np.argmin(np.abs(SPOT_SHOCKS)))
    print(f"  grid (0, 0) cell [{grid.index[i0]} / {grid.columns[j0]}]: "
          f"{z[i0, j0]:.6e}   (must be 0)")
    grid.to_csv(OUT / "stress_grid.csv")

    # -------------------------------------------------------- reverse stress
    _hr("6. REVERSE STRESS -- SMALLEST SHOCK REACHING A STATED LOSS")
    rows = []
    for target in REVERSE_TARGETS:
        r = reverse_stress(rev, target)
        rows.append({
            "target_loss": -abs(target),
            "reachable": r.get("reachable", False),
            "dS_rel": r.get("dS_rel", np.nan),
            "dVol_abs": r.get("dVol_abs", np.nan),
            "P&L": r.get("P&L", np.nan),
            "distance": r.get("distance", np.nan),
            "note": r.get("note", ""),
        })
    revdf = pd.DataFrame(rows)
    print(revdf.to_string(index=False, formatters={
        "target_loss": "{:>13,.0f}".format, "dS_rel": "{:>+8.2%}".format,
        "dVol_abs": "{:>+8.2%}".format, "P&L": "{:>13,.0f}".format,
        "distance": "{:>8.2f}".format}))
    print("\n  Distance is Euclidean in (spot %, vol point) space, which treats a 10%")
    print("  spot move as equidistant from 10 vol points. That is a modelling choice,")
    print("  defensible for a short-dated equity book and not in general; reweighting")
    print("  moves the answer. Search box: spot +/-40%, vol -30 to +60 points.")
    revdf.to_csv(OUT / "reverse_stress.csv", index=False)

    return {"scenarios": scen, "grid": grid, "reverse": revdf}


# ==========================================================================
# Stage 4 -- lambda sensitivity
# ==========================================================================

def stage_sensitivity(ctx: dict) -> dict:
    _hr("7. SENSITIVITY -- VaR ACROSS EWMA DECAY LAMBDA")
    tables = {lam: estimator_table(ctx, lam) for lam in LAMBDA_GRID}

    orderings: dict[tuple, list] = {}
    for lam, tbl in tables.items():
        for conf in tbl["confidence"].unique():
            sub = tbl[(tbl["horizon_days"] == 1) & (tbl["confidence"] == conf)]
            rank = tuple(sub.sort_values("VaR", ascending=False)["method"])
            orderings.setdefault(conf, []).append((lam, rank))

    confs = list(dict.fromkeys(tables[BASE_LAMBDA]["confidence"]))
    rows = []
    for lam, tbl in tables.items():
        sub = tbl[tbl["horizon_days"] == 1]
        row = {"lambda": lam}
        for m in METHODS:
            for conf in confs:
                hit = sub[(sub["method"] == m) & (sub["confidence"] == conf)]
                if len(hit):
                    row[f"{m}_{conf}"] = float(hit["VaR"].iloc[0])
        rows.append(row)
    sens = pd.DataFrame(rows).set_index("lambda")
    print(sens.to_string(float_format=lambda x: f"{x:>14,.0f}"))

    print("\n  Only filtered_hs depends on lambda; the other three are reported at")
    print("  each lambda to make that visible rather than asserted.")
    for conf, pairs in orderings.items():
        print(f"\n  ordering at {conf} (largest VaR first):")
        for lam, rank in pairs:
            print(f"    lambda={lam:g} -> " + " > ".join(rank))

    stable = all(len({rank for _, rank in pairs}) == 1 for pairs in orderings.values())
    print(f"\n  ORDERING STABLE ACROSS LAMBDA: {'YES' if stable else 'NO -- ordering flips'}")
    if not stable:
        print("  => the estimator ranking is not a robust finding at these lambdas;")
        print("     it is reported as non-conclusive in docs/MODEL_VALIDATION.md 5.3.")

    combined = pd.concat(
        [t.assign(run=("baseline" if lam == BASE_LAMBDA else "sensitivity"))
         for lam, t in tables.items()], ignore_index=True)
    cols = ["run", "ewma_lambda", "method", "confidence", "horizon_days",
            "VaR", "ES", "n_obs", "n_eff_tail", "note"]
    combined[cols].to_csv(OUT / "var_summary.csv", index=False)

    return {"tables": tables, "sens": sens, "stable": stable}


# ==========================================================================
# Figure
# ==========================================================================

def _style_axis(ax) -> None:
    """Same axis treatment as src/diagnostics.py, so the risk figure reads as
    part of the same report rather than as a bolt-on."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def risk_figure(grid: pd.DataFrame, var_table: pd.DataFrame, backtest: dict,
                path: Path) -> Path:
    """Three panels: the stress surface, estimator disagreement, the backtest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter

    def thousands(x, _pos=None) -> str:
        if abs(x) >= 1e6:
            return f"{x / 1e6:,.1f}M"
        if abs(x) >= 1e3:
            return f"{x / 1e3:,.0f}k"
        return f"{x:,.0f}"

    fig = plt.figure(figsize=(14.0, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          hspace=0.30, wspace=0.16,
                          left=0.055, right=0.945, top=0.915, bottom=0.075)
    fig.suptitle("Risk: stress surface, estimator disagreement, VaR backtest",
                 color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left")

    # ---------------------------------------------------------- (a) heatmap
    # P&L around zero is a POLARITY encoding, so the ramp is diverging: two
    # hues with a neutral -- not a colour -- at the zero crossing. The loss pole
    # reuses the report's orange and the gain pole its blue.
    ax = fig.add_subplot(gs[0, :])
    cmap = LinearSegmentedColormap.from_list(
        "pnl", ["#7d2f10", "#eb6834", "#f6c3ad", "#eeeeea",
                "#b7d0ee", "#2a78d6", "#123c6d"])
    z = grid.to_numpy(float)
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    norm = TwoSlopeNorm(vmin=min(lo, -1.0), vcenter=0.0, vmax=max(hi, 1.0))
    im = ax.imshow(z, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([c.replace("spot", "") for c in grid.columns])
    ax.set_yticklabels([i.replace("vol", "") + " pt" for i in grid.index])
    ax.set_xlabel("spot shock")
    ax.set_ylabel("vol shock")
    iv, js = np.unravel_index(int(np.argmin(z)), z.shape)
    ax.set_title("(a) Book P&L under full repricing, spot x vol -- $ thousands, "
                 "every cell repriced leg by leg.  Worst cell "
                 f"{grid.columns[js].replace('spot', '')} / "
                 f"{grid.index[iv].replace('vol', '')} pt = {thousands(z[iv, js])}",
                 fontsize=10, loc="left", pad=8)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Every cell carries its number: the ramp answers "where", the label answers
    # "how much", and the palette's low-contrast steps require visible labels.
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            rgba = im.cmap(im.norm(z[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{z[i, j] / 1e3:,.0f}", ha="center", va="center",
                    fontsize=6.4, color=(SURFACE if lum < 0.55 else INK))

    # Outlined cell rather than a marker: a marker would sit on top of the very
    # number it is pointing at.
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((js - 0.5, iv - 0.5), 1.0, 1.0, fill=False,
                           edgecolor=INK, linewidth=2.0, zorder=4))
    # The two deepest vol-down rows are near-identical at large spot-down
    # shocks: the shocked vol has hit the 3% floor from CONFIG, so further
    # vol-down does nothing. Flagged so the flat band is not read as a bug.
    ax.text(0.0, -0.155, "Vol shocks are clipped into [vol_floor, vol_cap] = "
            "[3%, 150%] by src/surface.py, which is why the deepest vol-down "
            "rows flatten out.",
            transform=ax.transAxes, fontsize=7.5, color=INK_MUTED, ha="left")

    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.028)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    cb.outline.set_edgecolor(GRID)
    cb.formatter = FuncFormatter(thousands)
    cb.update_ticks()

    # -------------------------------------------------- (b) estimator bars
    ax = fig.add_subplot(gs[1, 0])
    sub = var_table[(var_table["horizon_days"] == 1)
                    & (var_table["ewma_lambda"] == BASE_LAMBDA)]
    confs = list(dict.fromkeys(sub["confidence"]))
    x = np.arange(len(confs), dtype=float)
    w = 0.20
    bars: dict[str, tuple] = {}
    for k, m in enumerate(METHODS):
        vals = [float(sub[(sub["method"] == m) & (sub["confidence"] == c)]["VaR"].iloc[0])
                for c in confs]
        es = [float(sub[(sub["method"] == m) & (sub["confidence"] == c)]["ES"].iloc[0])
              for c in confs]
        pos = x + (k - 1.5) * (w + 0.012)      # 2px-equivalent surface gap
        ax.bar(pos, vals, width=w, color=METHOD_COLORS[m], label=m,
               lw=0.8, edgecolor=SURFACE, zorder=2)
        # ES rides above each bar as a shape, not a fifth hue.
        ax.scatter(pos, es, marker="D", s=26, color=INK, zorder=4,
                   edgecolors=SURFACE, linewidths=1.4,
                   label="ES" if k == 0 else None)
        bars[m] = (pos, vals, es)
    # Selective direct labels: the tallest bar in each confidence group, which
    # is the number a reader takes away. Every value is in var_summary.csv.
    for gi in range(len(confs)):
        tallest = max(METHODS, key=lambda m: bars[m][1][gi])
        p, v = bars[tallest][0][gi], bars[tallest][1][gi]
        ax.annotate(thousands(v), (p, bars[tallest][2][gi]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(confs)
    ax.set_ylabel("1-day loss ($)")
    ax.set_title("(b) VaR (bars) and ES (diamonds) by estimator, 1-day",
                 fontsize=10, loc="left", pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.42)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2,
              loc="upper left", handlelength=1.4, columnspacing=1.2)
    _style_axis(ax)

    # ----------------------------------------------------- (c) backtest path
    ax = fig.add_subplot(gs[1, 1])
    alpha = min(ALPHAS)
    bt = backtest[alpha]
    idx, loss, var = bt["index"], -bt["pnl"], bt["var"]
    ax.plot(idx, loss, color=INK_MUTED, lw=0.9, label="realised daily loss")
    ax.plot(idx, var, color=METHOD_COLORS["historical"], lw=2.0,
            label=f"rolling {BACKTEST_WINDOW}d VaR {100 * (1 - alpha):.0f}%")
    hit = loss > var
    ax.scatter(idx[hit], loss[hit], s=30, color="#e34948", zorder=4,
               edgecolors=SURFACE, linewidths=1.2,
               label=f"exception ({int(hit.sum())} obs, "
                     f"{bt['bt'].expected:.1f} expected)")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_ylabel("loss ($)")
    ax.set_title(f"(c) Backtest -- Basel zone {bt['bt'].basel_zone.upper()}, "
                 f"Kupiec p {bt['bt'].kupiec_p:.3f}", fontsize=10, loc="left", pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.55 * (hi_y - lo_y))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
    _style_axis(ax)

    fig.savefig(path, dpi=CONFIG["output"]["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["var", "all"])
    ap.add_argument("--live", action="store_true", help="re-pull yfinance")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_inputs(live=args.live)
    var_res = stage_var(ctx)
    bt_res = stage_backtest(ctx)

    if args.stage == "var":
        stage_sensitivity(ctx)
        print(f"\n  wrote to output/: var_summary.csv, var_backtest.txt")
        print("\n[stage=var complete]")
        return

    st_res = stage_stress(ctx)
    stage_sensitivity(ctx)

    _hr("8. OUTPUTS")
    combined = pd.read_csv(OUT / "var_summary.csv")
    risk_figure(st_res["grid"], combined, bt_res, OUT / "risk_heatmap.png")
    written = ["var_summary.csv", "var_backtest.txt", "stress_scenarios.csv",
               "stress_grid.csv", "reverse_stress.csv", "risk_heatmap.png"]
    print("  wrote to output/: " + ", ".join(written))
    print(f"  repricing calls this run: {ctx['rev'].n_calls:,}")
    print("\n[done]")


if __name__ == "__main__":
    main()
