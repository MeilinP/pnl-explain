"""Project 3 -- Option Portfolio PnL Explain Engine.

    python run.py                  # full pipeline (data + all experiments)
    python run.py --stage data     # market data, surface, portfolio exposures only
    python run.py --live           # re-pull yfinance instead of using the cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import gap_days, load_market_data, realized_vol      # noqa: E402
from src.attribution import (Engine, Spec, hedge_effectiveness,       # noqa: E402
                             improvement, summarise)
from src.bucketing import bucket_vega_path                            # noqa: E402
from src.config import CONFIG                                         # noqa: E402
from src.diagnostics import (experiment_figure, frtb_metrics,         # noqa: E402
                             frtb_report, residual_figure,
                             residual_shape, worst_days)
from src.portfolio import (MarketState, aggregate, bucket_vega,       # noqa: E402
                           default_portfolio, load_portfolio,
                           value_portfolio)
from src.rates import build_level_overlay, load_curve                 # noqa: E402
from src.surface import VolSurface, build_surface, load_sabr_params   # noqa: E402

OUT = ROOT / CONFIG["output"]["dir"]
DATA = ROOT / "data"
CAL_DATE = "2026-08-07"          # Project 1's calibration date; lambda = 1 here
_X = CONFIG["experiments"]
BASE_KAPPA = float(_X["kappa_grid"][0])


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ==========================================================================
# Stage 1 -- data, surface, exposures
# ==========================================================================

def stage_data(live: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    md = load_market_data(ROOT, live=live)
    win = md.window
    spot = md.window_spot

    _hr("1. MARKET DATA")
    print(f"underlying      : {CONFIG['data']['underlying']}  (source: {md.source})")
    print(f"window          : {win[0].date()} -> {win[-1].date()}  "
          f"({len(win)} trading days, {len(win) - 1} daily moves)")
    print(f"spot            : {spot.iloc[0]:.2f} -> {spot.iloc[-1]:.2f}  "
          f"(min {spot.min():.2f}, max {spot.max():.2f}, "
          f"{spot.iloc[-1] / spot.iloc[0] - 1:+.1%} over window)")
    lr = np.log(spot).diff().dropna()
    print(f"realised vol    : {lr.std() * np.sqrt(252):.2%} (close-to-close, whole window)")

    gaps = gap_days(md.spot, win)
    thr = CONFIG["data"]["gap_threshold"]
    ret = spot.pct_change().dropna()
    dist = pd.DataFrame({"threshold": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]})
    dist["n_days"] = [int((ret.abs() > t).sum()) for t in dist["threshold"]]
    dist["pct_of_days"] = dist["n_days"] / len(ret)

    _hr(f"2. GAP DAYS  (|dS/S| > {thr:.0%} required by spec)")
    print(dist.to_string(index=False, formatters={"threshold": "{:.1%}".format,
                                                  "pct_of_days": "{:.1%}".format}))
    print(f"\n{len(gaps)} day(s) exceed the {thr:.0%} threshold:\n")
    show = gaps.copy()
    show["ret"] = show["ret"].map("{:+.2%}".format)
    show.index = show.index.date
    print(show.to_string(formatters={"prev_spot": "{:.2f}".format,
                                     "spot": "{:.2f}".format,
                                     "log_ret": "{:+.4f}".format}))
    gaps.to_csv(OUT / "gap_days.csv")
    dist.to_csv(OUT / "gap_distribution.csv", index=False)

    curve = load_curve(ROOT / CONFIG["upstream"]["sofr_curve"])
    ts = load_sabr_params(ROOT / CONFIG["upstream"]["sabr_params"])
    overlay = build_level_overlay(md.rate_level, curve.asof)

    _hr("3. UPSTREAM INPUTS  (Project 1 vol, Project 2 discounting)")
    print(f"discounting : [{curve.source}]\n              {curve.note}")
    print(f"vol surface : [{ts.source}]\n              {ts.note}")
    if overlay is not None:
        ow = overlay.reindex(win).ffill()
        print(f"rate overlay: {CONFIG['data']['rate_symbol']} vs curve as-of "
              f"({curve.asof.date() if curve.asof is not None else 'n/a'}): "
              f"{ow.min():+.2%} to {ow.max():+.2%}")
    else:
        print("rate overlay: NONE -- curve held static, d(r) = 0 across the window")

    positions = load_portfolio(DATA / "portfolio.csv")
    default_portfolio().to_csv(DATA / "portfolio.csv", index=False)

    ctx = {"md": md, "win": win, "spot": spot, "curve": curve, "ts": ts,
           "overlay": overlay, "positions": positions, "gaps": gaps}

    # ---- baseline surface / states, and the exposure report ----------------
    surface, states, lam, anchor = build_variant(ctx, dynamics="ewma",
                                                 kappa=BASE_KAPPA)
    ctx.update(surface=surface, states=states, lam=lam, rv_anchor=anchor)

    _hr("4. VOL SURFACE EVOLUTION  (baseline: symmetric EWMA, kappa=%.0f)" % BASE_KAPPA)
    rv = md.realized_vol.reindex(win).ffill()
    print(f"EWMA(lambda={CONFIG['model']['rv_lambda']}) realized vol anchor on "
          f"{CAL_DATE}: {anchor:.2%}")
    print(f"realized vol over window : {rv.min():.2%} .. {rv.max():.2%}, "
          f"end {rv.iloc[-1]:.2%}")
    print(f"level factor lambda_t    : {lam.min():.3f} .. {lam.max():.3f}, "
          f"end {lam.iloc[-1]:.3f}  (1.000 by construction on {CAL_DATE})")

    legs_first = value_portfolio(positions, states[win[0]], curve, surface)
    legs_last = value_portfolio(positions, states[win[-1]], curve, surface)

    _hr("5. PORTFOLIO")
    print(positions.to_string(index=False))

    for label, d, legs in (("ENTRY", win[0], legs_first), ("END", win[-1], legs_last)):
        _hr(f"6. GREEK EXPOSURES -- {label}  {d.date()}  "
            f"(spot {states[d].spot:.2f}, lambda {states[d].vol_factor:.3f})")
        disp = legs[["position_id", "structure", "option_type", "strike", "expiry",
                     "quantity", "T", "bucket", "iv", "value", "delta", "gamma",
                     "vega", "theta", "rho", "vanna", "volga"]].copy()
        disp["vega"] /= 100.0
        disp["theta"] /= 365.25
        disp["rho"] /= 1e4
        disp["vanna"] /= 100.0
        disp["volga"] /= 1e4
        print(disp.to_string(index=False, formatters={
            "T": "{:.3f}".format, "iv": "{:.2%}".format,
            "value": "{:>14,.0f}".format, "delta": "{:>12,.0f}".format,
            "gamma": "{:>10,.1f}".format, "vega": "{:>11,.0f}".format,
            "theta": "{:>10,.0f}".format, "rho": "{:>9,.0f}".format,
            "vanna": "{:>10,.1f}".format, "volga": "{:>10,.1f}".format}))
        tot = aggregate(legs)
        print(f"\n  BOOK   value {tot['value']:>14,.0f}   "
              f"delta(sh) {tot['delta']:>10,.0f}   gamma {tot['gamma']:>8,.1f}")
        print(f"         vega/pt {tot['vega'] / 100:>11,.0f}   "
              f"theta/day {tot['theta'] / 365.25:>10,.0f}   "
              f"rho/bp {tot['rho'] / 1e4:>8,.0f}")
        print(f"         delta BS {tot['delta']:>10,.0f}   "
              f"SABR {tot['delta_sabr']:>10,.0f}   "
              f"Bartlett {tot['delta_bartlett']:>10,.0f}")
        bv = bucket_vega(legs) / 100.0
        print("         vega by bucket (per pt): "
              + "  ".join(f"{k} {v:>9,.0f}" for k, v in bv.items()))

    rows = []
    for d in win:
        legs = value_portfolio(positions, states[d], curve, surface)
        tot = aggregate(legs)
        bv = bucket_vega(legs)
        rows.append({"date": d, "spot": states[d].spot,
                     "vol_factor": states[d].vol_factor,
                     "realized_vol": states[d].realized_vol,
                     "rate_shift": states[d].rate_shift,
                     **{k: float(tot[k]) for k in tot.index},
                     **{f"vega_{k}": float(v) for k, v in bv.items()}})
    daily = pd.DataFrame(rows).set_index("date")
    daily.to_csv(OUT / "exposures_daily.csv")

    _hr("7. EXPOSURE PATH THROUGH THE WINDOW")
    cols = ["value", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]
    print(pd.DataFrame({"min": daily[cols].min(), "max": daily[cols].max(),
                        "entry": daily[cols].iloc[0], "end": daily[cols].iloc[-1]}
                       ).to_string(float_format=lambda x: f"{x:>16,.1f}"))
    print(f"\ngamma sign changes: {int((np.sign(daily['gamma']).diff().fillna(0) != 0).sum())}"
          f"   vega sign changes: {int((np.sign(daily['vega']).diff().fillna(0) != 0).sum())}")
    ctx["daily"] = daily
    return ctx


def build_variant(ctx: dict, dynamics: str, kappa: float
                  ) -> tuple[VolSurface, dict, pd.Series, float]:
    """Surface + per-date market states for one (vol-dynamics, kappa) variant."""
    md, win, curve = ctx["md"], ctx["win"], ctx["curve"]
    rv_full = realized_vol(md.spot, dynamics)
    surface, anchor = build_surface(ctx["ts"], rv_full, CAL_DATE)
    surface.kappa = float(kappa)
    rv = rv_full.reindex(win).ffill()
    lam = (rv / anchor).rename("vol_factor")
    ov = ctx["overlay"].reindex(win).ffill() if ctx["overlay"] is not None else None
    states = {d: MarketState(date=d, spot=float(md.spot.loc[d]),
                             rate_shift=float(ov.loc[d]) if ov is not None else 0.0,
                             vol_factor=float(lam.loc[d]),
                             realized_vol=float(rv.loc[d])) for d in win}
    return surface, states, lam, anchor


# ==========================================================================
# Stage 2 -- attribution experiments
# ==========================================================================

def stage_attribution(ctx: dict) -> dict:
    positions, curve, win = ctx["positions"], ctx["curve"], ctx["win"]
    base = Engine(positions, ctx["states"], curve, ctx["surface"], dates=list(win))

    summaries: list[dict] = []
    frtb: dict[str, dict] = {}

    def run(spec: Spec, label: str) -> tuple[pd.DataFrame, dict]:
        df = base.run(spec)
        s = summarise(df, label)
        summaries.append(s)
        frtb[label] = frtb_metrics(df)
        return df, s

    # ---------------------------------------------------- EXPERIMENT 1: order
    _hr("8. EXPERIMENT 1 -- FIRST ORDER vs SECOND ORDER  (vega=exact, delta=BS)")
    df1, s1 = run(Spec("first", "bs", "exact"), "1st order (D,V,T,R)")
    df2, s2 = run(Spec("second", "bs", "exact"), "2nd order (+G,Vanna,Volga)")
    imp_order = improvement(s1, s2)
    print(f"  mean |residual|   1st order  ${s1['mean_abs_unexplained']:>12,.0f}")
    print(f"                    2nd order  ${s2['mean_abs_unexplained']:>12,.0f}")
    print(f"  RESIDUAL REDUCTION FROM SECOND-ORDER TERMS : {imp_order:+.2%}")
    print(f"  std residual      {s1['std_unexplained']:>12,.0f} -> {s2['std_unexplained']:>12,.0f}"
          f"   ({improvement(s1, s2, 'std_unexplained'):+.2%})")
    print(f"  R2 explained~actual  {s1['r2_explained_vs_actual']:.5f} -> "
          f"{s2['r2_explained_vs_actual']:.5f}")
    print(f"  residual as % of total |PnL|  {s1['mean_abs_unexp_pct_of_abs_pnl']:.2%} -> "
          f"{s2['mean_abs_unexp_pct_of_abs_pnl']:.2%}")

    # ---------------------------------------------------- EXPERIMENT 2: delta
    _hr("9. EXPERIMENT 2 -- BS DELTA vs SABR DELTA  (2nd order, vega=exact)")
    _, sbs = run(Spec("second", "bs", "exact"), "delta=BS")
    _, ssabr = run(Spec("second", "sabr", "exact"), "delta=SABR")
    _, sbart = run(Spec("second", "bartlett", "exact"), "delta=Bartlett")
    print(f"  mean |residual|   BS        ${sbs['mean_abs_unexplained']:>12,.0f}")
    print(f"                    SABR      ${ssabr['mean_abs_unexplained']:>12,.0f}"
          f"   ({improvement(sbs, ssabr):+.2%})")
    print(f"                    Bartlett  ${sbart['mean_abs_unexplained']:>12,.0f}"
          f"   ({improvement(sbs, sbart):+.2%})")
    hedge, hedge_daily = hedge_effectiveness(base)
    print("\n  Delta-HEDGE effectiveness (the test that actually discriminates):")
    print(hedge.to_string(float_format=lambda x: f"{x:>14,.4f}"))
    print(f"\n  std of daily PnL unhedged: ${hedge_daily['actual'].std(ddof=1):,.0f}")
    hedge.to_csv(OUT / "experiment_delta_hedge.csv")

    # ----------------------------------------------------- EXPERIMENT 3: vega
    _hr("10. EXPERIMENT 3 -- SINGLE VEGA vs BUCKETED VEGA  (2nd order, delta=BS)")
    _, ssing = run(Spec("second", "bs", "single"), "vega=single")
    _, sbuck = run(Spec("second", "bs", "bucketed"), "vega=bucketed")
    _, sexact = run(Spec("second", "bs", "exact"), "vega=exact(per-leg)")
    imp_bucket = improvement(ssing, sbuck)
    print(f"  mean |residual|   single    ${ssing['mean_abs_unexplained']:>12,.0f}")
    print(f"                    bucketed  ${sbuck['mean_abs_unexplained']:>12,.0f}"
          f"   ({imp_bucket:+.2%} vs single)")
    print(f"                    exact     ${sexact['mean_abs_unexplained']:>12,.0f}"
          f"   ({improvement(ssing, sexact):+.2%} vs single)")
    print(f"  bucketing recovers {imp_bucket / max(improvement(ssing, sexact), 1e-12):.1%} "
          f"of the single->exact gap")

    bvp = bucket_vega_path(positions, ctx["states"], curve, ctx["surface"], dates=win)
    bvp.to_csv(OUT / "vega_buckets.csv")
    print(f"\n  bucket-vega additivity (bump-and-reprice, per vol point):")
    print(f"    mean |sum of buckets - parallel bump| = "
          f"${bvp['interaction'].abs().mean() / 100:,.4f}  "
          f"({bvp['interaction'].abs().mean() / bvp['total_parallel'].abs().mean():.3%} "
          f"of mean |total vega|)")
    print("    Additivity is exact here BY CONSTRUCTION, not as a finding: every leg")
    print("    sits in exactly one bucket, so the bumps touch disjoint legs and the")
    print("    central difference cancels the even-order (volga) term. A surface")
    print("    model that let one bucket's bump leak into another expiry would show")
    print("    a non-zero interaction; this one cannot.")

    # A desk that marks one vol number and ignores second-order terms -- the
    # coarsest defensible variant, and the one that stresses the FRTB test.
    _, scoarse = run(Spec("first", "bs", "single"), "coarse (1st, single vega)")
    print(f"\n  coarse desk variant (1st order + single vega): mean |residual| "
          f"${scoarse['mean_abs_unexplained']:,.0f}, "
          f"R2 {scoarse['r2_explained_vs_actual']:.5f}")

    # ------------------------------------- SENSITIVITY A: symmetric vs GJR
    _hr("11. SENSITIVITY A -- SYMMETRIC EWMA vs GJR LEVERAGE")
    asym_rows = []
    asym_detail = {}
    for dyn in _X["vol_dynamics"]:
        surf, states, lam, anchor = build_variant(ctx, dyn, BASE_KAPPA)
        eng = Engine(positions, states, curve, surf, dates=list(win))
        d = eng.run(Spec("second", "bs", "exact"))
        s = summarise(d, f"vol_dynamics={dyn}")
        gapidx = ctx["gaps"].index.intersection(d.index)
        up_gaps = [i for i in gapidx if d.loc[i, "dS"] > 0]
        asym_rows.append({
            "dynamics": dyn, "rv_anchor": anchor,
            "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
            "mean_abs_unexplained": s["mean_abs_unexplained"],
            "std_unexplained": s["std_unexplained"],
            "r2": s["r2_explained_vs_actual"],
            "total_vanna": float(d["pnl_vanna"].sum()),
            "abs_total_vanna": abs(float(d["pnl_vanna"].sum())),
            "mean_abs_vanna": float(d["pnl_vanna"].abs().mean()),
            "vanna_on_up_gaps": float(d.loc[up_gaps, "pnl_vanna"].sum()),
            "resid_on_gap_days": float(d.loc[gapidx, "unexplained"].abs().mean()),
        })
        # Hedge test per variant: Bartlett's delta assumes spot and vol move
        # together with correlation rho < 0. The symmetric EWMA has NO leverage
        # effect, so that assumption is false there; the GJR variant supplies
        # one. If the reasoning is right, Bartlett must improve from ewma to gjr.
        h, _ = hedge_effectiveness(eng)
        for k in h.index:
            asym_rows[-1][f"hedge_std_{k}"] = float(h.loc[k, "std_hedged_pnl"])
            asym_rows[-1][f"hedge_gain_{k}"] = float(h.loc[k, "std_reduction_vs_bs"])
        asym_detail[dyn] = (d, s)
    asym = pd.DataFrame(asym_rows).set_index("dynamics")
    asym.to_csv(OUT / "experiment_vol_dynamics.csv")
    print(asym.to_string(float_format=lambda x: f"{x:>16,.4f}"))
    e, g = asym.loc["ewma"], asym.loc["gjr"]
    print(f"\n  residual        ewma ${e['mean_abs_unexplained']:,.0f}  ->  "
          f"gjr ${g['mean_abs_unexplained']:,.0f}   "
          f"({(e['mean_abs_unexplained'] - g['mean_abs_unexplained']) / e['mean_abs_unexplained']:+.2%})")
    print(f"  total vanna PnL ewma ${e['total_vanna']:,.0f}  ->  gjr ${g['total_vanna']:,.0f}")
    print(f"  vanna on UP-gap days  ewma ${e['vanna_on_up_gaps']:,.0f}  ->  "
          f"gjr ${g['vanna_on_up_gaps']:,.0f}")
    print(f"  mean |residual| on gap days  ewma ${e['resid_on_gap_days']:,.0f}  ->  "
          f"gjr ${g['resid_on_gap_days']:,.0f}")
    print("\n  delta-hedge std under each vol dynamics (the leverage cross-check):")
    hcols = [c for c in asym.columns if c.startswith("hedge_std_")]
    print(asym[hcols].to_string(float_format=lambda x: f"{x:>14,.0f}"))
    print("  Bartlett's delta assumes d(alpha) = rho*nu*F^-beta*dF with rho<0, i.e. a")
    print("  leverage effect. Under the symmetric EWMA the surface has none, so the")
    print("  assumption is false; the GJR variant supplies one. Bartlett's hedge std")
    print(f"  moves {asym.loc['ewma', 'hedge_std_bartlett']:,.0f} -> "
          f"{asym.loc['gjr', 'hedge_std_bartlett']:,.0f} "
          f"({(asym.loc['ewma', 'hedge_std_bartlett'] - asym.loc['gjr', 'hedge_std_bartlett']) / asym.loc['ewma', 'hedge_std_bartlett']:+.1%}).")
    for dyn in _X["vol_dynamics"]:
        summaries.append(asym_detail[dyn][1])
        frtb[f"vol_dynamics={dyn}"] = frtb_metrics(asym_detail[dyn][0])

    # ------------------------------------------- SENSITIVITY B: kappa sweep
    _hr("12. SENSITIVITY B -- VEGA BUCKETING GAIN vs TERM-DAMPING KAPPA")
    krows = []
    for kap in _X["kappa_grid"]:
        surf, states, _, _ = build_variant(ctx, "ewma", kap)
        eng = Engine(positions, states, curve, surf, dates=list(win))
        ss = summarise(eng.run(Spec("second", "bs", "single")), f"kappa={kap} single")
        sb = summarise(eng.run(Spec("second", "bs", "bucketed")), f"kappa={kap} bucketed")
        se = summarise(eng.run(Spec("second", "bs", "exact")), f"kappa={kap} exact")
        krows.append({
            "kappa": kap,
            "resid_single": ss["mean_abs_unexplained"],
            "resid_bucketed": sb["mean_abs_unexplained"],
            "resid_exact": se["mean_abs_unexplained"],
            "bucket_improvement": improvement(ss, sb),
            "exact_improvement": improvement(ss, se),
        })
    ksens = pd.DataFrame(krows)
    ksens["bucket_share_of_gap"] = (ksens["bucket_improvement"]
                                    / ksens["exact_improvement"].replace(0, np.nan))
    ksens.to_csv(OUT / "experiment_kappa.csv", index=False)
    print(ksens.to_string(index=False, float_format=lambda x: f"{x:>16,.4f}"))
    signs = np.sign(ksens["bucket_improvement"].to_numpy(float))
    flipped = bool(len(set(signs.tolist())) > 1)
    print(f"\n  bucketing gain by kappa: "
          + ", ".join(f"kappa={r.kappa:g} -> {r.bucket_improvement:+.2%}"
                      for r in ksens.itertuples()))
    print(f"  CONCLUSION STABLE ACROSS KAPPA: {'NO -- sign flips' if flipped else 'YES'}")
    if flipped:
        print("  => the bucketing experiment does NOT support a conclusion; "
              "this is stated in the README.")

    # ------------------------------------------------------------- outputs
    _hr("13. DIAGNOSTICS AND OUTPUTS")
    main = base.run(Spec("second", "bs", "exact"))
    main.to_csv(OUT / "daily_attribution.csv")

    summary = pd.DataFrame(summaries).drop_duplicates("variant").set_index("variant")
    comp_tot = {f"total_pnl_{c}": float(main[f"pnl_{c}"].sum())
                for c in ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]}
    for k, v in comp_tot.items():
        summary[k] = np.nan
    summary.loc["2nd order (+G,Vanna,Volga)", list(comp_tot)] = list(comp_tot.values())
    summary.to_csv(OUT / "attribution_summary.csv")

    shape = residual_shape(main)
    wd = worst_days(main)
    wd.to_csv(OUT / "worst_days.csv")

    print("  component totals over the window (2nd order, delta=BS, vega=exact):")
    for c in ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]:
        print(f"    {c:<7} ${main[f'pnl_{c}'].sum():>15,.0f}")
    print(f"    {'ACTUAL':<7} ${main['actual'].sum():>15,.0f}")
    print(f"    {'EXPLND':<7} ${main['explained'].sum():>15,.0f}")
    print(f"    {'RESID':<7} ${main['unexplained'].sum():>15,.0f}")

    print("\n  residual shape (full sample | excluding the single worst day "
          f"{shape['excluded_day']}):")
    for tag, nm in (("abs_dS", "vs |dS|  "), ("abs_dsig", "vs |dsig|")):
        print(f"    {nm}   linear R2 {shape[f'{tag}_linear_r2']:.4f} | "
              f"{shape[f'{tag}_linear_r2_ex1']:.4f}    "
              f"quadratic {shape[f'{tag}_quadratic_r2']:.4f} | "
              f"{shape[f'{tag}_quadratic_r2_ex1']:.4f}    "
              f"cubic {shape[f'{tag}_cubic_r2']:.4f} | "
              f"{shape[f'{tag}_cubic_r2_ex1']:.4f}")
    pd.Series(shape).to_csv(OUT / "residual_shape.csv", header=["value"])

    print("\n  worst 5 residual days:")
    print(wd[["dS_pct", "dsig_vega_weighted", "book_gamma", "book_vega",
              "actual", "unexplained", "cause"]].head(5).to_string(
        formatters={"dS_pct": "{:+.2%}".format,
                    "dsig_vega_weighted": "{:+.4f}".format,
                    "book_gamma": "{:>10,.1f}".format,
                    "book_vega": "{:>12,.0f}".format,
                    "actual": "{:>13,.0f}".format,
                    "unexplained": "{:>13,.0f}".format}))

    rep = frtb_report(frtb)
    (OUT / "frtb_test.txt").write_text(rep)
    print("\n" + rep.split("* = breaches")[0].strip())

    residual_figure(main, OUT / "residual_diagnostics.png",
                    " -- 2nd order, BS delta, per-leg vega")
    order_df = pd.DataFrame(
        {"mean_abs_unexplained": [s1["mean_abs_unexplained"],
                                  s2["mean_abs_unexplained"],
                                  ssing["mean_abs_unexplained"]]},
        index=["1st order", "2nd order", "2nd, single vega"])
    experiment_figure(ksens, asym, order_df, OUT / "experiments.png")

    written = ["daily_attribution.csv", "attribution_summary.csv", "vega_buckets.csv",
               "worst_days.csv", "frtb_test.txt", "residual_shape.csv",
               "experiment_delta_hedge.csv", "experiment_vol_dynamics.csv",
               "experiment_kappa.csv", "residual_diagnostics.png", "experiments.png",
               "gap_days.csv", "gap_distribution.csv", "exposures_daily.csv"]
    print("\n  wrote to output/: " + ", ".join(written))

    return {"main": main, "summary": summary, "frtb": frtb, "shape": shape,
            "worst": wd, "kappa": ksens, "asym": asym, "hedge": hedge,
            "imp_order": imp_order, "imp_bucket": imp_bucket, "flipped": flipped,
            "bucket_path": bvp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "all"])
    ap.add_argument("--live", action="store_true", help="re-pull yfinance")
    args = ap.parse_args()

    ctx = stage_data(live=args.live)
    if args.stage == "data":
        print("\n[stage=data complete]")
        return
    stage_attribution(ctx)
    print("\n[done]")


if __name__ == "__main__":
    main()
