"""Residual analysis, FRTB attribution test, and the diagnostic figure.

The attribution itself is arithmetic. What it fails to explain, and when, is the
content -- so everything here is about the residual.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .attribution import SECOND_ORDER
from .config import CONFIG

_F = CONFIG["frtb"]
_X = CONFIG["experiments"]
_OUT = CONFIG["output"]

# Categorical slots 1-7 of the validated reference palette, assigned to the
# seven Greek components in fixed order and never cycled. Validated for the
# adjacent pairlist (stacked areas): worst adjacent CVD dE 9.1, normal-vision
# 19.6. Three slots fall below 3:1 contrast on the light surface, so the
# relief rule applies -- every series carries a legend label and every number
# is also written to CSV.
COMPONENT_COLORS: dict[str, str] = {
    "delta": "#2a78d6", "gamma": "#eb6834", "vega": "#1baf7a",
    "theta": "#eda100", "rho": "#e87ba4", "vanna": "#008300",
    "volga": "#4a3aa7",
}
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#dcdcd8"


# --------------------------------------------------------------------------
# FRTB-style attribution test
# --------------------------------------------------------------------------

def frtb_metrics(df: pd.DataFrame) -> dict:
    """The three Basel PLA metrics comparing risk-theoretical to hypothetical PnL.

    Risk-theoretical PnL is the Greek explanation; hypothetical PnL is the full
    revaluation. A desk failing these cannot use its internal model for capital
    and drops to the standardised approach, which is materially more expensive.
    """
    unexp, act, exp = df["unexplained"], df["actual"], df["explained"]
    std_act = float(act.std(ddof=1))
    mean_ratio = float(unexp.mean()) / std_act if std_act else np.nan
    var_ratio = (float(unexp.var(ddof=1)) / float(act.var(ddof=1))
                 if act.var(ddof=1) else np.nan)
    rho, _ = spearmanr(exp, act)
    return {
        "mean_ratio": mean_ratio,
        "mean_ratio_limit": _F["mean_ratio_limit"],
        "mean_ratio_pass": bool(abs(mean_ratio) <= _F["mean_ratio_limit"]),
        "variance_ratio": var_ratio,
        "variance_ratio_limit": _F["variance_ratio_limit"],
        "variance_ratio_pass": bool(var_ratio <= _F["variance_ratio_limit"]),
        "spearman": float(rho),
        "spearman_floor": _F["spearman_floor"],
        "spearman_pass": bool(rho >= _F["spearman_floor"]),
        "mean_unexplained": float(unexp.mean()),
        "std_actual": std_act,
    }


def frtb_report(metrics_by_variant: dict[str, dict]) -> str:
    """Human-readable FRTB report across attribution variants."""
    L = ["FRTB-STYLE PROFIT-AND-LOSS ATTRIBUTION TEST",
         "=" * 72, "",
         "Risk-theoretical PnL = Greek explanation; hypothetical PnL = full reval.",
         "Basel thresholds: |mean ratio| <= 10%, variance ratio <= 20%.",
         "A desk that fails must use the standardised approach instead of an",
         "internal model, which materially increases its capital requirement.", ""]
    hdr = f"{'variant':<28}{'mean ratio':>13}{'var ratio':>12}{'spearman':>11}{'result':>10}"
    L += [hdr, "-" * len(hdr)]
    for name, m in metrics_by_variant.items():
        overall = "PASS" if (m["mean_ratio_pass"] and m["variance_ratio_pass"]) else "FAIL"
        L.append(f"{name:<28}{m['mean_ratio']:>12.2%}{'*' if not m['mean_ratio_pass'] else ' '}"
                 f"{m['variance_ratio']:>11.2%}{'*' if not m['variance_ratio_pass'] else ' '}"
                 f"{m['spearman']:>11.4f}{overall:>10}")
    L += ["", "* = breaches its threshold.", ""]
    for name, m in metrics_by_variant.items():
        L += [f"{name}:",
              f"    mean(unexplained)  = {m['mean_unexplained']:>15,.0f}",
              f"    std(actual)        = {m['std_actual']:>15,.0f}",
              f"    mean ratio         = {m['mean_ratio']:>15.4%}   "
              f"limit +/-{m['mean_ratio_limit']:.0%}   "
              f"{'PASS' if m['mean_ratio_pass'] else 'FAIL'}",
              f"    variance ratio     = {m['variance_ratio']:>15.4%}   "
              f"limit  {m['variance_ratio_limit']:.0%}     "
              f"{'PASS' if m['variance_ratio_pass'] else 'FAIL'}",
              f"    spearman(exp,act)  = {m['spearman']:>15.4f}   "
              f"floor  {m['spearman_floor']:.2f}     "
              f"{'PASS' if m['spearman_pass'] else 'FAIL'}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Residual shape
# --------------------------------------------------------------------------

def _fit(x: np.ndarray, y: np.ndarray, deg: int) -> tuple[np.ndarray, float]:
    """Polynomial fit through the origin-free basis; returns (coeffs, R^2)."""
    c = np.polyfit(x, y, deg)
    resid = y - np.polyval(c, x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return c, (1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else np.nan)


def residual_shape(df: pd.DataFrame) -> dict:
    """Fit the residual against |dS| and |d(sigma)| to diagnose the missing order.

    A quadratic in |dS| that survives the second-order attribution means gamma
    is not enough; a cubic term that adds explanatory power on top points at
    missing speed (d3V/dS3).
    """
    out: dict = {}
    dsig = df["dsig_vega_weighted"].to_numpy(float)
    y_all = df["unexplained"].to_numpy(float)
    # The single worst day is an extreme leverage point for a polynomial fit.
    # Every R^2 is therefore reported twice: full sample, and excluding that one
    # observation. If a shape conclusion does not survive the drop, it rests on
    # one data point and is not a conclusion.
    drop = int(np.argmax(np.abs(y_all)))
    keep = np.ones(len(y_all), dtype=bool)
    keep[drop] = False
    out["excluded_day"] = str(df.index[drop].date())

    for tag, x_all in (("abs_dS", np.abs(df["dS"].to_numpy(float))),
                       ("abs_dsig", np.abs(dsig))):
        for suffix, sel in (("", slice(None)), ("_ex1", keep)):
            x, y = x_all[sel], y_all[sel]
            for deg, nm in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
                c, r2 = _fit(x, y, deg)
                out[f"{tag}_{nm}_r2{suffix}"] = r2
                out[f"{tag}_{nm}_lead_coef{suffix}"] = float(c[0])
            out[f"{tag}_quad_gain_over_linear{suffix}"] = (
                out[f"{tag}_quadratic_r2{suffix}"] - out[f"{tag}_linear_r2{suffix}"])
            out[f"{tag}_cubic_gain_over_quad{suffix}"] = (
                out[f"{tag}_cubic_r2{suffix}"] - out[f"{tag}_quadratic_r2{suffix}"])
    return out


def _cause(row: pd.Series, gap_thr: float, dsig_thr: float) -> str:
    """One-line attributed cause for a large-residual day."""
    bits = []
    if abs(row["dS_pct"]) > gap_thr:
        bits.append(f"gap {row['dS_pct']:+.2%}")
    elif abs(row["dS_pct"]) > gap_thr / 2:
        bits.append(f"large move {row['dS_pct']:+.2%}")
    if abs(row["dsig_vega_weighted"]) > dsig_thr:
        bits.append(f"vol shift {row['dsig_vega_weighted'] * 100:+.2f}pt")
    comps = {c: abs(row.get(f"pnl_{c}", 0.0)) for c in SECOND_ORDER}
    dom = max(comps, key=comps.get)
    bits.append(f"dominant term {dom}")
    if abs(row["dS_pct"]) > gap_thr / 2 and abs(row["book_gamma"]) > 0:
        same = np.sign(row["unexplained"]) == np.sign(row["book_gamma"])
        bits.append("gamma under-stated" if same else "gamma over-stated")
    if not bits:
        bits.append("no single dominant driver")
    return "; ".join(bits)


def worst_days(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """The n days with the largest absolute residual, with a diagnosed cause."""
    n = _X["worst_days"] if n is None else n
    gap_thr = CONFIG["data"]["gap_threshold"]
    dsig_thr = float(df["dsig_vega_weighted"].abs().quantile(0.90))
    top = df.reindex(df["unexplained"].abs().sort_values(ascending=False).index).head(n)
    cols = ["spot_0", "spot_1", "dS", "dS_pct", "dsig_vega_weighted",
            "book_gamma", "book_vega", "book_vanna", "actual", "explained",
            "unexplained", "unexplained_pct_of_actual"] + \
           [f"pnl_{c}" for c in SECOND_ORDER]
    out = top[[c for c in cols if c in top.columns]].copy()
    out["cause"] = [_cause(r, gap_thr, dsig_thr) for _, r in top.iterrows()]
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def _style_axis(ax) -> None:
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


def _thousands(x, _pos) -> str:
    """Axis formatter. One rule for the whole axis -- switching format at 1e3
    produced duplicate labels ("2k, 2k") when ticks straddled the boundary."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:,.1f}k"
    return f"{x:,.0f}"


def residual_figure(df: pd.DataFrame, path: Path, title_suffix: str = "") -> Path:
    """Four-panel residual diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), facecolor=SURFACE)
    fig.suptitle(f"Unexplained PnL diagnostics{title_suffix}",
                 color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left")

    unexp = df["unexplained"]
    sd = float(unexp.std(ddof=1))

    # -- 1. residual time series with +/-2 sigma bands ----------------------
    ax = axes[0, 0]
    ax.axhspan(-2 * sd, 2 * sd, color="#2a78d6", alpha=0.08, lw=0)
    ax.axhline(2 * sd, color="#2a78d6", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(-2 * sd, color="#2a78d6", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.plot(df.index, unexp, color=INK, lw=1.0)
    out = unexp[unexp.abs() > 2 * sd]
    ax.scatter(out.index, out, s=22, color="#e34948", zorder=3,
               label=f"beyond 2 sigma ({len(out)} of {len(unexp)})")
    ax.set_title("Unexplained PnL by date", fontsize=10, loc="left")
    ax.set_ylabel("unexplained ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    ax.annotate(f"2 sigma = ${2 * sd:,.0f}", xy=(0.985, 0.04), xycoords="axes fraction",
                ha="right", fontsize=8, color=INK_MUTED)
    worst = unexp.abs().idxmax()
    ax.annotate(f"{worst.date()}  ${unexp.loc[worst]:,.0f}\n"
                f"({df.loc[worst, 'dS_pct']:+.2%} gap)",
                xy=(worst, unexp.loc[worst]), xytext=(14, 26),
                textcoords="offset points", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    _style_axis(ax)

    # -- 2. residual vs |dS| with fitted quadratic --------------------------
    ax = axes[0, 1]
    x = df["dS"].abs().to_numpy(float)
    y = unexp.to_numpy(float)
    ax.scatter(x, y, s=14, color="#2a78d6", alpha=0.55, lw=0, label="daily residual")
    xs = np.linspace(x.min(), x.max(), 200)
    c2, r2_2 = _fit(x, y, 2)
    c3, r2_3 = _fit(x, y, 3)
    ax.plot(xs, np.polyval(c2, xs), color="#eb6834", lw=2.0,
            label=f"quadratic fit  $R^2$={r2_2:.3f}")
    ax.plot(xs, np.polyval(c3, xs), color="#4a3aa7", lw=1.6, ls="--",
            label=f"cubic fit  $R^2$={r2_3:.3f}")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_title("Residual vs |dS|  (curvature beyond gamma)", fontsize=10, loc="left")
    ax.set_xlabel("|dS| ($ per share)")
    ax.set_ylabel("unexplained ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    _style_axis(ax)

    # -- 3. residual vs |d sigma| -------------------------------------------
    ax = axes[1, 0]
    xv = df["dsig_vega_weighted"].abs().to_numpy(float) * 100.0
    ax.scatter(xv, y, s=14, color="#1baf7a", alpha=0.55, lw=0, label="daily residual")
    xs = np.linspace(xv.min(), xv.max(), 200)
    cv, r2_v = _fit(xv, y, 2)
    ax.plot(xs, np.polyval(cv, xs), color="#eb6834", lw=2.0,
            label=f"quadratic fit  $R^2$={r2_v:.3f}")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_title("Residual vs |d sigma|  (vol-of-vol beyond volga)",
                 fontsize=10, loc="left")
    ax.set_xlabel("|d sigma| (vol points, vega-weighted)")
    ax.set_ylabel("unexplained ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    _style_axis(ax)

    # -- 4. cumulative decomposition, signed stack --------------------------
    # Contributions change sign, so positives stack up from zero and negatives
    # stack down; a naive stackplot would silently mis-render the mixed signs.
    ax = axes[1, 1]
    pos = np.zeros(len(df))
    neg = np.zeros(len(df))
    for comp in SECOND_ORDER:
        cum = df[f"pnl_{comp}"].cumsum().to_numpy(float)
        up, dn = np.clip(cum, 0, None), np.clip(cum, None, 0)
        col = COMPONENT_COLORS[comp]
        ax.fill_between(df.index, pos, pos + up, color=col, lw=0.8,
                        edgecolor=SURFACE, label=comp)
        ax.fill_between(df.index, neg, neg + dn, color=col, lw=0.8,
                        edgecolor=SURFACE)
        pos, neg = pos + up, neg + dn
    ax.plot(df.index, df["actual"].cumsum(), color=INK, lw=1.8,
            label="actual (full reval)")
    ax.plot(df.index, df["explained"].cumsum(), color=INK, lw=1.2, ls="--",
            label="explained")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_title("Cumulative PnL by component", fontsize=10, loc="left")
    ax.set_ylabel("cumulative ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_MUTED, ncol=3,
              loc="upper left")
    _style_axis(ax)

    for ax in (axes[0, 0], axes[1, 1]):
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha("center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=_OUT["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path


def experiment_figure(kappa_df: pd.DataFrame, asym_df: pd.DataFrame,
                      order_df: pd.DataFrame, path: Path) -> Path:
    """Three-panel summary of the comparison experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), facecolor=SURFACE)
    fig.suptitle("Comparison experiments", color=INK, fontsize=13,
                 fontweight="bold", x=0.008, ha="left")

    ax = axes[0]
    labels = order_df.index.tolist()
    vals = order_df["mean_abs_unexplained"].to_numpy(float)
    ax.bar(labels, vals, color=["#2a78d6", "#eb6834", "#1baf7a"][:len(labels)],
           width=0.55)
    for i, v in enumerate(vals):
        ax.annotate(f"${v:,.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=INK)
    ax.set_title("Mean |residual| by attribution variant", fontsize=10, loc="left")
    ax.set_ylabel("mean |unexplained| ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    _style_axis(ax)

    ax = axes[1]
    ks = kappa_df.sort_values("kappa")     # rows arrive baseline-first, not sorted
    ax.plot(ks["kappa"], ks["bucket_improvement"] * 100,
            marker="o", ms=7, color="#2a78d6", lw=2.0, label="bucketed vs single")
    ax.axhline(0, color="#e34948", lw=1.0, ls="--")
    for _, r in ks.iterrows():
        ax.annotate(f"{r['bucket_improvement']:+.1%}", (r["kappa"],
                    r["bucket_improvement"] * 100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, max(ks["bucket_improvement"].max() * 100 * 1.35, 5))
    ax.set_xticks(ks["kappa"].tolist())
    ax.set_title("Vega-bucketing gain vs term-damping kappa", fontsize=10, loc="left")
    ax.set_xlabel("kappa")
    ax.set_ylabel("residual reduction (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower left")
    _style_axis(ax)

    # Both series are vanna dollars on one scale. Pairing mean |residual| ($43)
    # with total vanna ($53k) on a shared axis would hide the smaller bar
    # entirely -- that number lives in experiment_vol_dynamics.csv instead.
    ax = axes[2]
    idx = np.arange(len(asym_df))
    w = 0.38
    ax.bar(idx - w / 2, asym_df["abs_total_vanna"], w, color="#008300",
           label="|total vanna PnL|")
    ax.bar(idx + w / 2, asym_df["vanna_on_up_gaps"].abs(), w, color="#1baf7a",
           label="|vanna PnL on up-gap days|")
    for i, (a, b) in enumerate(zip(asym_df["abs_total_vanna"],
                                   asym_df["vanna_on_up_gaps"].abs())):
        ax.annotate(f"${a / 1e3:,.1f}k", (i - w / 2, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=INK)
        ax.annotate(f"${b / 1e3:,.1f}k", (i + w / 2, b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(asym_df.index)
    ax.set_title("Vanna attribution: symmetric EWMA vs GJR leverage",
                 fontsize=10, loc="left")
    ax.set_ylabel("$")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    _style_axis(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=_OUT["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path
