# Model Validation Report — Option Book VaR and Stress Framework

**Model ID:** MKT-VAR-001
**Model owner:** M. Pan
**Validation performed by:** M. Pan (self-validation; see §7 on independence)
**Date:** 2026-09-08 (run of `run_risk.py`; risk snapshot 2026-08-07)
**Framework:** Structured to SR 11-7 (Federal Reserve / OCC *Supervisory Guidance on Model Risk Management*), which organises validation into three pillars: conceptual soundness, ongoing monitoring, and outcomes analysis.

---

## 1. Model purpose and intended use

Estimate one-day and multi-day loss distributions for an eight-leg SPY option
book, and quantify loss under prescribed stress scenarios.

**In scope:** market risk on listed SPY options and the underlying, driven by
three factors — spot, implied volatility level, and the discount rate.

**Out of scope, and therefore not measured by this model:**
- Counterparty and credit risk
- Funding and liquidity risk; bid/ask and market impact on unwind
- Dividend risk (treated as deterministic)
- Skew and term-structure risk beyond a parallel vol level shift (see §4.1 —
  this is the model's most consequential limitation)

A number produced by this model is not a statement about how much the book
could lose. It is a statement about how much the book could lose *from those
three factors, under the assumptions in §4*.

---

## 2. Model methodology

Four VaR estimators run on the same book and window:

| Estimator | Method | Distributional assumption |
|---|---|---|
| Historical simulation | Empirical order-statistic quantile of realised daily P&L | None |
| Parametric (delta-normal ± gamma) | Normal quantile of Greek-implied P&L variance | Normal returns; local Taylor validity |
| Filtered historical simulation | Barone-Adesi; EWMA-devolatilised residuals rescaled to current vol | Conditional homoskedasticity of residuals |
| Full revaluation | Historical shocks applied by full leg-level repricing | None beyond the pricing model |

Expected Shortfall is reported alongside every VaR. ES is the average loss
conditional on exceeding VaR; unlike VaR it is subadditive, and unlike VaR it
is sensitive to how bad the tail is rather than only where it starts. FRTB
replaced VaR with ES for exactly this reason.

Stress testing uses three families — historical replays, hypothetical shocks,
and a reverse stress test — all applied by full repricing (§4.3).

---

## 3. Data

| Input | Source | Window | Known issues |
|---|---|---|---|
| Spot path | yfinance daily close | 2025-08-01 .. 2026-08-07 (256 valuation dates, 255 daily moves; EWMA burn-in from 2024-06-01) | Close-to-close only; no intraday path |
| Implied vol | SABR surface (Project 1) evolved per §5.1 of the attribution README | Single calibration 2026-08-07: 8 expiries, T 0.115y–2.357y, ATM 13.23%–19.72%, ρ −0.639..−0.503, ν 0.319..2.152, β = 0.5 | Single-date calibration; ρ and ν frozen |
| Discount curve | Bootstrapped SOFR OIS (Project 2) + ^IRX level overlay | Curve as-of 2026-03-09: 21 pillars, 0.025y–40.04y, zeros 3.3037%–3.8176%; overlay spans the P&L window | Single snapshot; no curve history |
| Realised P&L | `daily_attribution.csv` from the attribution pipeline | 255 daily moves | — |

**Sample adequacy.** 255 observations at 99% implies an expected 2.55 tail
observations. ES at that confidence is therefore an average over roughly
two or three points. It is reported because it is required, and it should be
read as indicative of order of magnitude only. This is disclosed here rather
than left for the reader to infer from the sample size.

---

## 4. Assumptions and limitations

### 4.1 Single vol factor — the binding limitation

Volatility risk is represented by one factor: a parallel shift in the level of
the surface. The bucketed-vega work in the attribution project shows why this
is the binding constraint: on 2026-04-07 the book's bucket vegas were
+30,085 / −41,536 / +32,499 per vol point across 3M–6M / 6M–1Y / 1Y+, against
a parallel total of just +21,049. **The buckets largely cancel.** A parallel-shift
stress therefore reports a modest long-vega book while the ladder shows a
large calendar position that any non-parallel move would hit.

Consequence: **VaR and stress numbers from this model are downward-biased for
any scenario in which the term structure twists.** Historically, twists are
what happens in a vol shock. This is a known, quantified, uncorrected
limitation, not a residual uncertainty.

Remediation: extend the risk factor set from one vol level to the four vega
buckets already computed by `bucketing.py`. Tracked as an open item.

### 4.2 Square-root-of-time scaling

Multi-day figures scale the one-day number by √h. This requires iid returns
and a locally linear book. **Both fail here.** The book's gamma changes sign
19 times over the window, so a 10-day move is not ten one-day moves applied to
a constant delta. Every scaled output carries a flag in its `note` field.

Direction of the error is not signable a priori: √t under-states the tail when
returns are positively autocorrelated in a crisis, and over-states convexity
losses for a book that is long gamma over the relevant range.

### 4.3 Stress applied by full repricing, deliberately

Stress scenarios are repriced from scratch rather than Taylor-expanded. A
Greek-based stress number is approximately meaningless at the sizes that
matter, because the Greeks themselves have moved by the time the shock is
fully realised. The attribution project's worst residual day (2026-03-31,
+2.91% gap, book gamma at its window maximum of +39.5) is a 3% move showing
this effect; a 20% scenario is far outside where a Taylor expansion holds.

### 4.4 Historical scenario calibration

The six historical episodes use widely reported index-level moves, rounded.
They capture the shape of each event, not a tick-accurate replay. Reruns
against vendor data should be expected to differ at the margin; the second
significant digit is not meaningful.

### 4.5 Structural blind spot in the historical set

All six historical scenarios pair a spot fall with a vol rise. A book that is
long vol and short spot passes all six and may still be badly exposed. The
hypothetical set exists specifically to cover that gap, including an "up-crash"
(spot +15%, vol +15 points) with no historical precedent in this sample.

§5.4 confirms this is not a theoretical concern on this book: the cheapest path
to every reverse-stress target is a spot fall with vol flat or *falling*, which
is the one shape none of the six historical episodes has.

### 4.6 Shocked vols are clipped, so deep vol-down stress is under-stated

`src/surface.py` clips every implied vol into `[vol_floor, vol_cap]` = [3%,
150%] from `src/config.py`. A stress shock is applied to each leg's own vol mark
and then passes through that clip, so a large negative vol shock saturates
rather than going through the floor.

This is visible in `output/stress_grid.csv`: at spot −30% and at spot −25%, the
−15 and −20 point vol rows are identical **to the cent** (−$954,270.4355 and
−$1,231,853.6536 respectively), because both shocks have already driven the
legs onto the 3% floor. Those cells are not measuring a −20 point shock; they
are measuring whatever shock first reaches the floor.

Consequence: **grid cells in the deep vol-down region are floor artefacts and
under-state the loss** an unclipped model would report. The clip is inherited
from the attribution pipeline, where it guards against an unphysical vol during
a bump-and-reprice; it was not designed for stress magnitudes.

This reaches the headline number. The worst cell reported in §5.5 — spot −20% /
vol −15 points, −$1,582,048 — differs from the −20 point cell in the same column
by **$24 on a $1.58M loss**, so that cell is itself essentially on the floor.
**The worst-cell figure is a lower bound, not an estimate**, and the true
worst-case under an unclipped vol-down shock is worse by an unquantified amount.
Tracked as an open item.

---

## 5. Testing performed

### 5.1 Benchmarking — estimator disagreement is the output

The four estimators are run on identical data. Where they disagree, the gap
measures the cost of each approximation rather than being noise to be averaged
away.

Book snapshot 2026-08-07: value $1,662,294; delta 4,314 sh; gamma −29.9;
vega −$1,338/pt; theta +$491/day; rho +$361/bp. Source: `output/var_summary.csv`.

**One-day VaR and ES, EWMA λ = 0.94 (losses positive, dollars)**

| Estimator | VaR 95% | VaR 97.5% | VaR 99% | ES 95% | ES 97.5% | ES 99% |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 53,279 | 65,709 | 70,650 | 65,750 | 73,151 | 79,784 |
| Parametric (δ+γ+vega) | 49,135 | 58,548 | 69,492 | 61,617 | 69,834 | 79,615 |
| Filtered HS | 64,054 | 71,016 | 131,686 | 87,443 | 105,092 | 142,883 |
| Full revaluation | 45,545 | 53,033 | 65,588 | 58,061 | 65,796 | 80,219 |

**Ten-day VaR and ES (√t scaled from the above — see §4.2)**

| Estimator | VaR 95% | VaR 97.5% | VaR 99% | ES 95% | ES 97.5% | ES 99% |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 168,484 | 207,789 | 223,416 | 207,919 | 231,324 | 252,300 |
| Parametric (δ+γ+vega) | 155,377 | 185,143 | 219,753 | 194,849 | 220,835 | 251,763 |
| Filtered HS | 202,555 | 224,572 | 416,427 | 276,518 | 332,332 | 451,837 |
| Full revaluation | 144,026 | 167,705 | 207,409 | 183,606 | 208,065 | 253,675 |

Tail counts behind each order-statistic figure, from 255 moves: 13 observations
at 95%, 7 at 97.5%, 3 at 99%. The parametric row is a closed form and uses the
whole sample, which is why it needs no tail count and is also why it cannot
represent a tail shape it has not assumed.

**What the run confirmed.** The expected pattern held for two of the three
approximations: historical (70,650) and filtered historical (131,686) both
exceed delta-normal (69,492) at 99%, because the realised P&L distribution has
fat tails the normal assumption cannot represent, and because the gamma term is
quadratic in a variable whose distribution is not normal to begin with.

**What the run contradicted.** Full revaluation came out *below* parametric at
every confidence — 45,545 vs 49,135 at 95%, 53,033 vs 58,548 at 97.5%, 65,588
vs 69,492 at 99%. The largest gap is **−9.4%**, well inside the 50% escalation
threshold in §6.

**A material gap between parametric and full-revaluation VaR is the
quantitative case for not running a delta-normal framework on this book.** On
this book, at this snapshot, at one-day shock sizes, **that gap is not
material**, and the finding is reported as such rather than asserted from the
prior. The realised daily moves are small enough that the delta-gamma-vega
expansion tracks the repriced book to within a tenth.

That result does not extend past one-day sizes, and the same run measures where
it stops holding. Against the eleven stress scenarios in §5.5 the identical
expansion, using the identical Greeks, misses the repriced answer by a **median
of 32%** and by as much as **89%** (COVID crash 2020: −$917k expanded against
−$484k repriced). A small gap at VaR sizes and a large one at stress sizes is a
single coherent finding about where the Taylor expansion's radius of validity
ends — not two results in tension.

The filtered historical estimator is the outlier in the other direction: at 99%
it reports 131,686, roughly double every other estimator. It rescales
standardised residuals to the *current* vol level, and the snapshot date sits in
a quiet regime, so its three tail residuals are being multiplied up. With three
observations behind it, that number carries the §3 sample-adequacy caveat with
particular force.

### 5.2 Backtesting — count and independence

Two tests, because they fail differently:

- **Kupiec POF** — is the *number* of exceptions consistent with the stated
  confidence level?
- **Christoffersen independence** — are exceptions *independent*, or clustered?

A model can pass Kupiec and fail Christoffersen: the right number of
exceptions, all in the same week. That is the failure mode that costs money,
and count-only backtesting cannot see it.

Basel traffic-light zones are reported, scaled from their 250-day definition.

**Power caveat, stated up front:** at 255 observations and 99% confidence, the
Kupiec test has low power against moderate misspecification. A green zone here
is weak evidence of adequacy, not a pass. Reporting a green zone without this
caveat would overstate what the test establishes.

**Results.** Historical VaR, 125-day rolling estimation window, 130 days
tested (2026-02-02 .. 2026-08-07). Full output in `output/var_backtest.txt`.

| Confidence | Exceptions | Expected | Kupiec LR | Kupiec p | Christoffersen LR | Christoffersen p | Basel zone |
|---|---:|---:|---:|---:|---:|---:|---|
| 95% | 12 | 6.50 | 3.9632 | 0.0465 | 0.0152 | 0.9020 | **RED** |
| 97.5% | 9 | 3.25 | 7.0991 | 0.0077 | 0.0000 | 1.0000 | **RED** |
| 99% | 3 | 1.30 | 1.6400 | 0.2003 | 0.0000 | 1.0000 | YELLOW |

**This is an adverse result and is reported as one.** Kupiec rejects the model
at both 95% (p = 0.047) and 97.5% (p = 0.008): the exception count is roughly
double to triple its expectation, so the historical estimator **under-states
risk** at the two lower confidence levels. At 99% the count test does not
reject, but three exceptions against 1.30 expected is still an over-run, the
zone is yellow rather than green, and the power caveat above means that
non-rejection is close to uninformative.

Christoffersen does not reject at any level (p = 0.90, 1.00, 1.00). The
exceptions are spread through the window rather than clustered. This is the one
favourable half of the backtest and it should not be read as offsetting the
other half: independent exceptions arriving at three times the promised rate is
still a model that is wrong about the level.

Mechanism, so the failure is diagnosed rather than merely recorded: the 125-day
window is half the Basel length and was chosen so that a 255-move sample leaves
130 days to test. A short window tracks a vol regime shift quickly but holds
few tail observations — at 95% the VaR is the 7th-worst of 125 — so the estimate
is noisy exactly where the exception count is decided. The §7 conclusion is
qualified accordingly, and remediation is tracked as an open item.

### 5.3 Sensitivity analysis

VaR is recomputed across the EWMA decay parameter λ ∈ {0.90, 0.94, 0.97} and
confidence levels {95%, 97.5%, 99%}. The reported conclusion is the *stability
of the ordering* across estimators, not any single point estimate. If the
ordering flips under a defensible parameter choice, the finding is reported
as non-conclusive.

**Result: `ORDERING STABLE ACROSS LAMBDA: YES`.** At every λ in the grid and
every confidence level, the one-day ordering is

> filtered HS > historical > parametric > full revaluation

Only the filtered estimator depends on λ; the other three are recomputed at
each λ anyway so that their invariance is demonstrated rather than asserted.
Its 99% figure is the λ-sensitive number and it is sensitive: 157,859 at
λ = 0.90, 131,686 at 0.94, 107,787 at 0.97 — a spread of **46%** of the
baseline. The *ordering* survives that spread; **the level does not**, so the
filtered 99% figure should be quoted as a range and not as a point estimate.

| λ | Filtered HS 95% | 97.5% | 99% |
|---|---:|---:|---:|
| 0.90 | 65,455 | 71,140 | 157,859 |
| 0.94 (baseline) | 64,054 | 71,016 | 131,686 |
| 0.97 | 66,174 | 74,448 | 107,787 |

### 5.4 Reverse stress test

Solves for the smallest shock, by Euclidean distance in (spot %, vol point)
space, that produces a stated loss. The distance metric is a modelling choice,
not a fact: treating 10% of spot as equidistant from 10 vol points is
defensible for a short-dated equity book and is not defensible in general.
Reweighting changes the answer.

**Results** (search box: spot ±40%, vol −30 to +60 points; `output/reverse_stress.csv`):

| Target loss | Reachable | Spot shock | Vol shock | Realised P&L | Distance |
|---|---|---:|---:|---:|---:|
| −$100,000 | yes | −3.33% | 0.00 pt | −113,111 | 3.33 |
| −$250,000 | yes | −6.67% | 0.00 pt | −255,972 | 6.67 |
| −$500,000 | yes | −10.67% | −2.25 pt | −506,318 | 10.90 |
| −$1,000,000 | yes | −16.00% | −5.25 pt | −1,012,676 | 16.84 |

Read against §5.1: a 3.3% down move — which the attribution window contains
several of — reaches a loss of $113k, above the 99% one-day full-revaluation
VaR of $65,588. That is not a contradiction. VaR is a quantile of the *joint*
historical distribution, in which a spot fall arrives with a vol rise that this
short-gamma, short-vega book is partly hedged against; the reverse stress test
searches over spot and vol independently and is free to pick the pairing that
hurts. The gap between the two is a measure of how much of the book's apparent
safety is carried by that correlation assumption.

Note the shape of the answer: the cheapest route to every one of these losses
is a spot fall with vol **flat or falling**, never a vol rise. That is the §4.5
blind spot showing up quantitatively — all six historical scenarios pair a fall
with a vol spike, and none of them is the cheapest path to a large loss here.

### 5.5 Stress scenario results

Eleven scenarios, all applied by full repricing to the 2026-08-07 snapshot
(`output/stress_scenarios.csv`). `P&L (Taylor)` is the second-order Greek
expansion of the same shock, reported only to measure the approximation error
of §4.3 — no risk number in this report is taken from it.

| Set | Scenario | Spot | Vol | Rate | P&L (repriced) | % of book | P&L (Taylor) | Taylor error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Historical | 2022 rates repricing | −25.0% | +15 pt | +400 bp | **−785,466** | −47.3% | −1,111,457 | −41.5% |
| Historical | Lehman / GFC 2008 | −30.0% | +45 pt | −150 bp | −607,146 | −36.5% | −1,082,809 | −78.3% |
| Historical | COVID crash 2020 | −34.0% | +60 pt | −150 bp | −484,394 | −29.1% | −916,575 | −89.2% |
| Historical | Black Monday 1987 | −20.4% | +50 pt | −100 bp | −451,436 | −27.2% | −306,290 | +32.2% |
| Historical | Volmageddon 2018 | −4.1% | +20 pt | −10 bp | −98,302 | −5.9% | −69,418 | +29.4% |
| Historical | Aug 2024 carry unwind | −3.0% | +38 pt | −20 bp | −1,268 | −0.1% | +188,971 | (sign flip) |
| Hypothetical | Spot −10%, vol unchanged | −10.0% | 0 pt | 0 bp | −437,670 | −26.3% | −423,066 | +3.3% |
| Hypothetical | Rates +200bp, spot flat | 0.0% | 0 pt | +200 bp | +74,108 | +4.5% | +72,271 | −2.5% |
| Hypothetical | Spot flat, vol −30% | 0.0% | −30 pt | 0 bp | +111,549 | +6.7% | +241,252 | +116.3% |
| Hypothetical | Spot +10%, vol −10 pt | +10.0% | −10 pt | 0 bp | +263,679 | +15.9% | +308,376 | +17.0% |
| Hypothetical | Spot +15%, vol +15 pt | +15.0% | +15 pt | 0 bp | +349,167 | +21.0% | +265,071 | −24.1% |

Median |Taylor error| across the eleven: **32%**. The Aug 2024 row is the
sharpest case — the expansion returns +$188,971 where repricing returns
−$1,268, so it is wrong about the *sign*, not merely the size. This is §4.3
measured rather than asserted.

Worst scenario: 2022 rates repricing, −$785,466, **−47.3% of book value**.

**The spot × vol grid finds worse, and finds it in the interior.**
`output/stress_grid.csv` reprices a 13 × 13 grid over spot ±30% and vol −20 to
+40 points. The worst cell is **spot −20% / vol −15 points: −$1,582,048, or
−95.2% of book value** — nearly the whole book. Per §4.6 that figure sits on the
3% vol floor and is a lower bound. It is an interior cell, not a
corner: the same vol shock at spot −30% loses only $954,270.

Decomposed leg by leg, the $627,778 of recovery between those two cells comes
almost entirely from one position. The long 100 puts struck at 625 go from
+$50,271 to +$813,709 as they move deep into the money (+$763,438), while the
short 80 puts struck at 560 give back only $108,740 of it. Every call leg is
already worthless at both spot levels and contributes nothing to the
difference. The book is net long puts, so past roughly −20% its convexity turns
back in its favour and the loss curve bends up again.

A scenario set that tests only corners misses this cell by $628,000. That is the
argument for running the grid rather than a scenario list, and it is the reason
the reverse stress test in §5.4 is the section a desk can act on.

---

## 6. Ongoing monitoring plan

| Item | Frequency | Trigger for escalation |
|---|---|---|
| VaR exception count | Daily | 2 exceptions in 10 days, or yellow zone |
| Christoffersen independence | Monthly | p < 0.05 |
| Estimator divergence (parametric vs full reval) | Monthly | Gap > 50% |
| Bucket vega vs parallel vega ratio | Weekly | Ratio > 2 (signals §4.1 binding) |
| Recalibration of the vol surface | Quarterly, or after any vol regime shift | — |

---

## 7. Validation conclusion and limitations of this validation

**Conclusion:** the framework is fit for its stated purpose — monitoring and
sizing market risk on a small listed option book — subject to the §4.1
limitation, which is material and quantified rather than merely disclosed.

**Qualification from the outcomes analysis, added after the run.** That
conclusion covers the *framework*. It does not extend to one of the estimators
inside it. §5.2 backtests historical VaR on a 125-day rolling window and Kupiec
rejects it at 95% (12 exceptions against 6.50 expected, p = 0.047) and at 97.5%
(9 against 3.25, p = 0.008), with Basel red zones at both. Christoffersen finds
no clustering at any level, so the failure is one of level, not of timing.

**The rolling-window historical estimator is therefore not fit to size risk at
95% or 97.5% as configured, and should not be used for limit-setting at those
levels until the window length is revisited.** The 99% figure is not rejected
but sits in the yellow zone at 3 exceptions against 1.30 expected, on a
130-day tested sample whose low power is stated in §5.2 — non-rejection there is
weak evidence, not a pass.

This is an outcomes-analysis failure, which is the pillar of SR 11-7 that exists
to catch exactly this. It is recorded here rather than resolved, because
resolving it means re-specifying the estimation window and re-running the
backtest, and a validation report that quietly retuned a parameter until the
model passed would not be a validation.

**Three limitations of this validation itself, stated because a validation that
does not bound its own reliability is not a validation:**

1. **Independence.** SR 11-7 requires effective challenge from parties
   independent of model development. This report was produced by the model's
   developer. It is structured to the guidance and applies the tests the
   guidance calls for, but it does not satisfy the independence requirement
   and should not be represented as though it did.

2. **Shared-model circularity.** Full revaluation and the Greek approximations
   are both computed from the same two-state-variable surface. The
   full-revaluation benchmark is therefore not independent of the models it
   benchmarks; it bounds Taylor-expansion error, not pricing-model error. The
   same circularity is why all eleven variants pass the FRTB PLA test in the
   attribution project — a result that is close to guaranteed by construction
   and, as noted there, does not validate the framework.

3. **Single-snapshot stress.** Every stress and reverse-stress number is
   computed against one dated book, 2026-08-07, with the shock applied
   instantaneously and no time decay. The grid describes that book on that day.
   It says nothing about how the exposure evolves, and the §5.5 figures should
   not be carried forward to a later position without a rerun.

**Open items:**
- Extend risk factors from one vol level to four vega buckets (§4.1)
- Replace √t scaling with overlapping-window empirical multi-day P&L (§4.2)
- Remove or raise the 3% vol floor for stress repricing, and re-run the grid;
  the current worst-cell figure is a floor-bound lower bound (§4.6)
- Re-specify the backtest estimation window and re-run Kupiec; the 125-day
  choice was driven by sample length, not by an accuracy argument (§5.2, §7)
- Quote the filtered-HS 99% figure as a λ range, not a point estimate (§5.3)
- Obtain independent review before any use beyond personal research

---

## References

- Federal Reserve SR 11-7 / OCC 2011-12, *Supervisory Guidance on Model Risk Management*
- Basel Committee, *Minimum Capital Requirements for Market Risk* (FRTB), 2019
- Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*
- Christoffersen (1998), *Evaluating Interval Forecasts*
- Barone-Adesi, Giannopoulos & Vosper (1999), filtered historical simulation
