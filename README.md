# bscpp

[![CI](https://github.com/spovalova/black-scholes-mc/actions/workflows/ci.yml/badge.svg)](https://github.com/spovalova/black-scholes-mc/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

A derivatives pricing and quantitative research toolkit: a C++ pricing core
(analytic Black-Scholes, Monte Carlo, Longstaff-Schwartz American, Heston
stochastic vol) exposed to Python via pybind11, plus a Python layer that
prices live option chains against real market data, constructs and analyzes
multi-leg strategies, fits arbitrage-checked implied vol surfaces, simulates
delta-hedging P&L with pluggable rebalancing policies and full attribution,
and aggregates portfolio-level risk. See [CHANGELOG.md](CHANGELOG.md) for
version history.

## Research finding

Most of this project replicates known results correctly (Black-Scholes,
Heston, SVI, LSM) -- useful for learning, not itself a contribution. One
piece goes further: **is Whalley & Wilmott's (1997) asymptotically-optimal
hedging band actually cost-risk-minimizing on real market data, or only
under the continuous-monitoring assumption it's derived under -- and if
it's wider than theory, is that daily (vs. continuous) rebalancing, not
knowing the true volatility, or real markets not being GBM?**

![Cost-risk objective vs. Whalley-Wilmott band multiplier, three arms, 95% bootstrap CI bands](assets/frontier.png)

`examples/hedging_policy_frontier_study.py` sweeps the WW band width (via
the exact identity `band(risk_aversion = lam0/c^3) = c * band(lam0)`,
regression-tested in `test_policies.py`) across real daily closes for 20
liquid tickers over 3 years (420 rolling out-of-sample windows) and 3
risk-aversion regimes. Each width is scored by a **scale-invariant**
mean-variance objective, `J(c) = mean(cost/premium0) + lam*mean(variance/
premium0^2)` -- cost and variance normalized by each window's own option
premium *before* pooling, not after (see `bscpp.backtest.frontier`).
This matters: dollar cost scales with spot*turnover and dollar variance
with spot^2*sigma^2*T, so a raw-dollar objective makes a fixed lambda mean
something different for a ~$580 SPY window than a $100 simulated path --
exactly the confound a cross-configuration comparison (real vs. simulated)
can't afford.

One of the three regimes is cost-dominated -- the objective keeps
improving to the edge of the tested grid, a methodological trap, not a
finding. The other two are well-posed (interior optimum, genuinely
trading cost against variance), and **both show a real, statistically
significant gap**: the empirically cost-risk-minimizing band is **8x wider
than theory (+35.7%, bootstrap CI [0.0054, 0.0078], n=420,
n_effective≈178)** at the moderate risk-aversion regime, and **4x wider
(+20.7%, CI [0.0049, 0.0071], n_effective≈189)** at the high one. Checked
per-ticker, not just pooled: 16/16 and 19/19 tickers with their own
well-posed optimum individually show c\*>1 -- broad-based, not one name
driving it.

Run it yourself: `python examples/hedging_policy_frontier_study.py`
(needs `POLYGON_API_KEY`, base equities tier only).

**Control experiments** (`examples/gbm_control_experiment.py`, no API key
needed) isolate *why*, running the identical sweep/objective on simulated
GBM paths at the same daily-rebalancing cadence, in two arms: `hedge_vol`
= the *true* simulation vol (isolates discretization alone), and
`hedge_vol` = a trailing-window realized-vol estimate computed exactly as
the real study computes it (isolates discretization + vol-estimation
error together). Result: **`gbm_true_vol` alone reproduces the real-data
optimum almost exactly** -- `c*=8x` (+36.5%) at the moderate regime and
`c*=4x` (+20.3%) at the high one, matching the real study's `8x`/`4x` on
this grid. Adding vol-estimation error (`gbm_estimated_vol`) does **not**
move the result closer to real data -- if anything it moves *away*
(`c*=4x` at the moderate regime, undershooting real data's `8x`) or stays
flat (`c*=4x` at the high regime, unchanged). That rules out vol-
estimation error as the primary mechanism and points squarely at
discretization: WW's continuous-monitoring assumption, violated by
ordinary once-daily rebalancing under otherwise-exact GBM dynamics, is
**sufficient on its own** to reproduce the real-data band-widening at
this resolution -- real-market structure (fat tails, volatility
clustering, autocorrelation) is not needed as an additional explanation.

Run it yourself: `python examples/gbm_control_experiment.py` (no API key
needed) followed by `python examples/plot_hedging_frontier.py` to
regenerate the figure above.

**This corrects an earlier draft of this finding** run before the
scale-invariant objective existed, on 5 tickers/50 windows instead of 20/
420, with a `RISK_AVERSIONS` grid not shared across arms. That version
reported a real-data gap of only `+7%` (`c*~2x`) against a GBM-true-vol
gap of `+20%` (`c*=4x`) -- GBM *overshooting* real data, the opposite
conclusion from the one above. The earlier numbers weren't wrong given
what they measured, but what they measured wasn't comparable across arms:
a raw-dollar objective on a small, noisy real sample against a differently
-scaled GBM config. Corrected here rather than quietly replaced -- see
CHANGELOG.

## Layout

```
cpp/                          C++ core
  include/bscpp/
    types.hpp                  shared structs (MarketInputs, Greeks, MCResult)
    black_scholes.hpp           analytic BS pricer, Greeks, IV solver, batch variants
    brent.hpp                    shared bracketed root-finder (BS + CRR implied vol)
    monte_carlo.hpp              European MC pricer (antithetic + CRN Greeks)
    longstaff_schwartz.hpp        American MC pricer (LSM regression, cross-check only)
    crr_tree.hpp                  American binomial tree (production American pricer)
    heston.hpp                     Heston stochastic-vol semi-analytic + MC pricer
  src/                          implementations + pybind11 bindings

python/bscpp/                 Python package (imports the compiled extension)
  __init__.py                  Pythonic wrappers, trading_greeks() unit conversion
  strategies.py                 multi-leg strategies: straddle/strangle/vertical/
                                 strip/strap/butterfly, net Greeks, exact breakevens
  risk.py                        portfolio Greeks aggregation + configurable limit checks
  stats.py                       inference for dependent data: effective sample size,
                                  Newey-West HAC t-stats, stationary block bootstrap,
                                  dependent-correlation CIs
  backtest/
    data_provider.py            DataProvider interface: PolygonProvider, MockProvider
    engine.py                    StripPricer / Backtester -- chain pricing vs. market
    hedging.py                   HedgingBacktester -- delta-hedging P&L, transaction
                                  costs, vega-aware attribution, pluggable policies
    policies.py                  rebalancing policies: exact-delta, fixed band, and
                                  Whalley-Wilmott (1997) asymptotically-optimal band
    vol_surface.py                SVI fitting + two independent arbitrage checks
    heston_calibration.py          fits Heston params to a chain's implied vols,
                                    with regularization + a stability diagnostic

tests/                        pytest suite (76 tests). `pytest -m "not slow"`
                               runs the fast subset (~15s); full suite includes
                               heavy MC convergence and multi-start calibration checks
examples/                     runnable demos -- all but run_backtest.py (non-mock),
                               real_data_hedging_demo.py, and
                               real_data_validation_study.py need no API key
```

## What's implemented

**C++ pricing core**
- Closed-form Black-Scholes-Merton (calls/puts, continuous dividend yield)
  and full Greeks (delta, gamma, vega, theta, rho).
- Implied-vol solver: Newton-Raphson first, falling back to **Brent's
  method** (bracket-guaranteed convergent, never divides by vega) for
  deep-ITM/OTM and near-expiry cases where Newton's derivative-based step
  is unreliable. Not a port of Jaeckel's "Let's Be Rational" (2015, the
  solver `py_vollib` uses) -- slower per call, comparable robustness in the
  well-posed regime; see `black_scholes.hpp`.
- Batch pricing/IV variants (`bs_price_with_greeks_batch`,
  `bs_implied_vol_batch`) that loop in C++ rather than crossing the
  Python/C++ boundary once per contract, plus **NumPy-native**
  `bs_price_with_greeks_batch_arrays`/`bs_implied_vol_batch_arrays`
  (`py::array_t` in and out, struct-of-arrays, zero per-contract Python
  object construction on either side of the call -- the list-of-
  `MarketInputs` variants still construct N real Python objects before
  the batch call even starts). Measured, not assumed: 7.1x/16.6x/20.5x
  faster than the list-based path at n=13/100/500 contracts -- the win
  grows with chain size because it's dominated by avoiding N object
  constructions, not by the C++ compute loop itself being faster (isolate
  just the call, ignoring construction, and the gap shrinks to ~1.2x).
  `StripPricer` and `calibrate_heston`'s IV-space residual callback (the
  actual profiled hot path, ~300 calls per calibration) both use the
  array-native version now.
- Long C++ calls (`MonteCarloPricer`, `AmericanPricer`, `HestonMCPricer`,
  `heston_price`/`heston_price_batch`, `crr_price`/`crr_implied_vol`, the
  batch functions above) release the GIL for the duration of the call, so
  other Python threads aren't blocked -- applied selectively, not as a
  blanket default: releasing/reacquiring the GIL has its own small fixed
  cost, so it's skipped on sub-microsecond calls (`bs_price` etc.) where
  that cost would dominate rather than pay for itself.
- **OpenMP** parallelizes the path loops in `MonteCarloPricer`,
  `AmericanPricer` (LSM), and `HestonMCPricer` -- each path/output index
  seeks to its OWN disjoint Philox counter position before drawing (see
  `philox.hpp`), so results are reproducible regardless of thread count:
  confirmed directly (`test_openmp_determinism.py`) that the underlying
  draws are exactly reproducible at any thread count, while the
  *aggregated* price/std_error can differ in the last few ULPs at scale
  -- OpenMP's `reduction(+:...)` sums per-thread partial results in a
  thread-count-dependent order, and floating-point addition isn't
  associative. That's standard, expected behavior for any parallelized
  numerical reduction (BLAS/LAPACK included), stated precisely here
  rather than oversold as "bit-identical." Measured speedup (Apple M4
  Pro, 8 performance cores): European MC 6.1x at 8 threads (46.0ms ->
  7.6ms), Heston MC 7.4x (919.1ms -> 124.7ms), LSM only 2.0x (594.4ms ->
  294.8ms) -- capped by Amdahl's law: only path *generation* is
  parallelized, not LSM's backward-induction regression step, which is a
  real fraction of its total cost. The build detects OpenMP via an actual
  compile-and-link check (`setup.py`), not a platform assumption, and
  degrades gracefully to a correct sequential build if it's unavailable
  -- `#pragma omp` directives are silently ignored by a compiler that
  doesn't have OpenMP enabled, so a failed/skipped detection costs
  parallelism, not correctness. Verified working end-to-end on macOS
  (Apple Clang + Homebrew `libomp`) in this environment; Linux (`-fopenmp`)
  and Windows (`/openmp`) use standard, well-established flags but aren't
  independently verified here -- CI (3 OSes) is the real cross-platform
  test.
- **Philox4x64-10** (Salmon et al. 2011) counter-based RNG (`philox.hpp`)
  underlies every Monte Carlo pricer below, replacing `std::mt19937_64`.
  Two things this actually buys, not just "a different generator": (1)
  cross-platform reproducibility that's *verified*, not assumed -- raw
  output is checked bit-for-bit against `numpy.random.Philox` across 7
  seeds including edge cases (`test_philox.py`), the strongest available
  portability claim (matching an independent, already-portable reference
  implementation exactly, not merely being internally self-consistent
  the way `std::mt19937_64` + `std::normal_distribution` looked before
  the earlier fix -- see CHANGELOG); (2) counter-based generation means
  `seek(counter)` reaches any point in the stream directly, with no need
  to replay prior draws -- confirmed disjoint, non-overlapping streams
  from the same seed via a `stream` parameter (`AmericanPricer`'s
  calibration/pricing path sets now use this instead of an arbitrary
  seed offset), which is what makes per-thread streams for parallelized
  path generation possible with zero coordination. Measured cost, not
  assumed free: ~1.45x slower per normal draw than the previous
  `mt19937_64`-backed version (25.8ns vs 17.8ns, Apple M4 Pro) -- Box-
  Muller's transcendental functions dominate enough that this is well
  under the ~3.9x gap in raw generator throughput alone.
- Monte Carlo European pricer under GBM: antithetic variates for variance
  reduction, Greeks via bump-and-reprice with common random numbers. The
  reported standard error is computed over antithetic **pair means** (the
  correct i.i.d. unit for a negatively-correlated antithetic sample; see
  CHANGELOG for the fix history), calibration-tested against realized
  estimator dispersion across independent seeds.
- American-style pricing via Longstaff-Schwartz (2001) least-squares Monte
  Carlo, using two independently-seeded path sets (calibration and
  pricing) -- matches QuantLib's `MCLongstaffSchwartzEngine`, avoiding the
  look-ahead bias of regressing and pricing on the same paths. Validated
  against the paper's own benchmark (S=36, K=40, r=6%, vol=20%, T=1y ->
  ~4.47-4.48) and the no-early-exercise identity for dividend-free calls.
  **Not this project's production American pricer** (see below) -- kept
  as an independent cross-check on the CRR tree and for the path-
  dependent/multi-factor generalizations Monte Carlo is suited for and a
  binomial tree isn't.
- Dividend-aware Cox-Ross-Rubinstein binomial tree (`crr_price`,
  `crr_implied_vol`, `bscpp.price_american_crr`) -- the actual production
  American pricer: deterministic (no simulation noise), microseconds per
  price at 200 steps, and what `StripPricer(american=True)` uses to solve
  American implied vol in the chain pipeline instead of the closed-form
  European Black-Scholes formula (see "Chain pricing" below for why that
  swap matters). Cross-checked against LSM and against the same
  Longstaff-Schwartz (2001) benchmark above.

**Rate curve** (`bscpp.curve.ZeroCurve`)
- Minimal piecewise-flat, continuously-compounded zero-rate curve --
  `df(t)`, `zero_rate(t)`, `forward_rate(t1, t2)` -- not a curve-
  bootstrapping engine (building one from real market instruments is a
  materially larger, different undertaking, out of scope here; see the
  scope statement below). `StripPricer`, `HedgingBacktester`, and
  `calibrate_heston` all accept either a bare float (a flat rate) or a
  `ZeroCurve` via `resolve_rate`, resolved to the maturity-appropriate
  scalar the underlying single-rate C++ pricers actually need. None of
  the three defaults `rate` anymore -- a hardcoded 0.05 is exactly the
  kind of assumption that shouldn't be silently inherited by every
  caller. See `examples/real_data_hedging_demo.py` for an end-to-end
  multi-pillar curve, not just a unit test of the class in isolation.

**Day-count clock** (`bscpp.clock.Clock`)
- One explicit day-count convention -- `ACT/365` (calendar days, default)
  or `TRADING/252` -- instead of `365`/`252` appearing as an unexamined
  magic number at each call site. `year_fraction`, `time_to_expiry`,
  `elapsed` (a floored-at-one-day step length), and
  `annualized_realized_vol` all go through it; `StripPricer` and
  `HedgingBacktester` both take a `clock=` parameter (defaulting to
  `Clock()`, i.e. ACT/365) instead of computing day-counts inline.
  Concrete motivation, not a hypothetical: this project's own history has
  a real bug where a validation study scaled trading-day returns by
  `sqrt(365)` while the backtester it was validating accrued calendar
  time, inflating a reported result ~20% (see CHANGELOG's "vol-clock
  inconsistency" fix) -- exactly the class of bug one shared, explicit
  clock is meant to structurally prevent. The three example scripts that
  each hand-rolled their own (duplicated, previously drift-prone)
  realized-vol estimator now call `Clock().annualized_realized_vol`
  instead -- confirmed bit-for-bit identical to the old formula before
  the swap (`test_clock.py`), so none of the already-published research
  numbers changed.

**Multi-leg strategies** (`bscpp.strategies`)
- `straddle`, `strangle`, `vertical_spread`, `butterfly`, `strip` (long 1
  call + 2 puts, bearish-biased), `strap` (long 2 calls + 1 put,
  bullish-biased).
- `StrategyPricer.price()` returns net Greeks across all legs (accepts a
  per-strike vol dict for skew-aware pricing). `payoff_diagram()` computes
  the expiration P&L curve and exact breakevens via linear interpolation
  between each leg's strike (the payoff of a vanilla-option portfolio is
  piecewise-linear in spot, so this is exact, not a numerical root-find).

**Implied vol surface** (`bscpp.backtest.vol_surface`)
- `fit_svi_slice`: least-squares fit of Gatheral's SVI parameterization
  (`w(k) = a + b(rho(k-m) + sqrt((k-m)^2+sigma^2))`) to a chain's implied
  vols.
- Two independent no-arbitrage checks: `svi_butterfly_arbitrage_check`
  applies **Breeden-Litzenberger (1978)** numerically (prices calls off
  the fitted smile, takes a finite-difference second derivative in
  strike); `svi_gatheral_jacquier_check` implements the closed-form
  `g(k) >= 0` condition from Gatheral & Jacquier (2013) directly on the
  SVI parameters. Both enforce the `w(k) > 0` precondition and are
  confirmed to agree on both a well-behaved and a pathological test slice.
- Known gap: fitting slices independently gives no guarantee of
  consistency *across* expiries (no calendar-spread arbitrage guarantee);
  that's what Gatheral & Jacquier's SSVI extension provides and this
  project doesn't implement.

**Delta-hedging P&L + attribution** (`bscpp.backtest.hedging`)
- `HedgingBacktester`: sells an option and delta-hedges it against a real
  or simulated price path, correctly crediting dividend income on the
  stock leg when `dividend_yield > 0`.
- `transaction_cost_bps`: cost of crossing the bid-ask spread on every
  trade. Flows through `attribute_pnl` as an exact term (not a Taylor
  approximation), leaving `attribution_error` unchanged at any cost level.
- `attribute_pnl`: decomposes daily P&L into financing + gamma + theta
  terms from a second-order Taylor expansion of the option's own pricing
  function (derivation in the method's docstring). The decomposition's
  daily sum is an exact accounting identity against the simulation's own
  cumulative P&L, and on a deterministic path collapses to Carr & Madan's
  (2002) closed-form limit `0.5*Gamma*S^2*(hedge_vol^2 - realized_vol^2)*dt`
  to within ~0.5%.
- `hedge_vol` may be a time series (re-marked at each date); `attribute_pnl`
  then reports an explicit **vega** term, identically zero in the
  constant-vol configuration.
- **Rebalancing policies** (`bscpp.backtest.policies`): `DeltaPolicy`
  (baseline), `BandPolicy` (fixed no-trade band), and
  `WhalleyWilmottPolicy` (1997 asymptotically-optimal band; published
  Gamma^(2/3)/cost^(1/3)/risk-aversion^(-1/3) scalings verified numerically
  to <1e-9). Band policies cut total spread cost >30% vs. daily
  rebalancing in this suite's own simulated-path tests. `CallablePolicy`
  is a plug-in point for custom or learned (e.g. RL) policies.
- **`bscpp.stats`**: effective sample size, Newey-West HAC t-stats,
  Politis-Romano stationary block bootstrap, dependent-correlation CIs.
  No function returns a bare p-value -- every result carries its
  assumptions (HAC lags, block length, effective N) in its repr.

**Chain pricing vs. real market data** (`bscpp.backtest.engine`)
- `StripPricer`/`Backtester`: pull a chain slice from a `DataProvider`,
  solve implied vol per contract (batched), price with both BS and MC,
  and report theoretical price vs. observed market mid. Every row carries
  an `iv_source` column (`quoted`/`solved`/`fallback`); fallback rows are
  NaN rather than a silently invented vol.
- `extract_forward_and_carry`: the implied forward and cost-of-carry from
  put-call parity at the strike minimizing `|C-P|` (the standard desk
  recipe -- that strike's parity estimate is least contaminated by wide
  bid-ask spreads and early-exercise premium, both of which grow away
  from the money). `StripPricer` uses the result two ways: it solves IV
  **OTM-only** (calls above the implied forward, puts below -- deep-ITM
  prices are almost pure intrinsic value, an ill-conditioned IV solve,
  and for American-style equity options carry an early-exercise premium
  the European solver can't account for), and it prices off the
  **market-implied dividend/carry** (`q = r - implied_carry`) instead of
  an assumed `dividend_yield` whenever the chain has paired call/put
  quotes -- replacing an assumption with a market-implied number, not
  just reporting one alongside it. `implied_forward`/`implied_carry` are
  reported as chain columns either way.
- `StripPricer(..., american=True)`: solves IV against the dividend-aware
  CRR binomial tree instead of closed-form European Black-Scholes, and
  reports a `crr_price`/`crr_error_vs_market`/`crr_error_pct` alongside
  the always-computed European `bs_price` view. Real equity chains are
  American-style; solving a European IV from an American market price
  silently absorbs an early-exercise premium the European formula can't
  represent -- confirmed directly (not just argued): at the same market
  price, the American-consistent solve infers a measurably LOWER put IV
  than the European solve does, exactly the mismatch this removes (see
  `test_strip_pricer_american_mode_reports_crr_price_and_higher_put_iv_
  solve`). Defaults to `False` -- not because European is preferred, but
  because `MockProvider`'s synthetic chain generates its own "true"
  prices via European BS internally, so `american=True` against
  `MockProvider` specifically would be a self-consistency mismatch, not a
  more realistic test; against real data (`PolygonProvider`) it's the
  more realistic choice. `bs_price`/Greeks remain the European values at
  whichever IV was solved either way -- CRR has no closed-form Greeks, so
  those stay an explicit approximation in `american=True` mode, not exact
  American sensitivities.

**Portfolio risk aggregation** (`bscpp.risk`)
- `PortfolioRiskManager`: true DOLLAR delta/gamma (not raw share-equivalent
  Greeks) across multi-underlying `Position`s, grouped by underlying, with
  configurable `RiskLimits` and breach flagging. Dollar delta = `delta *
  quantity * spot`; dollar gamma = `gamma * quantity * spot^2 / 100` (the
  standard "$ change in dollar delta per 1% move" convention) -- unlike
  `StrategyPricer`'s raw per-share Greeks (correct there, since one
  strategy never mixes underlyings), a cross-underlying limit checked in
  share-equivalent units isn't actually comparable across names of
  different spot prices, which defeats the point of a portfolio-level
  limit. This is dollar-Greeks aggregation with limits, not a production
  risk system -- no live feed, margin model, or kill switch.

**Heston stochastic volatility** (`bscpp.heston_price`, `bscpp.backtest.heston_calibration`)
- Semi-analytic pricer via the Heston (1993) characteristic function,
  using the Albrecher, Mayer, Schoutens & Tistaert (2007) **"Little
  Trap"** reformulation (avoids the branch-cut discontinuity in the
  original formula's complex logarithm). Confirmed term-by-term against
  QuantLib's `AnalyticHestonEngine`.
- An independent Monte Carlo pricer (full-truncation Euler for the CIR
  variance process) cross-checks the characteristic-function formula:
  collapses to Black-Scholes as vol-of-vol -> 0, agrees with MC across
  strikes/types/a Feller-violating stress case, exact put-call parity.
- The P1/P2 integrals use adaptive Simpson quadrature with adaptive
  upper-bound extension rather than a fixed-node table, self-terminating
  on measured error.
- `calibrate_heston` fits (kappa, theta, xi, rho, v0) in **IV space**
  (price-space residuals underweight the OTM wings, where the vol
  information actually lives), with Tikhonov-style regularization toward
  sane priors -- Heston has a known identifiability issue where `v0` can
  otherwise land on a degenerate corner of the loss surface.
  `calibrate_heston_with_stability` runs several perturbed initial
  guesses and reports whether fit quality *and* the parameters themselves
  are stable across starts.
- `heston_price_batch` prices a whole strike grid in one call by sharing
  characteristic-function evaluations across strikes (the CF doesn't
  depend on strike, only its phase factor does) over a fixed, not
  adaptive, quadrature grid -- profiling showed `heston_price` accounting
  for 96.8% of a calibration call's runtime, and `calibrate_heston` now
  uses this path internally. Fixed quadrature can't inherit the adaptive
  pricer's self-terminating accuracy for free, so resolution is chosen by
  measured error rather than assumed: a maturity **and** vol-of-vol sweep
  against the adaptive price found both short maturity and high xi
  independently degrade a fast, cheap grid, so `calibrate_heston` falls
  back to a validated-accurate higher-resolution grid outside that
  regime. Net effect on a 9-strike calibration: 254ms -> 82ms (3.1x),
  fitted parameters and fit RMSE unchanged. See
  `_batch_resolution_for_maturity` in `heston_calibration.py` for the
  measured thresholds and a disclosed residual limitation at
  near-degenerate correlation.
- `heston_satisfies_feller_condition`: checks `2*kappa*theta >= xi^2` as a
  diagnostic (many market-calibrated fits violate it in practice).

**Trading-desk Greeks** (`bscpp.trading_greeks`)
- Converts calculus Greeks (vega/rho per 1.00 of vol/rate, theta per year)
  to desk convention (vega/rho per 1%, theta per day).

## Scope

What this project deliberately does not do, and why. Explicit scoping
turns an omission into a stated boundary instead of leaving a reviewer to
guess whether it's an oversight.

**Pricing and vol surface**
- No local vol (Dupire) or SSVI/other cross-expiry-consistent surface
  model. `fit_svi_slice` fits one expiry at a time with no guarantee of
  calendar-spread consistency across expiries -- that consistency is
  exactly what SSVI's extension provides, and it isn't implemented here.
- No stochastic vol model beyond Heston (SABR, rough vol). Heston is
  implemented end-to-end (semi-analytic pricer, independent MC
  cross-check, calibration); adding a second SV model is a different,
  larger undertaking than deepening the one already here.
- No curve-*bootstrapping* engine. `ZeroCurve` takes zero rates as direct
  input pillars; building a curve from real market instruments (SOFR
  futures, bond yields) is a materially different undertaking from using
  an already-built one.
- No separate repo/overnight financing curve. `HedgingBacktester` prices
  the option AND accrues the cash leg's financing at the option's own
  remaining-maturity rate -- a stated simplification (see its `__init__`
  docstring), not a real desk's independently-curved repo rate.
- No full trading-day calendar (exchange holidays, session-specific
  hours). `Clock`'s `TRADING/252` convention approximates trading days as
  5/7 of calendar days; a real holiday calendar is out of scope.
- No exact American Greeks. `StripPricer(american=True)` reports the
  European Black-Scholes Greeks at the CRR-solved IV, not true American
  sensitivities -- CRR has no closed form for those, and bump-and-reprice
  through the tree hasn't been added.
- No Andersen-Broadie duality bounds on LSM. American Monte Carlo prices
  remain point estimates with an MC standard error, not a certified
  [lower, upper] interval.
- No thread-safety for concurrent calls on the SAME `MonteCarloPricer`/
  `AmericanPricer`/`HestonMCPricer` instance. Releasing the GIL for these
  calls (see above) means two Python threads calling `.price()` on the
  *same* instance at the same time would race on that instance's internal
  path-generation counter -- a real, not hypothetical, consequence of
  the GIL-release change. Safe pattern: one pricer instance per thread
  (cheap to construct); this project doesn't add locking or any other
  cross-call synchronization on top of that.

**Data and execution**
- No full historical options-tick database. `Backtester`'s multi-day loop
  is explicit about pricing off the *current* chain repeatedly unless the
  provider supports genuine historical snapshots (Polygon's do, on a paid
  tier above this project's key).
- No real-time market data feed, order/execution management, or margin
  and capital modeling -- the difference between "correct pricing math"
  and "safe to connect to a real trading account," a different, much
  larger undertaking than this project.
- No kill switch, for the same reason.
- No market-impact model. Transaction costs are a flat bps-of-notional
  charge on every trade; there's no size-dependent slippage/impact curve.

**Research methodology**
- No claim that the hedging-band finding (see "Research finding" above)
  generalizes beyond the regimes actually tested. Risk-aversion regimes
  without a genuine interior optimum are reported as inconclusive
  grid-boundary artifacts, not papered over with a number a wider search
  would just move again.

## External benchmarks

"3.1x faster than my own previous version" (the `heston_price_batch`
CHANGELOG entry) is a self-referential claim -- it says nothing about
whether the *result* is fast in any absolute sense. `benchmarks/`
compares this project's pricers against two independent, established
references on identical inputs: [QuantLib](https://www.quantlib.org/)
1.43 (analytic BS, CRR binomial American at the same step count, analytic
Heston) and [vollib](http://vollib.org/) 1.0.11 (analytic BS/BSM price
and implied vol; vollib doesn't price American or Heston, hence no third
column there). Every benchmark file asserts numerical agreement with the
reference within a stated tolerance *before* timing anything -- a fast
wrong answer isn't a result (see each file's `test_*_correctness`).

Hardware: Apple M4 Pro, macOS 15.1, Python 3.13.5, single-threaded,
`pytest-benchmark` 5.2.3, median of 30-150k rounds depending on the test
(warm cache, `time.perf_counter`). QuantLib's `NPV()` is cached by its
lazy-evaluation object model until a watched quote changes -- confirmed
by direct measurement (~6x faster for a cached call vs. a genuinely
recomputed one) -- so every QuantLib benchmark perturbs the spot quote by
an alternating +-1e-10 before each call to force real recomputation; see
`benchmarks/conftest.py`.

| Pricer | bscpp | QuantLib | vollib | bscpp vs. faster reference |
|---|---:|---:|---:|---|
| BS price | 0.07us | 1.50us | 1.61us | **21x faster** |
| BS implied vol | 0.12us | 4.79us | 7.59us | **39x faster** |
| American (CRR, 500 steps) | 84.9us | 516.3us | -- | **6x faster** |
| Heston price | 336.2us | 24.4us | -- | **14x SLOWER** |

The Heston loss is real and already explained in the code, not a
surprise found here: `heston.hpp` deliberately uses adaptive Simpson
quadrature (self-terminating on measured error, no risk of a
transcribed-table copying error) rather than the fixed-node quadrature
table production engines like QuantLib's `AnalyticHestonEngine`
typically use for speed. `heston_price_batch` (see CHANGELOG) already
takes the fixed-node approach for the calibration hot path specifically,
where the same characteristic-function evaluations are shared across a
whole strike grid -- this single-call, single-strike benchmark is
exactly the case that optimization doesn't cover, so the gap shown here
is the honest cost of the adaptive path a cold single-price call still
pays in full. A fixed-node COS/FFT engine would likely close most of
this gap for the single-price case too, at the cost of carrying a second
Heston pricer to keep cross-checked against this one -- a real tradeoff,
not yet made.

Run it yourself: `pip install -e ".[benchmark]"` then
`pytest benchmarks/ --benchmark-only --benchmark-columns=min,mean,stddev,median,rounds`.

## Setup

Requires a C++17 compiler (Xcode command line tools on macOS) and Python
3.9+. No CMake needed -- the extension builds via `pybind11.setup_helpers`.

OpenMP (parallelizes the Monte Carlo/LSM/Heston-MC path loops -- see
"What's implemented") is optional and auto-detected: `setup.py` actually
compiles and links a trivial OpenMP program before enabling it, so a
missing toolchain silently builds sequential instead of failing. On Linux
and Windows this typically just works (`-fopenmp`/`/openmp` are part of
the standard toolchain); on macOS, Apple Clang doesn't bundle OpenMP --
`brew install libomp` first if you want it enabled (fully optional; the
build and every pricer are correct either way, just single-threaded
without it).

```bash
cd black-scholes-mc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q -m "not slow"   # fast suite, ~15s
```

## Run the demos (no API key needed)

```bash
python examples/american_pricing_demo.py     # European vs. American (CRR, cross-checked vs LSM)
python examples/strategy_demo.py               # straddle/strangle/vertical/strip/strap/butterfly
python examples/portfolio_risk_demo.py          # net Greeks + limit breaches across a portfolio
python examples/hedging_pnl_experiment.py       # realized-vs-implied vol P&L sweep + transaction costs
python examples/vol_surface_fit_demo.py         # SVI fit + arbitrage check on a synthetic smile
python examples/heston_calibration_demo.py       # Heston calibration + stability diagnostic vs. SVI
python examples/run_backtest.py --mock --ticker SPY   # full chain-pricing pipeline
```

Sample output from `hedging_pnl_experiment.py` (selling an ATM call, hedging
daily at 30% vol, 60 days to expiry, 400 simulated paths per realized-vol
level, frictionless):

```
 realized_vol  hedge_vol  mean_hedging_pnl  std_hedging_pnl
         0.10       0.30              3.19             0.66
         0.20       0.30              1.61             0.61
         0.30       0.30             -0.01             0.59
         0.40       0.30             -1.64             1.06
         0.50       0.30             -3.24             1.75
```

## Run against real data (needs an API key, no options entitlement required)

```bash
python examples/real_data_hedging_demo.py --ticker AAPL --hedge-vol 0.30
python examples/real_data_validation_study.py
```

`real_data_hedging_demo.py` pulls real historical daily closes, runs the
delta-hedging backtest against them, and prints the P&L attribution
(including transaction costs) -- works on Polygon/Massive's **base**
equities tier (the daily aggregates endpoint doesn't require an options
plan). Sample attribution output from a real run against AAPL:

```
financing_pnl          -1.82
gamma_pnl               -6.80
theta_pnl                6.99
transaction_cost_pnl    -0.39
predicted_pnl           -2.01
realized_pnl            -2.26
attribution_error       -0.25
```

`real_data_validation_study.py` is the multi-ticker, multi-window version:
real daily closes for 5 liquid names (AAPL, MSFT, GOOGL, AMZN, SPY) over
~9 months, rolled forward across 10 overlapping 45-day windows per ticker
using trailing realized vol as the `hedge_vol` forecast. Across 50
ticker-windows:

```
Hit rate (sign(hedge_vol - forward_realized_vol) == sign(hedging_pnl)): 80.0%
Correlation(vol_gap, hedging_pnl): r=0.877
```

Two caveats reported alongside the number, not left implicit: the windows
overlap within each ticker and the tickers share a market vol regime, so
the effective sample size is well below 50 (no i.i.d. p-value is
reported, for that reason); and the relationship being reproduced is close
to a mechanical identity (Carr & Madan 2002), so this is a
software-correctness check on real data, not evidence of a tradable edge.

`run_backtest.py`/`StripPricer` (chain snapshots) is the one demo that
needs an options-capable data plan -- see below.

## Real market data

The backtester targets Polygon.io's REST API (rebranded "Massive";
`PolygonProvider` still hits `api.polygon.io`). Options chain data
requires at least an "Options Starter" plan (`403 NOT_AUTHORIZED`
otherwise); daily underlying price history (`get_price_history`, used by
the hedging backtest) works on the base equities tier. Historical chain
*snapshots* for arbitrary past dates require a higher tier still -- see
the `Backtester` docstring in `python/bscpp/backtest/engine.py`. Paginated
endpoints follow `next_url` to completion with retry/backoff on
rate-limit/5xx responses.

Set your key via environment variable or a local `.env` (gitignored):

```bash
cp .env.example .env
# edit .env and set POLYGON_API_KEY=...
```

Then:

```bash
python examples/run_backtest.py --ticker SPY --expiration 2026-09-19
```

## Python API quick reference

```python
import bscpp

# Analytic price / Greeks
price = bscpp.price(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, option_type="call")
inputs = bscpp.make_inputs(100, 100, 0.05, 0.2, 1.0, "call")
greeks = bscpp.bs_greeks(inputs)
desk_greeks = bscpp.trading_greeks(greeks)  # vega/rho per 1%, theta per day

# Monte Carlo (European) and American (CRR tree -- production; LSM -- cross-check only)
mc = bscpp.price_mc(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, num_paths=200_000)
am = bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=200)
am_lsm = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=100_000, num_steps=50)

# Heston stochastic vol (semi-analytic; HestonMCPricer for the independent MC cross-check)
hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)
heston_price = bscpp.heston_price(100, 100, 0.05, 0.0, 1.0, bscpp.OptionType.Call, hp)

# Multi-leg strategies
strat_pricer = bscpp.StrategyPricer(rate=0.05)
result = strat_pricer.price(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)
plot_df, breakevens = strat_pricer.payoff_diagram(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)

# Portfolio risk aggregation
positions = [
    bscpp.Position("AAPL call", "call", quantity=10, underlying="AAPL", spot=200, rate=0.05,
                    strike=210, vol=0.25, maturity=0.5),
    bscpp.Position("AAPL hedge", "stock", quantity=-500, underlying="AAPL", spot=200, rate=0.05),
]
risk_mgr = bscpp.PortfolioRiskManager(bscpp.RiskLimits(max_abs_vega=50))
net = risk_mgr.net_greeks(positions)
breaches = risk_mgr.check_limits(positions)
```

```python
from bscpp.backtest import (
    MockProvider, StripPricer, HedgingBacktester,
    fit_svi_slice, svi_butterfly_arbitrage_check, svi_gatheral_jacquier_check,
    calibrate_heston, calibrate_heston_with_stability,
)

provider = MockProvider(spot=450.0, base_vol=0.18)
pricer = StripPricer(provider, rate=0.05)
chain = pricer.price_strip("SPY", expiration, strike_range=(0.9, 1.1))

svi = fit_svi_slice(chain["strike"], chain["model_iv"], spot=450.0, t_years=chain["T"].iloc[0])
arb = svi_butterfly_arbitrage_check(svi, spot=450.0, rate=0.05)       # numerical (Breeden-Litzenberger)
arb2 = svi_gatheral_jacquier_check(svi)                                # closed-form g(k), cross-check

heston = calibrate_heston(chain["strike"], chain["type"], chain["model_iv"],
                           spot=450.0, t_years=chain["T"].iloc[0], rate=0.05)
stability = calibrate_heston_with_stability(chain["strike"], chain["type"], chain["model_iv"],
                                             spot=450.0, t_years=chain["T"].iloc[0], rate=0.05)

hedger = HedgingBacktester(rate=0.05, transaction_cost_bps=5.0)
result = hedger.run(price_history, strike=450, expiration=expiration, hedge_vol=0.18)
attributed = hedger.attribute_pnl(result)  # financing / gamma / theta / transaction cost breakdown
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 -- see `LICENSE`.
