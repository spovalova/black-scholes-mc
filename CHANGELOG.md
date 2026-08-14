# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased] - M1: hedging laboratory core

The three capabilities that make the hedging backtester a research
instrument rather than a demo: optimal rebalancing policies, vega-aware
attribution, and publication-grade statistics. 18 new tests (76 total).

### Research

- **`bscpp.backtest.frontier`** (new, shared by both scripts below):
  extracted the band-multiplier x risk-aversion grid runner and scoring
  pipeline that both frontier scripts had been duplicating, and fixed a
  real methodological gap in the process -- the mean-variance objective
  (`cost + lambda*variance`) was scored in raw dollars, so a fixed
  `lambda` meant something different for a ~$580 SPY window than a $100
  GBM path (dollar cost scales with spot*turnover, dollar variance with
  spot^2*sigma^2*T). Normalized both terms by each window's own option
  premium *before* pooling across windows, making `c*` genuinely
  comparable across tickers of different price levels and across real
  vs. simulated arms. This changes the numbers below materially from an
  earlier (uncorrected) draft of this finding -- see the last bullet.
- **`examples/hedging_policy_frontier_study.py`**: tests whether
  Whalley-Wilmott's (1997) asymptotically-optimal hedging band is actually
  cost-risk-minimizing on real market data. Widened from 5 tickers/50
  windows to 20 tickers/420 windows (3 years each), and added a per-ticker
  breakdown so the finding isn't just a pooled number. 2 of 3
  risk-aversion regimes are well-posed (interior optimum): `c*=8x` theory
  (+35.7%, bootstrap CI excludes 0, n=420) and `c*=4x` theory (+20.7%),
  both broad-based (16/16 and 19/19 tickers individually show `c*>1` in
  their own well-posed regime). See the README's "Research finding".
- **`examples/gbm_control_experiment.py`**: now runs TWO control arms
  isolating discretization from vol-estimation error, not one. `hedge_vol`
  = true simulation vol isolates "WW assumes continuous monitoring, this
  only rebalances daily" alone; `hedge_vol` = a trailing-window estimate
  (computed identically to the real study) adds vol-estimation error on
  top, on the same underlying paths. Finding: `gbm_true_vol` alone
  reproduces the real-data optimum almost exactly (`c*=8x`/`4x`, matching
  the real study's `8x`/`4x` on this grid); adding estimation error does
  NOT move the result closer to real data. Discretization, not vol-
  clustering/fat tails and not vol-estimation error, is sufficient to
  explain the real-data finding at this resolution.
- **`examples/plot_hedging_frontier.py`** (new): renders `assets/
  frontier.png` -- J(c) vs. c, one panel per risk-aversion regime, one
  line + shaded 95% bootstrap CI per arm, theory (c=1) and each arm's
  empirical optimum marked. Reads the grid CSVs the two scripts above
  save to `examples/output/`, so it's fast to iterate on without
  re-fetching market data or re-running thousands of simulated hedges.
- **Correction**: an earlier draft of this finding (5 tickers/50 windows,
  a single GBM arm, a raw-dollar objective) reported the *opposite*
  qualitative conclusion -- real data `+7%`/`c*~2x` vs. GBM-true-vol
  `+20%`/`c*=4x`, i.e. GBM overshooting real data, suggesting real-market
  structure pulls the optimum back toward theory. That comparison wasn't
  apples-to-apples: the raw-dollar objective conflated the real study's
  ~$580 SPY-scale windows with the GBM control's $100-scale paths, and
  the small real sample made the pooled estimate noisier. Corrected here
  rather than silently replaced; the corrected, larger-sample,
  scale-invariant result is the one to trust.

### Added

- **`bscpp.clock.Clock`**: one explicit day-count convention (`ACT/365`
  default, `TRADING/252` available) instead of `365`/`252` as an
  unexamined magic number scattered per call site. `year_fraction`,
  `time_to_expiry`, `elapsed` (floored-at-one-day step length), and
  `annualized_realized_vol` all route through it. `StripPricer` and
  `HedgingBacktester` (`run` and `attribute_pnl`, kept consistent so the
  exact P&L-decomposition accounting identity still holds) both take a
  `clock=` parameter instead of inline day-count arithmetic. Replaced 3
  duplicated hand-rolled `annualized_realized_vol` implementations
  (`hedging_policy_frontier_study.py`, `gbm_control_experiment.py`,
  `real_data_validation_study.py`) with `Clock().annualized_realized_vol`
  -- confirmed bit-for-bit identical to the old formula before swapping
  (`test_annualized_realized_vol_matches_original_inline_formula_
  exactly`), and reran `gbm_control_experiment.py` after the swap to
  confirm byte-identical output, so none of Stage 1's already-published
  numbers (see README's "Research finding") changed. Motivated by a real
  incident, not a hypothetical: this project once had exactly the bug a
  shared, explicit clock is meant to prevent (a validation study scaling
  trading-day returns by `sqrt(365)` while the backtester it validated
  accrued calendar time, inflating a result ~20% -- see the "Vol-clock
  inconsistency" entry below). 9 new tests (`test_clock.py`).
- **`bscpp.crr_price`/`crr_implied_vol`/`price_american_crr`**: dividend-
  aware Cox-Ross-Rubinstein binomial tree, now this project's production
  American pricer -- deterministic, ~14us/price at num_steps=200 (matches
  the timing claim, see `test_crr_tree.py`), converges to the Longstaff-
  Schwartz (2001) benchmark (S=36,K=40,r=6%,vol=20%,T=1y -> ~4.487, within
  1.8 std errors of the existing LSM implementation) and to European BS
  for a no-dividend call. `StripPricer(american=True)` now solves IV
  against CRR instead of closed-form European BS and reports `crr_price`/
  `crr_error_vs_market`/`crr_error_pct` alongside the always-present
  European `bs_price` view. Removes the mismatch of solving a European IV
  from an American market price (which silently absorbs an early-exercise
  premium the European formula can't represent) -- confirmed directly,
  not just argued: at the identical market price, the American-consistent
  solve infers a measurably LOWER put IV than the European solve does.
  Defaults to `american=False`: not because European is preferred, but
  because `MockProvider`'s synthetic chain generates its own "true"
  prices via European BS internally, so `american=True` against
  `MockProvider` would be a self-consistency mismatch in tests, not a
  more realistic one; `american=True` is the more realistic choice
  against real (`PolygonProvider`) data. `brent.hpp` extracted from
  `black_scholes.cpp` (previously private to that translation unit) so
  `CRRPricer::implied_vol` reuses the identical bracketed solver instead
  of a second copy -- both price monotonically in vol. LSM repositioned
  in docs/README as the cross-check and path-dependent/multi-factor
  generalization, not the production American pricer, per its actual role
  now. 8 new tests (`test_crr_tree.py`) plus a `StripPricer` integration
  test confirming the American-vs-European IV divergence direction.
- **`bscpp.curve.ZeroCurve`**: minimal piecewise-flat zero-rate curve
  (`df(t)`, `zero_rate(t)`, `forward_rate(t1, t2)`) plus `resolve_rate`,
  threaded through `StripPricer`, `HedgingBacktester` (both option pricing
  and the cash leg's financing accrual, resolved at the option's own
  remaining maturity at each step -- see `attribute_pnl`'s matching fix
  so the exact P&L-decomposition identity still holds under a real curve,
  not just a flat rate), and `calibrate_heston`/`calibrate_heston_with_
  stability`/`heston_fit_rmse`. `rate` is no longer defaulted anywhere in
  that chain -- `StripPricer.rate`, `MockProvider.rate`, and
  `realized_vs_implied_experiment`'s `rate` all became required
  parameters (previously `=0.05`); `HedgingBacktester.rate` and
  `calibrate_heston`'s `rate` were already required. Every caller
  (7 call sites) updated to pass a rate explicitly. 9 new tests
  (`test_curve.py`); full suite unaffected in the flat-rate case (`resolve
  _rate` is behavior-preserving there), demonstrated with a genuine
  3-pillar curve end-to-end in `real_data_hedging_demo.py`.
- **`bscpp.backtest.engine.extract_forward_and_carry`**: implied forward
  and cost-of-carry from put-call parity at the strike minimizing
  `|C-P|` (the standard desk recipe). `StripPricer` now (a) solves IV
  OTM-only -- calls above the implied forward, puts below -- instead of
  every row regardless of moneyness, avoiding the classic ill-conditioned
  deep-ITM IV solve and the early-exercise-premium contamination an
  American-style ITM quote carries that a European solver can't account
  for, and (b) prices off the market-implied carry (`q = r -
  implied_carry`) instead of an assumed `dividend_yield` whenever the
  chain has paired call/put quotes. `implied_forward`/`implied_carry` are
  new chain columns either way. 4 new tests (`test_backtest.py`).
  **Behavior change**: roughly half of a full two-sided chain's rows are
  now legitimately NaN (the ITM leg at each strike) rather than solved --
  correct, but it means code that filters a chain to `type == "call"` (or
  `"put"`) for downstream calibration/fitting now silently gets only the
  OTM half of the strike range. Fixed at all 3 existing call sites
  (`heston_calibration_demo.py`, `vol_surface_fit_demo.py`,
  `test_heston.py`'s `_short_dated_mock_chain`) to instead keep both
  types and drop the NaN rows -- the genuine OTM-only smile, not an
  accidentally-truncated one. One of those (the Heston v0-degeneracy
  demo/test) needed its maturity moved from 45 to 30 days: the corrected,
  properly-conditioned OTM-only smile no longer reproduces that specific
  failure mode at 45 days, confirming OTM-only solving was fixing a real
  ill-conditioning problem, not just a labeling one.
- **`bscpp.backtest.policies`** -- the rebalancing-policy ladder from the
  transaction-cost literature: `DeltaPolicy` (rebalance to exact delta,
  the baseline), `BandPolicy` (fixed no-trade band, trade to the nearest
  edge on breach), `WhalleyWilmottPolicy` (Whalley & Wilmott 1997
  asymptotically-optimal band; its published Gamma^(2/3), cost^(1/3), and
  risk-aversion^(-1/3) scalings are verified numerically to <1e-9), and
  `CallablePolicy` (plug-in point for custom/learned policies).
  `HedgingBacktester.run` takes `policy=`; default preserves the classic
  daily-rebalanced behavior exactly.
- **Vega-aware attribution**: `hedge_vol` may now be a time series (the
  option is re-marked at each date's vol); `attribute_pnl` gains explicit
  `vega_pnl` and `delta_gap_pnl` terms (the latter accounts for
  deliberately holding away from delta inside a no-trade band). Both are
  identically zero in the constant-vol, rebalance-to-delta configuration
  -- exact backward compatibility, regression-tested.
- **`bscpp.stats`** -- honest inference for dependent data: effective
  sample size (validated against AR(1) theory), Newey-West HAC t-stats,
  Politis-Romano stationary block bootstrap (coverage-tested), and
  dependent-correlation CIs (shown to avoid false certainty on
  independent-but-autocorrelated noise). No function returns a bare
  p-value; result objects carry their assumptions in their repr.
- **`heston_price_batch`**: profiling `calibrate_heston` (cProfile) showed
  `heston_price` consuming 96.8% of runtime; since the Heston
  characteristic function doesn't depend on strike, batching shares CF
  evaluations across a strike grid over a fixed (not adaptive) quadrature.
  A naively high-resolution fixed grid was tried first and was *slower*
  than the original per-strike loop (449ms vs. 217ms) -- adaptive
  quadrature is already cheap at typical parameters, so a safety-
  conservative fixed grid's per-call cost dominated at only 13 strikes.
  Made resolution configurable instead; a maturity-and-vol-of-vol sweep
  against the trusted adaptive price found a fast low-resolution default
  degrades independently under short maturity or high xi, so
  `calibrate_heston` now auto-selects resolution per calibration call.
  Net: 254ms -> 82ms (3.1x) on a 9-strike calibration, fitted parameters
  and RMSE unchanged (regression-tested).

### Fixed

- **Monte Carlo/LSM/Heston-MC seeded reproducibility was platform-
  dependent.** All three used `std::normal_distribution`, whose exact
  algorithm the C++ standard leaves implementation-defined -- libstdc++
  and libc++ produce different variate sequences from the same seed even
  though `std::mt19937_64` itself is bit-for-bit portable. CI ran on 3
  OSes and passed only because test tolerances were loose enough to hide
  the divergence. Replaced with a hand-rolled Box-Muller transform built
  only from the generator's raw (fully-specified) output and portable
  arithmetic (`cpp/include/bscpp/portable_normal.hpp`); verified correct
  (mean/var/skew/kurtosis match a standard normal over 5M draws) and the
  full test suite (all statistical/convergence tests) passes unchanged.
- **`bscpp.risk.Position.greeks` computed share-equivalent delta/gamma
  but called it "dollar Greeks."** 100 shares of raw delta means very
  different risk on a $20 stock than a $2000 one, so a cross-underlying
  limit checked in share-equivalent units -- the entire reason this
  module exists over `StrategyPricer`'s per-share convention -- wasn't
  actually comparable across names. Now computes true dollar delta
  (`delta * quantity * spot`) and dollar gamma (`gamma * quantity *
  spot^2 / 100`, the standard "$ change in dollar delta per 1% move"
  convention); vega/theta/rho were already correctly dollarized by
  quantity alone and are unchanged. `RiskLimits` values are now
  interpreted in dollar terms, not share counts --
  `portfolio_risk_demo.py`'s example limits updated accordingly.
- **Vol-clock inconsistency in the validation study**:
  `annualized_realized_vol` scaled trading-day returns by sqrt(365) while
  the backtester accrues calendar-day time -- inflating vol ~20% vs. the
  usual convention. Realized variance is now computed per unit of elapsed
  calendar time, matching the backtester's clock exactly.
- The validation study's headline inference replaced: i.i.d. Pearson
  p-value (meaningless for overlapping windows on co-moving tickers)
  removed in favor of a block-bootstrap CI with effective sample size
  reported alongside nominal N.

## [0.2.0] - 2026-08-12

Foundation release ("M0"): four correctness bugs found in an external
adversarial review are fixed with regression tests, and the project gains a
license, CI, and versioning. Where a fix invalidates a previously published
claim, the claim is corrected in the README rather than papered over.

### Fixed

- **Antithetic Monte Carlo standard error was overstated by ~32%.**
  `MonteCarloPricer` pooled all 2N antithetic samples into the i.i.d.
  variance formula, but antithetic pairs are negatively correlated by
  construction. The estimator now computes variance over antithetic PAIR
  MEANS (N i.i.d. observations). Regression test
  (`test_antithetic_std_error_is_calibrated`) verifies the reported
  std_error against the realized dispersion of the estimator across 120
  independent seeds. Every downstream "within k std errors" test tolerance
  is now measured against a calibrated yardstick.
- **`svi_gatheral_jacquier_check` returned false negatives on slices with
  negative total variance.** The g(k) >= 0 butterfly criterion is only
  meaningful where w(k) > 0; a slice with negative implied variance (an
  outright arbitrage) previously PASSED the check. Both arbitrage checks
  now enforce the precondition and reject such slices with
  `reason="negative_total_variance"`. Regression test added.
- **Heston pricer could return slightly negative prices deep OTM at short
  maturity** (quadrature noise exceeding the tiny true price; observed
  -2.3e-7). Prices are now clamped at the no-arbitrage floor of zero on
  both the call and the parity-derived put leg. Regression test added.
- **`PolygonProvider` silently truncated large option chains.** The v3
  snapshot and reference endpoints paginate via `next_url`; the provider
  fetched exactly one page (limit=250). All paginated endpoints now follow
  cursors to completion, and requests retry with exponential backoff on
  429/5xx. Regression test with a mocked 3-page session added.

### Changed

- **`StripPricer` no longer invents implied vols.** Rows whose IV solve
  fails (or that lack a usable quote) previously got a silent 0.20
  placeholder priced as if it were data. Every row now carries an
  `iv_source` column (`"quoted"` / `"solved"` / `"fallback"`), and
  fallback rows have NaN `model_iv` and NaN pricing outputs so no
  downstream consumer (SVI fitting, Heston calibration, error statistics)
  can mistake them for data.
- Test suite split into fast and `slow` markers (`pytest -m "not slow"`
  runs in ~15s; the full suite includes heavy Monte Carlo convergence and
  multi-start calibration checks).

### Added

- `LICENSE` (Apache-2.0). Previously the repository had no license, which
  legally barred any use.
- GitHub Actions CI: 3 OS x 2 Python versions on the fast suite, plus a
  full-suite job on Linux.
- `__version__` attribute sourced from package metadata; version bumped to
  0.2.0.
- This changelog.

## [0.1.0]

Initial development history (7 commits): C++ Black-Scholes/MC core, LSM
American pricing, Heston (little-trap CF + MC cross-check), SVI fitting
with two arbitrage checks, delta-hedging backtester with transaction costs
and P&L attribution, portfolio risk aggregation, Polygon/Massive data
provider, multi-ticker real-data validation study.
