# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased] - M1: hedging laboratory core

The three capabilities that make the hedging backtester a research
instrument rather than a demo: optimal rebalancing policies, vega-aware
attribution, and publication-grade statistics. 18 new tests (76 total).

### Research

- **`examples/hedging_policy_frontier_study.py`**: tests whether
  Whalley-Wilmott's (1997) asymptotically-optimal hedging band is actually
  cost-risk-minimizing on real market data (5 tickers, 50 rolling
  out-of-sample windows), by sweeping the band width via the exact
  `band(risk_aversion=lam0/c^3) = c*band(lam0)` identity (regression-tested
  in `test_policies.py`) and scoring against the same mean-variance
  objective the theory optimizes. Finding: 2 of 3 tested risk-aversion
  regimes were cost-dominated (objective still improving at the grid edge
  even after a 4x wider search -- a methodological artifact, reported as
  such rather than as a result). The one well-posed regime shows a real,
  small (+7%) gap: the empirically-optimal band is ~2x wider than theory,
  statistically distinguishable from zero via stationary block bootstrap.
  See the README's "Research finding" section.

### Added

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
