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
- **Second methodology correction: post-selection inference and
  cross-ticker dependence in the frontier study's bootstrap CI.** An
  external review of the finding above identified two real statistical
  gaps in how its significance was assessed (not in the point estimate
  itself, which was already scale-invariant and pooled correctly):
  - **Post-selection inference**: `c*` was chosen as the argmin of J(c)
    over 7 (now 11) candidates, and the bootstrap CI for "c* vs c=1" was
    computed on the SAME sample that did the selecting -- picking the
    best of several noisy candidates and testing best-vs-baseline on the
    same data is a textbook winner's-curse setup, yielding an
    anti-conservatively narrow CI.
  - **Cross-ticker contemporaneous dependence**: the study pools ~20
    co-moving tickers' rolling windows; a block bootstrap over rows
    ordered `(ticker, date)` -- even with blocks long enough to preserve
    WITHIN-ticker serial dependence -- is blind to windows from DIFFERENT
    tickers sharing the same `window_start` being correlated (same market
    vol regime; a broad-market name like SPY moves much of the basket).
    `n_effective≈178` in the original writeup was optimistic for exactly
    this reason.
  - **The fix that was tried first didn't work, and this was caught
    empirically before shipping it, not after**: the commonly-suggested
    "cheap fix" for post-selection inference -- re-running the argmin
    selection INSIDE every bootstrap resample -- was implemented first
    (`bscpp.backtest.frontier._selection_adjusted_cluster_bootstrap`,
    since removed). Before trusting it, it was checked with a Monte Carlo
    coverage simulation: many fresh synthetic samples where every
    candidate has the IDENTICAL true objective (no real effect), checking
    how often a nominal-95% CI wrongly excludes zero -- a well-calibrated
    procedure should do this ~5% of the time. The "reselect inside the
    bootstrap" version FAILED this test, at one tested scale giving a
    WORSE false-positive rate (20%) than doing nothing at all (12.5%,
    against the 5% target). This is a known subtlety in the statistics
    literature: percentile bootstrap CIs are not generally valid for
    argmax/argmin-based statistics -- the resampled statistic's own
    distribution is itself selection-biased, the same problem one level
    up. **Split-sample selection** (`c*` chosen on the chronologically
    FIRST HALF of calendar periods, tested without reselecting on the
    held-out second half) passed the identical coverage simulation
    cleanly at realistic scale (42 periods x 20 tickers: 3.3%
    false-positive rate vs. a 5% target, vs. 8% for no-split-at-all) and
    is what's actually implemented
    (`bscpp.backtest.frontier._split_sample_bootstrap`). Both the passing
    and failing simulations are preserved in `test_frontier.py`
    (previously this module had NO dedicated tests despite driving the
    project's headline empirical claim), specifically so this doesn't
    regress silently back to the invalid version.
  - **Cross-ticker dependence fixed together, not separately**: the
    held-out half's bootstrap resamples whole CALENDAR PERIODS as one
    unit (`bscpp.stats.cluster_bootstrap_indices`, new -- every ticker's
    rows for a resampled period move together), not individual rows.
    Verified directly (`test_frontier.py`): on synthetic panel data with
    an induced common per-period shock across "tickers" (the same-date
    correlation real market data has), the cluster bootstrap's implied
    sampling uncertainty lands close to the TRUE Monte-Carlo-measured
    sampling distribution, while a naive row-level bootstrap on the same
    data understates it by more than 1.5x.
  - **A real, unrelated bug found while re-running the GBM control arms
    under the new methodology**: `gbm_control_experiment.py`'s window
    generator offset each simulated path's calendar anchor by `path_idx`
    alone, not also by vol level -- so all 5 vol levels reused the SAME
    10 calendar anchors, collapsing 50 nominally-independent windows onto
    just 10 distinct `window_start` values. Since these paths share no
    randomness across vol levels (fresh `rng.normal()` draws per
    iteration) and have no real reason to be calendar-coincident, this
    only made an already-independent control arm's new cluster-bootstrap
    CI needlessly conservative -- caught because the fixed, more careful
    methodology surfaced a suspiciously thin effective sample size where
    the data should have supported a much larger one, not because
    anything crashed. Fixed by giving each vol level its own large
    calendar offset (`gbm_control_experiment.py`); confirmed 50 windows
    now produce 50 distinct periods.
  - **Also fixed in the same pass** (small, but caught by the same
    review): `run_policy_grid`'s `except Exception: continue` around each
    backtester call, and its premium-floor skip, dropped cells with no
    counting -- if failures correlated with particular `(c, lam0)` cells
    or tickers, those cells would be differentially and invisibly
    thinned, a direct violation of this project's own "no silent caps"
    scope commitment. Now counts drops by reason, attaches them to the
    returned grid's `.attrs`, and `warnings.warn()`s if drops exceed 1%
    of the expected grid (0% on both the real and GBM studies' latest
    runs). The exception clause was also narrowed from bare `Exception`
    to `ValueError` specifically (the only exception type the pricing
    core actually raises, via `std::invalid_argument` -> pybind11's
    default translation) -- a real bug (e.g. an `AttributeError` from a
    logic error) now surfaces instead of being silently swallowed.
    Synthetic option strikes were also switched from `round(spot0/5)*5`
    to true ATM (`spot0` exactly) -- these are synthetic options with no
    listed-strike constraint to respect, so rounding only injected up to
    ~4% moneyness noise on cheaper names, heterogeneous across tickers,
    for no offsetting benefit.
  - **The grid was also refined** from 7 power-of-2-only points (0.25-16)
    to 11 (adding 1.5/3/6/11) -- the old grid could only report "c* is
    somewhere in (4, 16)" at its resolution; the real study's `c*=8`
    read as more precise than the grid actually supported. Both studies
    now resolve to an exact grid point (`6` and `4` respectively).
  - **The GBM control arms were widened** from 10 to 100 paths per vol
    level (50 -> 500 windows per arm) -- at 10 paths, the control arm
    meant to precisely measure vol-estimation error's contribution had
    roughly 1/8th the real study's statistical power, underpowered for
    the "rules out vol-estimation error" language the original writeup
    used. The wider arms complete a full 11-point-grid x 3-regime sweep
    in well under a minute (measured, printed by the script itself), so
    there was no reason to stay thin.
  - **Net effect on the published numbers**: real data `c*=8x`/`+35.7%`
    -> `c*=6x`/`+32.9%` at the moderate risk-aversion regime;
    `c*=4x`/`+20.7%` -> `c*=4x`/`+19.7%` at the high regime (unchanged
    point, tighter validation). GBM-true-vol `c*=8x`/`+36.5%` ->
    `c*=6x`/`+30.0%` (now matching real data exactly) and `c*=4x`/`+20.3%`
    -> `c*=3x`/`+17.7%` (one grid step short of real data, a small,
    honestly-reported residual rather than the earlier "sufficient on its
    own, full stop" framing). The finding survived a materially more
    rigorous test of itself -- that's part of the result, not a
    disclaimer on it.
  - `bscpp.stats._stationary_bootstrap_indices` (the primitive
    `cluster_bootstrap_indices` and every bootstrap in this project builds
    on) was also rewritten from a per-element Python loop to a fully
    vectorized construction (Bernoulli block-start indicators + one
    vectorized fresh-start draw per block, no explicit loop) while this
    module was open for the cluster-bootstrap addition -- measured 16.5x
    faster on a 420-observation, 2000-resample CI (303ms -> 18ms),
    verified to preserve the exact geometric block-length distribution
    the original per-element version had (`test_frontier.py`), not just
    assumed equivalent from the derivation.
- **Ruled out a remaining confound in the GBM control arms, rather than
  leaving it as a disclosed caveat**: `gbm_true_vol` diffuses its
  simulated price path's variance over each date's real calendar-day gap
  (3 calendar days across a weekend, matching `HedgingBacktester`'s own
  ACT/365 clock) -- but real markets realize roughly one trading day of
  variance over a weekend, not three, so the arm wasn't quite "GBM under
  the real study's exact conditions." Added a third arm
  (`gbm_true_vol_trading_clock`, `examples/gbm_control_experiment.py`)
  using a uniform `dt=1/252` per business-day step regardless of the real
  calendar gap -- removing the artifact from the price diffusion while
  leaving the backtester's own financing/theta accounting unchanged (so
  the test isolates the diffusion-only effect, not a second, different
  question). Result: `c*` is IDENTICAL to the calendar-clock arm at both
  well-posed regimes (`6x` and `3x`) -- the weekend-variance artifact
  is not contributing to `gbm_true_vol`'s close match to real data.
- **`run_policy_grid` repriced every window from scratch once per policy
  cell**, even though pricing (the C++ crossing) never depends on the
  policy being tested -- only the (risk_aversion, band_multiplier) sweep
  does. `HedgingBacktester` gained `price_path()`, which batches an
  entire window's price/Greeks computation into one
  `bs_price_with_greeks_batch_arrays` call instead of one Python-to-C++
  crossing per day, and `run()` was refactored into `price_path()` (pricing)
  + the new `_run_from_pricing()` (the inherently sequential cash/shares/
  transaction-cost policy simulation, which never touches the pricer).
  `run_policy_grid` now calls `price_path()` ONCE per window and reuses
  the table across all `len(risk_aversions) * len(multipliers)` policy
  cells via `_run_from_pricing()`, instead of re-deriving identical
  pricing dozens of times per window.
  - Verified equivalent, not just assumed from the refactor's structure:
    `price_path()`'s output matches an independently re-implemented
    day-by-day pricing loop to 1e-9 across 7 scenarios (call/put, OTM,
    dividend, time-varying hedge_vol, deep ITM/OTM at expiry --
    `test_price_path_matches_independent_per_day_pricing`,
    `tests/test_hedging.py`), and `_run_from_pricing()`'s output is
    byte-identical to calling `run()` directly for the same inputs
    (`test_run_from_pricing_matches_run_directly`). The full pre-existing
    `test_hedging.py` suite (11 tests, covering P&L attribution, Carr-
    Madan closed-form agreement, transaction costs, dividend handling)
    passes unchanged against the refactored `run()`.
  - Measured on a synthetic 420-window x 3-risk-aversion x 11-multiplier
    grid (13,860 cells, matching the real study's scale): 8.03s before
    this refactor, 4.81s after (1.67x) -- and the resulting `c*`/`gap_pct`
    at every risk-aversion regime are identical before and after, cross-
    checked via a direct before/after run on the same synthetic grid
    rather than assumed from the unit tests alone.

### Added

- **NumPy-native batch API, selective GIL release, OpenMP path loops** --
  the "systems, not just C++" item.
  - `bs_price_with_greeks_batch_arrays`/`bs_implied_vol_batch_arrays`:
    `py::array_t` struct-of-arrays in and out, zero per-contract Python
    object construction (the list-of-`MarketInputs` batch functions still
    construct N real objects before the call starts -- the exact overhead
    batching was meant to remove, just moved one step earlier). Measured
    7.1x/16.6x/20.5x faster than the list-based path at n=13/100/500
    contracts (the gap grows with n because it's dominated by avoiding N
    object constructions, not the C++ loop itself: isolating just the
    call, the gap is only ~1.2x). Wired into `StripPricer` and
    `calibrate_heston`'s IV-space residual callback (the actual profiled
    hot path, ~300 calls/calibration) -- calibration wall-clock is
    unchanged, honestly reported: Heston pricing, not the BS IV step, is
    96.8% of that cost per the original profiling, so a large relative
    win in a small-fraction component doesn't move total calibration
    time. `StripPricer.price_strip` end-to-end: 3.1ms for a 50-row chain
    (no clean isolated before/after here -- other pipeline costs are
    unaffected by this change and weren't independently re-measured).
  - `py::call_guard<py::gil_scoped_release>()` on long C++ calls
    (`MonteCarloPricer`, `AmericanPricer`, `HestonMCPricer`,
    `heston_price`/`_batch`, `crr_price`/`crr_implied_vol`, the batch
    functions) -- applied selectively: skipped on sub-microsecond calls
    (`bs_price` etc.) where the release/reacquire's own fixed cost would
    dominate rather than pay for itself.
  - OpenMP parallelizes the path loops in all three MC pricers. Redesigned
    `generate_normals`/`simulate_paths`/`HestonMCPricer::price` so each
    output index/path seeks to its OWN disjoint Philox counter position
    (base + i, see `philox.hpp`) instead of drawing from one shared,
    sequentially-advancing generator -- output depends only on the index,
    never on thread count or scheduling, which is what makes the loop
    safe to parallelize at all. `AmericanPricer`'s two `Philox4x64`
    members became a `seed_` + per-stream block cursor (constructs a
    fresh local generator per path instead); cursors advance after each
    call so a reused pricer instance draws fresh, non-repeating paths
    (verified directly, not assumed). Precisely verified, not oversold:
    the underlying draws are exactly reproducible at any thread count
    (`test_openmp_determinism.py`), but the *aggregated* price/std_error
    can differ in the last few ULPs at scale, because OpenMP's
    `reduction(+:...)` sums per-thread partials in a thread-count-
    dependent order and floating-point addition isn't associative --
    standard, expected behavior for any parallelized numerical reduction,
    stated exactly rather than claimed as blanket "bit-identical."
    Measured speedup (Apple M4 Pro, 8 performance cores): European MC
    6.1x at 8 threads (46.0ms -> 7.6ms), Heston MC 7.4x (919.1ms ->
    124.7ms), LSM only 2.0x (594.4ms -> 294.8ms) -- Amdahl's law: only
    path generation is parallelized, not LSM's backward-induction
    regression step. `setup.py` detects OpenMP via an actual compile-
    and-link check (not a platform assumption) and degrades gracefully
    to a correct sequential build if unavailable -- `#pragma omp` is
    silently ignored by a compiler without it enabled. Verified working
    end-to-end on macOS (Apple Clang + Homebrew `libomp`, not bundled by
    default) in this environment; Linux/Windows use standard flags but
    aren't independently verified here -- CI (3 OSes) is the real
    cross-platform test. New, real limitation disclosed in the README's
    Scope section: releasing the GIL means concurrent calls on the SAME
    pricer instance from multiple Python threads would race on that
    instance's path-generation cursor -- one instance per thread is the
    safe pattern, no locking added on top.
- **`bscpp::Philox4x64`** (`philox.hpp`): counter-based RNG (Salmon,
  Moraes, Sanches, Pande 2011) replacing `std::mt19937_64` in every Monte
  Carlo pricer (`MonteCarloPricer`, `AmericanPricer`, `HestonMCPricer`).
  Upgrades the earlier reproducibility fix (`std::normal_distribution` ->
  hand-rolled Box-Muller on `std::mt19937_64`, see the "Fixed" section
  below) from "internally self-consistent" to "matches an independent,
  already-portable reference implementation bit-for-bit": raw output is
  cross-validated against `numpy.random.Philox` across 7 seeds including
  edge cases (`test_philox.py`), not just assumed correct because the
  constants looked right. Getting the constants/round-function/portable-
  multiply right on the first attempt without that reference would have
  been lucky, not verified -- the raw output was compared directly, not
  inferred from statistical shape.
  - Portable 64x64->128-bit multiply deliberately avoids `__uint128_t` (a
    GCC/Clang extension MSVC doesn't support -- this project's CI runs on
    `windows-latest` too): the schoolbook 32-bit-limb expansion is
    standard C++, verified bit-exact against Python's arbitrary-precision
    integers across random and edge-case inputs before being trusted in
    the actual generator.
  - `AmericanPricer`'s calibration/pricing path sets now use Philox's
    `stream` parameter (`rng_(seed, 0)`, `rng_calibration_(seed, 1)`)
    instead of an arbitrary seed offset (`seed + 1768237423ULL`) --
    provably non-overlapping streams from the same seed, not "probably
    far enough apart in a 2^19937-period sequential state."
  - Counter-based generation (`seek(counter)` reaches any stream position
    directly, no replay needed) is what makes per-thread streams for
    parallelized path generation possible with zero coordination --
    confirmed disjoint via `seek()` to different counters
    (`test_philox_seek_to_distinct_counters_gives_disjoint_streams`), sets
    up the OpenMP work below.
  - Measured cost, not assumed free: ~1.45x slower per normal draw than
    the previous `mt19937_64`-backed version (25.8ns vs 17.8ns, Apple M4
    Pro) -- Box-Muller's transcendental functions dominate enough that
    this is well under the ~3.9x gap in raw generator throughput alone
    (5.83ns vs 1.48ns/draw). Full test suite (122 tests) passes unchanged
    -- Philox produces equally-valid uniform variates, so every
    statistical/convergence test that was robust to *which* RNG algorithm
    generated the paths remains robust here too.
- **`benchmarks/`** (new, not part of the default test suite -- see
  `benchmarks/README.md`): external speed comparison against QuantLib
  1.43 and vollib 1.0.11 on identical inputs, correctness-asserted before
  anything is timed. Replaces the self-referential "3.1x faster than my
  own previous version" framing with an absolute reference. Results
  (Apple M4 Pro, single-threaded, `pytest-benchmark`): BS price 21x
  faster, BS implied vol 39x faster, American (CRR, 500 steps) 6x faster
  -- and Heston price **14x slower**, published because it's true, with
  the honest reason (adaptive Simpson quadrature vs. QuantLib's
  fixed-node table, a tradeoff already documented in `heston.hpp` before
  this benchmark existed to quantify it) rather than omitted. Caught and
  fixed a real methodology trap before trusting any of these numbers:
  QuantLib's `NPV()` is cached by its lazy-evaluation object model until
  a watched quote changes (~6x faster for a cached call vs. a genuinely
  recomputed one, confirmed by direct measurement) -- every QuantLib
  benchmark now perturbs the spot quote before each call to force real
  recomputation (`benchmarks/conftest.py`), matching what a calibration
  loop repricing against a moving quote would trigger naturally. Full
  table and hardware disclosure in the README's "External benchmarks"
  section.
- **README "Scope" section**: elevated the single "Out of scope" bullet
  (previously buried at the end of "What's implemented") into its own
  top-level section, and expanded it to cover everything surfaced by this
  round of work -- no curve-bootstrapping engine, no separate repo/
  financing curve, no full trading-day calendar, no exact American
  Greeks, no Andersen-Broadie duality bounds on LSM -- alongside the
  original data/execution/margin scope statement. Explicit scoping turns
  an omission into a stated boundary instead of leaving a reviewer to
  guess whether it's an oversight.
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
- **`heston_price_cos`**: Fang & Oosterlee (2008) COS-method pricer,
  built specifically to close the ~13x single-price gap to QuantLib's
  `AnalyticHestonEngine` published in the "External benchmarks" section
  (the adaptive-quadrature `heston_price` is the right default for
  correctness-first use; `heston_price_batch` above already closed this
  gap for the batched-calibration case, not the cold single-price one).
  Shares `char_function` with `heston_price` -- its `j=2` branch is
  already the plain risk-neutral characteristic function of `ln(S_T)`,
  needing no separate re-derivation.
  - **Caught a real bug before trusting this, not after**: the first
    draft copied Fang & Oosterlee's published payoff-coefficient formula
    `K*(chi_k - psi_k)` verbatim, which assumes their paper's log-
    moneyness convention `x = ln(S_T/K)`. This implementation uses
    absolute log-price `x = ln(S_T)` (to match `char_function`), where
    `K` multiplies only the `psi` (constant) term, not `chi` -- the
    borrowed formula was silently wrong by construction. Caught by cross-
    checking against a from-scratch, independent COS reimplementation
    using the textbook Black-Scholes characteristic function (bypassing
    Heston entirely, to isolate a COS-assembly bug from a Heston-CF bug)
    before this method was ever exposed to Python: the first draft priced
    a $10.45 BS call at $6,315 -- not a subtle drift, the unmistakable
    kind of error a cross-check against a trusted reference catches
    immediately and a "does it compile and return a number" smoke test
    does not.
  - **A single fixed truncation range is not robust across this pricer's
    full parameter range**, found the same way: a fixed `[a,b]` domain
    (from the characteristic function's numerically-estimated first two
    cumulants) sized to survive `heston_price`'s own worst-case stress
    tests (1-day maturity, Feller-violating vol-of-vol) still missed by
    0.6% on that same case, and adding a numerically-estimated 4th
    cumulant (the standard Fang-Oosterlee refinement for fat tails) made
    it *worse* -- the finite-difference estimate is noisy enough at short
    maturities to widen the domain when it shouldn't. Replaced both with
    an adaptive scheme: widen the truncation range and term count
    *together*, iteration by iteration, stopping when two successive
    (unclamped -- see below) estimates agree, falling back to
    `heston_price` itself if they never do within a bounded number of
    iterations -- the same "self-terminating on measured error, not
    assumed at a fixed truncation" philosophy `heston_price`'s own
    adaptive quadrature already uses.
  - **The convergence check itself had a false-positive trap**: a random
    300-case stress sweep (maturities 1 day - 3 years, kappa/theta/xi/
    rho/v0 spanning well-behaved to badly Feller-violating) found a
    long-maturity, badly-Feller-violating case where two successive
    iterations both landed on the no-arbitrage floor of exactly `0.0` by
    coincidence (the true price was ~$0.80) -- comparing the *clamped*
    price made that look like convergence. Fixed by comparing the raw,
    unclamped sum across iterations instead, only applying the floor to
    the final returned value.
  - **Verified, not assumed**: <0.1% relative error against `heston_price`
    across that same 300-case random sweep (0 mismatches, vs. 3 before the
    fixes above) plus the existing hand-picked stress regimes
    (`test_heston.py`); measured **1.19x** slower than QuantLib's
    `AnalyticHestonEngine` (25.9us vs 21.7us, min-timed) on the
    single-price case the 13x-slower benchmark measured, vs.
    `heston_price`'s 13.5x (`benchmarks/test_heston_benchmark.py`).
- **`heston_price_jacobian`/`heston_price_jacobian_batch`** (new,
  `cpp/include/bscpp/dual.hpp`): price() plus its exact partial
  derivatives w.r.t. all 5 Heston parameters via forward-mode automatic
  differentiation, replacing `calibrate_heston`'s reliance on scipy's
  default finite-difference Jacobian (`use_analytic_jacobian=True` now
  the default; set `False` to recover the previous behavior).
  - **Not literal complex-step, and this was worked out before writing any
    code, not discovered by a wrong first attempt**: `char_function`
    already uses the imaginary unit `i` internally (the Fourier phase
    factor), and `probability()` extracts `Re[...]` from the integrand at
    every quadrature node before integrating. `Re()` is not a holomorphic
    operation, so perturbing a parameter with the SAME `i`
    (`kappa -> kappa + i*h`) would corrupt exactly the real/imaginary
    split `Re[]` depends on -- silently, not with an error. The fix is a
    SECOND, independent differentiation unit (`ComplexDual5`/`RealDual5`
    in `dual.hpp`, tracking each parameter's derivative in its own array
    slot instead of reusing `i`), mathematically equivalent to
    "multicomplex-step" (Lantoine, Russell & Dargent 2012): `Re()`/`Im()`
    commute with it exactly, because they're real-linear projections, not
    because of any special-casing. Same zero-cancellation,
    no-tuning-parameter guarantee true complex-step gives for ordinary
    real functions, generalized to survive the `Re()` step this pricer's
    math requires.
  - **One formula, two instantiations, not a hand-duplicated second
    copy**: `char_function` became `char_function_impl<T>`, a template
    parameterized on the scalar type used for kappa/theta/xi/rho/v0
    (`T=double` reproduces today's exact arithmetic byte-for-byte --
    confirmed via the full existing test suite passing unchanged after the
    refactor, not assumed from it being "mechanical"; `T=ComplexDual5`
    computes the Jacobian). The adaptive Simpson quadrature
    (`adaptive_simpson`/`integrate_to_infinity`) was templated the same
    way, with an `abs_value()` helper so the convergence/refinement check
    always drives off the integral's VALUE, not its derivatives. Same
    reasoning as extracting `brent.hpp`: a second, separately-written copy
    of the same formula is a second place for a transcription error to
    hide, invisible until it silently disagrees with the first on some
    untested input.
  - **First cut measured 3.6x SLOWER than finite differences, caught before
    calling this "done," not after**: calling the per-strike
    `heston_price_jacobian` in a Python loop (mirroring the OBVIOUS design)
    took 97ms vs finite-difference's 27ms on a 13-strike calibration --
    each call redid its own adaptive quadrature from scratch, giving up
    exactly the cross-strike characteristic-function sharing
    `heston_price_batch` already exists for, while ALSO paying
    `ComplexDual5`'s ~5x wider per-node arithmetic. Fixed by adding
    `heston_price_jacobian_batch`, batching the Jacobian the same way
    `heston_price_batch` batches price() -- fixed (not adaptive) shared
    quadrature grid, characteristic-function-and-derivatives evaluated
    once per node and reused across every strike. Net: **~20% faster**
    at typical calibration resolution (25.9ms vs 32.4ms, median of 30
    runs) and **~41% faster** at the higher resolution short-dated
    calibrations fall back to (36.9ms vs 63.1ms) -- more modest than a
    naive "6 fewer residual evaluations per iteration" estimate would
    suggest (each analytic-Jacobian call still costs more per call than
    one `heston_price_batch` call), but real and measured, not assumed.
  - **Verified, not assumed**: analytic partials match central finite
    differences on `heston_price` across the same stress regimes
    `heston_price_cos` was validated against; the batched Jacobian matches
    the per-strike one; the IV-space Jacobian (converted via the implicit
    function theorem, `d(iv)/d(param) = dPrice/dparam / vega(iv)`) matches
    a plain central difference on `_heston_implied_vols`; and
    `calibrate_heston` with `use_analytic_jacobian=True` vs. `False`
    converges to matching fitted parameters and RMSE (within 1e-3) on the
    same data (`test_heston.py`).
- **`HestonMCPricer.price_qe`** (new): Andersen (2008) "QE" (Quadratic-
  Exponential) scheme for the Heston Monte Carlo cross-check, added as a
  SEPARATE method alongside `price` (full-truncation Euler), not a
  replacement -- matching this project's established pattern
  (`HestonPricer::price` vs. `price_cos`) of cross-checked alternatives.
  - **The problem it fixes was already disclosed, not discovered here**:
    `price`'s docs already documented that full-truncation Euler's
    discretization bias grows large when the Feller condition is badly
    violated (300 steps disagrees with the analytic price by ~40 standard
    errors at xi=3.0 against `2*kappa*theta=0.16`; needs ~3000 steps to
    converge to within 1). QE instead samples the CIR variance step from a
    distribution moment-matched to its true (non-central chi-squared)
    conditional law -- squared-Gaussian or exponential-tailed, chosen per
    step by the local variance-to-mean ratio, with a paper-recommended
    psi_c=1.5 switching threshold -- so v(t+dt) is exactly non-negative by
    construction instead of floored, fixing the bias at its source.
  - **Measured, not assumed**: in the SAME Feller-violating regime, QE
    reaches Euler's 3000-step accuracy (within 5 std errors of the
    analytic price) at just 20 steps; the matched-accuracy comparison
    (3000 Euler steps vs. 20 QE steps, same 150k paths) is **~150x
    faster** (2907ms vs. 19ms, single-threaded).
  - **Deliberately skips Andersen's martingale-correction variant** of the
    log-price drift term K0 (the paper's fix for a residual E[S_T] bias at
    large step sizes) -- not an oversight: verified directly that the bias
    is negligible at the step counts this method is actually used at, by
    pricing a call struck at 0 (payoff = S_T exactly, isolating the
    discounted forward from any strike-dependent effect) and confirming it
    matches the theoretical forward within Monte Carlo noise
    (`test_heston.py`). A disclosed simplification, not an unexamined one
    -- the correction exists in the paper specifically for regimes this
    implementation hasn't been asked to handle yet.
  - **Verified, not assumed**: matches the analytic price across the same
    stress regimes `heston_price`/`heston_price_cos` were both validated
    against (short maturity, dividends, various strikes/types, Feller-
    violating vol-of-vol), all at the same low (20) step count
    (`test_heston.py`).
- **`fit_svi_slice_quasi_explicit`** (new, `bscpp.backtest.vol_surface`):
  Zeliade Systems' (2009) "quasi-explicit" SVI calibration, added
  alongside the existing `fit_svi_slice` (plain 5-parameter nonlinear
  least squares), not a replacement.
  - **The problem it addresses is real, not hypothetical**: substituting
    `y=(k-m)/sigma` turns SVI's `w(k) = a + b(rho(k-m) +
    sqrt((k-m)^2+sigma^2))` into `w(k) = c1 + c2*y + c3*sqrt(y^2+1)`,
    LINEAR in `(c1,c2,c3) = (a, b*rho*sigma, b*sigma)` for any fixed
    `(m,sigma)` -- reducing the search from 5D nonlinear (which, like any,
    can land in a bad local optimum depending on where it starts) to 2D
    nonlinear with the other three parameters solved by an exact,
    convex, initial-guess-independent linear system at every candidate.
    Gave `fit_svi_slice` an `initial_guess` override (previously
    hardcoded) specifically to demonstrate this, not just assert it: on a
    short-dated, strongly-skewed, noisy smile, two different bad initial
    guesses degrade `fit_svi_slice` to RMSE ~0.19-0.22 (an effectively
    failed fit) while `fit_svi_slice_quasi_explicit` -- no initial guess
    needed -- fits the same data to RMSE ~0.0015 regardless.
  - **"Quasi", not fully, explicit**: the closed-form `(c1,c2,c3)` answer
    isn't always a valid SVI slice (needs `c3>=0`, `|c2|<=c3` i.e.
    `|rho|<=1`, and non-negative total variance at the minimum -- the
    same condition `svi_min_total_variance` already checks, just in
    `(c1,c2,c3)` form). Falls back to a constrained convex optimization
    when it isn't, still initial-guess-insensitive in the sense that
    matters (feasible region and objective are both convex).
  - **Grid density measured, not assumed, to not matter for the final
    answer**: the outer `(m,sigma)` search grid feeds a local (Nelder-
    Mead) refine step, and RMSE came back IDENTICAL from a 5x5 grid
    through 21x21 on every scenario tested -- the refine step does
    essentially all the real work, so a dense grid mostly just adds cost
    (a 21x21 grid measured up to ~30x slower than 5x5 for identical
    RMSE). Default shipped at 9x9, a margin above the measured-sufficient
    floor for real data less well-behaved than what was tested, not a
    value found necessary.
  - **`vega_weighted=True` (default)**: weights each strike's (variance-
    space) residual by vega^2 -- vega being d(BS price)/d(vol), the
    actual price sensitivity to an IV error there -- via the existing
    `bs_price_with_greeks_batch_arrays`. An unweighted fit treats a
    1-vol-point error at a near-zero-vega deep OTM strike the same as at
    the vega-heavy ATM strike, backwards for anything pricing off the
    fitted smile afterward. Verified to actually change the fit (not
    silently a no-op) on noisy data where unweighted and vega-weighted
    optima genuinely differ -- a noiseless synthetic smile that exactly
    matches the SVI functional form has no such tension (every point
    agrees regardless of weighting), so this needed noise to test for
    real, not just a clean recovery check.
  - **Not a speed win, stated as such**: ~7x slower than `fit_svi_slice`
    on a realistic 15-strike chain (10.3ms vs 1.5ms) -- the value is
    initial-guess robustness and vega weighting, not raw throughput, and
    the README doesn't claim otherwise.
  - **Verified, not assumed**: the `(c1,c2,c3)` reparametrization
    reproduces a known SVI slice's total variance to ~machine precision
    (`test_svi_conditional_linear_fit_matches_the_svi_formula_when_
    feasible`); the full fit recovers a known smile to the same bar
    `fit_svi_slice` is held to; and a well-behaved fit passes
    `svi_gatheral_jacquier_check` (`test_vol_surface.py`).
- **Type stubs, `py.typed`, and wheel-building infrastructure** (new).
  - `python/bscpp/py.typed` (PEP 561) plus hand-verified
    `python/bscpp/_core.pyi` for the compiled pybind11 extension (no
    Python source of its own for a type checker to read otherwise).
    Generated via `pybind11-stubgen`, then checked against `mypy --strict`
    before being trusted, not shipped as generated -- which caught a real
    issue: the generated stub for `MarketInputs.__init__` was invalid
    Python syntax (a required parameter following a defaulted one,
    mirroring the C++ binding's own declared argument order in
    `bindings.cpp` -- valid for pybind11's runtime keyword matching, not
    for a Python `def`). Fixed by marking the trailing parameters
    keyword-only in the stub, matching how this constructor is actually
    called everywhere in this codebase already (`bscpp.make_inputs`,
    always by keyword past `rate`). `pyproject.toml` gained
    `[tool.setuptools.package-data]` so both files actually ship in built
    wheels/sdists -- confirmed by building both and inspecting their
    contents, not assumed from the config alone.
  - `.github/workflows/wheels.yml` (new): builds wheels via `cibuildwheel`
    (Linux/macOS/Windows, CPython 3.10-3.12) on version tags or manual
    dispatch, uploaded as build artifacts only -- deliberately not wired
    to publish anywhere, a separate decision this workflow doesn't make.
    Running `cibuildwheel` locally before trusting the config (rather
    than assuming the YAML was correct) found a real, concrete failure:
    a Homebrew-built `libomp` on the build host targeted macOS 15.0,
    newer than the arm64 wheel's declared 11.0 minimum, so `delocate`
    (which bundles the dylib into the wheel) correctly refused to ship
    it. Rather than chase a Homebrew-version-dependent fix, added
    `BSCPP_SKIP_OPENMP=1` (`setup.py`) to make wheel builds independent
    of whatever happens to be on the build host -- distributed macOS
    wheels build single-threaded but portable; `pip install` from source
    with `brew install libomp` is unaffected. Verified end-to-end after
    the fix: a wheel built this way installs into a fresh venv and passes
    the full non-slow test suite (136 tests) against the installed
    package itself, not just the dev environment.

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
