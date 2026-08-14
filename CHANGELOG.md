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
