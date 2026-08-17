# Design decisions

A running log of non-obvious calls made in this project, why, and what was
tried and rejected along the way. Not a changelog (that's `CHANGELOG.md`,
which documents *what shipped*) -- this is *why*, including the dead ends.

**On process**: this project was built with heavy AI-assisted iteration
(Claude), not written line-by-line by hand over months -- the commit
history's compressed dates are honest about that, not something to
obscure. Where it mattered, the boundary is architecture, verification,
and judgment calls versus scaffolding: every entry below is a decision a
human made and checked, not a claim the AI made it correctly on the first
try. Several entries are explicitly failed-first-attempt-then-fixed,
kept in because that's the real story, not because it's flattering.

---

## 2026-08-17: Split-sample selection, not reselect-inside-bootstrap, for post-selection inference

**Decision**: the hedging frontier study's band-multiplier CI selects `c*`
on the first half of calendar periods and tests it, unselected, on the
held-out second half.

**Why**: picking the best of 11 candidate band widths and then bootstrap-
CI'ing best-vs-baseline on the *same* data that did the picking is a
textbook winner's-curse setup -- anti-conservative by construction.

**What was tried first and rejected**: the obvious "cheap fix" --
reselecting the argmin inside every bootstrap resample -- was implemented
and checked with a Monte Carlo coverage simulation (generate many
known-null synthetic samples, measure the nominal-95% CI's actual false-
positive rate). It gave a **20% false-positive rate**, worse than doing
nothing (12.5%), against a 5% target. Root cause: percentile bootstrap CIs
aren't generally valid for argmax/argmin-based statistics. Split-sample
selection passed the identical simulation cleanly (3.3% vs 8% no-split
baseline, at real-study scale). See `bscpp.backtest.frontier
._split_sample_bootstrap` and `tests/test_frontier.py` for the coverage
simulation kept as a permanent regression test, not just a one-off check.

## 2026-08-17: Cluster (not row-level) bootstrap for the panel's cross-sectional dependence

**Decision**: the frontier study's bootstrap resamples whole calendar
periods, not individual (ticker, window) rows.

**Why**: 20 tickers sharing the same calendar dates co-move (SPY literally
contains several of the others) -- treating same-date rows as independent
understates uncertainty. Verified against the true Monte-Carlo sampling
distribution on synthetic data with induced same-period correlation: the
cluster bootstrap lands close to the true sampling std and is >1.5x wider
(more honest) than the naive row-level version on that data.

## 2026-08-17: Price a window once per policy sweep, not once per policy

**Decision**: `run_policy_grid` calls `HedgingBacktester.price_path()`
once per window and reuses the priced table across all
`risk_aversion x band_multiplier` cells via the new `_run_from_pricing()`,
instead of repricing identical data through `run()` once per cell.

**Why**: pricing (the C++ crossing) never depends on the policy under
test -- only the cash/shares/transaction-cost simulation does. Measured
1.67x (8.03s -> 4.81s) on a 13,860-cell synthetic grid at the real study's
scale; verified byte-identical output to the unrefactored path via a
direct before/after run plus the full pre-existing `test_hedging.py`
suite.

## 2026-08-17: Test the discretization confound directly, not just by elimination

**Decision**: added a rebalance-frequency sweep to
`gbm_control_experiment.py` -- the same true-vol GBM paths rehedged on
2/3/5/10-business-day monitoring grids instead of daily, everything else
held fixed.

**Why**: the existing GBM control arms argued daily-rebalancing
discretization explains most of the real-data band-widening by
*isolating* it and showing a match, but never showed that moving the
discretization knob actually moves `c*`. This does.

**What the result actually showed (not what was expected)**: `c*` was
predicted to widen further as monitoring gets coarser (more slippage a
band can't avoid). It did the opposite -- shrank monotonically toward and
past theory's `1x` as the check-in interval widened. On reflection this
is the right sign: band width and monitoring frequency are substitute
levers on the same thing (trade frequency), so once monitoring itself
caps how often you can trade, a wide band stops buying extra protection.
Kept the wrong prediction in the README write-up rather than only the
corrected explanation -- the reasoning error is part of the result.
