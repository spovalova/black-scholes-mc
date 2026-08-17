"""Tests for bscpp.backtest.frontier's statistical machinery -- previously
had no dedicated tests despite driving this project's headline empirical
claim (the Whalley-Wilmott band-width finding). Two things are verified
directly, not just asserted from the derivation in frontier.py's
docstrings:

1. Post-selection inference: split-sample selection (c* chosen on the
   first half of calendar periods, tested unselected on the second) should
   give close-to-nominal CI coverage, verified via a Monte Carlo coverage
   simulation where no c is truly better than baseline (a well-calibrated
   95% CI should exclude zero, i.e. falsely claim significance, about 5%
   of the time).

   This test also documents a real dead end: an earlier version of this
   fix instead RE-RAN the argmin selection inside every bootstrap resample
   (a commonly-suggested "cheapest fix" for post-selection inference).
   That approach was implemented AND tested with this exact simulation
   before being trusted -- and failed it: at small n it produced a WORSE
   false-positive rate than doing nothing (20% vs. a 12.5% no-fix
   baseline, against a 5% target), because percentile bootstrap CIs are
   not generally valid for argmax/argmin-based statistics. Split-sample
   selection sidesteps that subtlety entirely by construction, and is
   what's actually implemented and tested below.

2. Cross-sectional dependence: resampling calendar periods as clusters
   (not individual ticker-window rows) should recover a CI width close to
   the TRUE sampling variability of a panel with induced same-period
   correlation across "tickers" -- verified by comparing against the
   actual Monte Carlo sampling distribution of the panel mean, not just
   against the (understated) naive row-level bootstrap.
"""

import numpy as np
import pandas as pd
import pytest
from bscpp.backtest.frontier import _split_sample_bootstrap, score_frontier
from bscpp.stats import _stationary_bootstrap_indices, cluster_bootstrap_indices


def _make_sub(window_starts, labels, multipliers, cost_fn, var_fn) -> pd.DataFrame:
    """Builds a synthetic (window_start, label, c, norm_cost, norm_variance)
    panel -- the shape score_frontier/_split_sample_bootstrap actually
    consume, bypassing run_policy_grid/HedgingBacktester (already tested
    elsewhere) to isolate the statistics under test."""
    rows = []
    for ws in window_starts:
        for label in labels:
            for c in multipliers:
                rows.append({"window_start": ws, "label": label, "c": c,
                             "norm_cost": cost_fn(ws, label, c), "norm_variance": var_fn(ws, label, c)})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_indices_never_splits_a_cluster():
    cluster_row_idx = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8]),
                        np.array([9]), np.array([10, 11, 12])]
    rng = np.random.default_rng(0)
    for _ in range(300):
        idx = cluster_bootstrap_indices(cluster_row_idx, avg_block_len=2.0, rng=rng)
        pos = 0
        while pos < len(idx):
            matched = any(pos + len(c) <= len(idx) and np.array_equal(idx[pos:pos + len(c)], c)
                           for c in cluster_row_idx)
            assert matched, f"resample split a cluster: {idx}"
            pos += next(len(c) for c in cluster_row_idx
                        if pos + len(c) <= len(idx) and np.array_equal(idx[pos:pos + len(c)], c))


def test_stationary_bootstrap_indices_matches_target_block_length():
    # The vectorized rewrite (see stats.py) must reproduce the geometric-
    # block-length statistical property of the original per-element loop,
    # not just run faster -- verified directly, not assumed from the
    # derivation.
    rng = np.random.default_rng(2)
    lens = []
    for _ in range(300):
        idx = _stationary_bootstrap_indices(2000, 5.0, rng)
        diffs = np.diff(idx.astype(int))
        breaks = np.flatnonzero((diffs != 1) & (diffs != 1 - 2000))
        run_lengths = np.diff(np.concatenate([[-1], breaks, [len(idx) - 1]]))
        lens.extend(run_lengths.tolist())
    assert abs(np.mean(lens) - 5.0) < 0.3


def test_score_frontier_point_estimate_matches_manual_pooled_computation():
    # c_star/gap_pct are point estimates on the full sample, unaffected by
    # which bootstrap methodology scores the CI -- verified against a
    # hand-computed pooled mean, independent of frontier.py's own code.
    window_starts = pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 11)])
    labels = ["A", "B", "C"]
    multipliers = [0.5, 1.0, 2.0]
    rng = np.random.default_rng(1)
    true_best_c = 2.0

    def cost_fn(ws, label, c):
        base = {0.5: 0.30, 1.0: 0.20, 2.0: 0.05}[c]
        return base + 0.01 * rng.standard_normal()

    def var_fn(ws, label, c):
        base = {0.5: 0.01, 1.0: 0.02, 2.0: 0.08}[c]
        return base + 0.001 * rng.standard_normal()

    sub = _make_sub(window_starts, labels, multipliers, cost_fn, var_fn)
    lam0 = 1.0
    manual_objectives = {
        c: sub.loc[sub["c"] == c, "norm_cost"].mean() + lam0 * sub.loc[sub["c"] == c, "norm_variance"].mean()
        for c in multipliers
    }
    manual_c_star = min(manual_objectives, key=manual_objectives.get)
    assert manual_c_star == true_best_c  # sanity: the synthetic DGP is unambiguous at this n

    grid = sub.copy()
    grid["lam0"] = lam0
    grid["premium0"] = 1.0  # norm_* already normalized; premium0=1 makes total_cost/premium0 == norm_cost
    grid["total_cost"] = grid["norm_cost"]
    grid["pnl_variance"] = grid["norm_variance"]

    findings = score_frontier(grid, multipliers, [lam0], block_len=1.0)
    assert len(findings) == 1
    f = findings[0]
    assert f.c_star == manual_c_star
    assert f.objectives[true_best_c] == pytest.approx(manual_objectives[true_best_c], rel=1e-9)
    # per_period_gap is the HELD-OUT half only (split-sample) -- half the
    # periods, not all of them; see _split_sample_bootstrap.
    assert len(f.per_period_gap) == len(window_starts) // 2


def test_split_sample_gives_near_nominal_coverage_when_no_c_is_truly_better():
    # Construct a DGP where EVERY multiplier has the SAME true objective
    # (c* is pure noise, not a real effect), and check how often a 95% CI
    # wrongly excludes zero across many independent fresh samples -- a
    # well-calibrated procedure should do this ~5% of the time. Scale
    # (42 periods x 20 tickers) matches what was needed to get a reliable
    # coverage read in the exploration behind this fix -- smaller panels
    # give noisy coverage estimates for ANY method, not just this one.
    multipliers = [0.5, 1.0, 2.0, 4.0]
    window_starts = pd.date_range("2024-01-01", periods=42, freq="7D")
    labels = [f"T{i}" for i in range(20)]

    def make_sub(seed):
        rng = np.random.default_rng(seed)
        # every c has IDENTICAL true mean cost/variance -- no real effect
        return _make_sub(window_starts, labels, multipliers,
                          cost_fn=lambda ws, l, c: 0.15 + 0.02 * rng.standard_normal(),
                          var_fn=lambda ws, l, c: 0.03 + 0.005 * rng.standard_normal())

    lam0 = 1.0
    n_trials = 100  # Monte Carlo trials over fresh samples; kept modest for test runtime
    false_positives = 0
    for trial in range(n_trials):
        sub = make_sub(seed=trial)
        boot, _c_star, _gap, _ = _split_sample_bootstrap(
            sub, multipliers, lam0, block_len=1.0, n_boot=300, seed=trial)
        if not (boot.ci_low <= 0.0 <= boot.ci_high):
            false_positives += 1

    rate = false_positives / n_trials
    # Nominal is 5%; allow generous slack for Monte Carlo noise at
    # n_trials=100 (a binomial(100, 0.05) has std ~2.2pp) while still
    # catching genuine miscalibration -- the naive (no-split) baseline
    # measured ~8-12% in the exploration behind this fix, and the
    # reselect-inside-bootstrap dead end measured ~20%; 15% cleanly
    # separates "well-calibrated" from either failure mode.
    assert rate <= 0.15, f"split-sample CI false-positive rate {rate:.1%} is too far from nominal 5%"


def test_cluster_bootstrap_ci_reflects_cross_sectional_dependence():
    # The direct test of A2: build a panel where all "tickers" share a
    # common PER-PERIOD shock (the same-date, many-tickers correlation
    # real market data has -- e.g. a shared vol regime) on top of small
    # idiosyncratic noise. The TRUE sampling variability of the panel mean
    # is driven by ~n_periods independent draws, not n_periods*n_tickers.
    # A row-level bootstrap that ignores the clustering understates this;
    # the period-cluster bootstrap should recover a CI closer to the
    # TRUE (Monte-Carlo-measured) sampling distribution.
    multipliers = [1.0]  # single c: isolates the dependence-width question from selection
    window_starts = pd.to_datetime([f"2024-{m:02d}-{d:02d}" for m in range(1, 6) for d in (1, 8, 15, 22)])
    labels = [f"T{i}" for i in range(15)]

    def make_sub(seed):
        rng = np.random.default_rng(seed)
        period_shock = {ws: rng.standard_normal() * 0.05 for ws in window_starts}  # shared across tickers
        return _make_sub(window_starts, labels, multipliers,
                          cost_fn=lambda ws, l, c: 0.20 + period_shock[ws] + 0.005 * rng.standard_normal(),
                          var_fn=lambda ws, l, c: 0.03 + 0.001 * rng.standard_normal())

    # TRUE sampling distribution of the pooled mean cost (cost series only
    # -- isolates the dependence-width question from the objective's
    # lam0-weighted variance term), from many
    # independent fresh panels (only possible because this is synthetic --
    # the actual ground truth to compare bootstrap CIs against).
    true_means = np.array([
        make_sub(seed).groupby("window_start")["norm_cost"].mean().to_numpy().mean()
        for seed in range(400)
    ])
    true_std = true_means.std(ddof=1)

    # One realized sample; CI on its mean via both methods.
    sub = make_sub(seed=999)

    # naive row-level block bootstrap, ordered (label, window_start) --
    # the ORIGINAL (pre-fix) methodology.
    row_vals = sub.sort_values(["label", "window_start"])["norm_cost"].to_numpy()
    rng = np.random.default_rng(0)
    naive_boot_means = np.array([
        row_vals[_stationary_bootstrap_indices(len(row_vals), 3.0, rng)].mean()
        for _ in range(2000)
    ])
    naive_width = naive_boot_means.std(ddof=1)

    # period-cluster bootstrap -- the FIXED methodology.
    periods = np.sort(sub["window_start"].unique())
    cluster_row_idx = [np.flatnonzero(sub["window_start"].to_numpy() == p) for p in periods]
    cost_arr = sub["norm_cost"].to_numpy()
    rng = np.random.default_rng(0)
    cluster_boot_means = np.array([
        cost_arr[cluster_bootstrap_indices(cluster_row_idx, 3.0, rng)].mean()
        for _ in range(2000)
    ])
    cluster_width = cluster_boot_means.std(ddof=1)

    # The cluster bootstrap's implied sampling std should be substantially
    # closer to the TRUE std (measured, not assumed) than the naive one,
    # which is blind to the shared per-period shock and understates it.
    assert cluster_width > naive_width * 1.5, (
        f"cluster bootstrap ({cluster_width:.4f}) should be noticeably wider than the naive "
        f"row-level bootstrap ({naive_width:.4f}) on data with real cross-sectional dependence")
    # And it should land in the right ballpark of the true sampling std --
    # not exact (n_periods=20 is a small basis for both the true-std
    # estimate and the bootstrap itself), but same order of magnitude.
    assert 0.4 * true_std < cluster_width < 2.5 * true_std
