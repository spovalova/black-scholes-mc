"""Control experiments: does daily-rebalancing discretization alone explain
the empirically-wide Whalley-Wilmott band found on real data -- and how
much of it is actually vol-estimation error rather than dynamics?

hedging_policy_frontier_study.py found that in the well-posed (risk-
variance-balanced) regimes, the empirically cost-risk-minimizing band
multiplier is c* != 1 on real data. Three candidate explanations:
  (a) Discretization: WW's derivation assumes continuous monitoring; daily
      rebalancing alone could produce some widening even under the
      theory's own GBM assumption.
  (b) Vol-estimation error: the real study marks the option at a TRAILING
      REALIZED vol estimate, not the true (unknowable) instantaneous vol --
      any resulting mismatch between hedge_vol and the path's actual
      volatility could itself widen the empirically-optimal band,
      independent of whether the underlying dynamics are GBM or not.
  (c) Real-market structure: fat tails, volatility clustering,
      autocorrelation that GBM doesn't have.

This script runs two GBM arms that isolate (a) and (b) from each other and
from (c):
  - "gbm_true_vol": hedge_vol = the TRUE simulation volatility (no
    estimation noise at all). Isolates (a) alone.
  - "gbm_estimated_vol": hedge_vol = a TRAILING-WINDOW REALIZED vol
    estimate computed exactly the way the real study computes it (same
    trailing-window length, same annualized_realized_vol formula), on the
    SAME underlying GBM paths as gbm_true_vol. Isolates (a)+(b) together;
    comparing its c* against gbm_true_vol's isolates (b) alone.

Both arms use the identical objective and methodology as
hedging_policy_frontier_study.py: J(c) = mean(cost/premium0) + lam0 *
mean(variance/premium0^2), normalized per-window by the window's own
option premium so results are directly comparable across configurations
of very different dollar scale (see bscpp.backtest.frontier). hedge_vol
is estimated on TRAILING data preceding each window, exactly as the real
study does -- never on the window itself, which would leak future
information into the marking vol.

Reading the result: if gbm_estimated_vol's c* lands close to the real
study's, while gbm_true_vol's does not, discretization is NOT the primary
driver -- vol-estimation error is doing most of the work, and that is a
sharper, more useful finding than "daily monitoring matters somewhat":
it says the real-data gap is substantially about not knowing the true
vol, not about markets failing to be GBM.

No market data or API key needed.

    python examples/gbm_control_experiment.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from bscpp.backtest.frontier import (
    print_frontier_report,
    run_policy_grid,
    score_frontier,
)
from bscpp.clock import Clock

GRID_OUTPUT_DIR = Path(__file__).parent / "output"

# Identical to hedging_policy_frontier_study.py (see that file for why the
# intermediate points were added, not just the power-of-2 anchors).
MULTIPLIERS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0]
RISK_AVERSIONS = [0.03, 0.3, 3.0]
TRANSACTION_COST_BPS = 5.0
RATE = 0.05
WINDOW_DAYS = 45
TRAILING_WINDOW_DAYS = 45  # same trailing length real study uses for the vol estimate

VOL_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35]
# Was 10 (50 windows total): with the real study's ~420 windows, a control
# arm at that scale had ~1/8th the statistical power to detect a
# vol-estimation-error effect it was supposed to be measuring precisely --
# underpowered for the strength of the "ruled out" language this study
# uses. 100 is cheap (the harness runs a full 11-point-grid x 3-regime
# sweep over 500 windows in well under a minute -- see the timing this
# script prints), so there's no reason to stay thin here.
PATHS_PER_VOL = 100
SPOT0 = 100.0
SEED = 20260813


# Identical estimator to hedging_policy_frontier_study.py's
# annualized_realized_vol: calendar clock (ACT/365), matching
# HedgingBacktester's own default clock exactly -- see bscpp.clock.Clock
# and the CHANGELOG fix history.
_CLOCK = Clock()
_annualized_realized_vol = _CLOCK.annualized_realized_vol


def simulate_gbm_windows(vol_levels, paths_per_vol, seed, dt_mode: str = "calendar") -> list[dict]:
    """For each (vol level, path index), simulates a TRAILING_WINDOW_DAYS
    calibration segment immediately followed by the WINDOW_DAYS hedging
    window, both under the same true vol and the same continuous business-
    day path (so the trailing estimate is a genuine out-of-sample estimate
    of what the hedging window will realize, exactly mirroring how the
    real study estimates hedge_vol on data preceding, not overlapping,
    the hedged window).

    dt_mode="calendar" (default): step sizes use each date's real
    calendar-day gap (1 midweek, 3 over a weekend) -- the same clock
    HedgingBacktester.run uses for real price data, so the PRICE PATH's
    own variance is diffused over 3 calendar days across a weekend even
    though real markets realize roughly ONE trading day's worth of
    variance there, not three. dt_mode="trading": every step instead uses
    a UNIFORM dt=1/252 regardless of the real calendar gap -- isolating
    whether that calendar-clock artifact in the SIMULATED PRICE PATH
    itself (not the backtester's own ACT/365 financing/theta accounting,
    which is unchanged in both modes -- only the diffusion's own step
    size varies here) is doing any of the work in "gbm_true_vol
    reproduces the real-data optimum almost exactly". Real calendar
    dates are used for indexing either way (needed for the backtester's
    own clock and for score_frontier's window_start clustering); only the
    variance-per-step assigned to each diffusion step differs.

    Returns two window lists (true_vol, estimated_vol) over the IDENTICAL
    simulated paths -- same random draws -- so any difference between the
    two arms' results is attributable only to hedge_vol being true vs.
    estimated, not to sampling noise from different paths.
    """
    if dt_mode not in ("calendar", "trading"):
        raise ValueError(f"dt_mode must be 'calendar' or 'trading', got {dt_mode!r}")
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp("2024-01-01")
    true_vol_windows, estimated_vol_windows = [], []

    for vol_idx, vol in enumerate(vol_levels):
        for path_idx in range(paths_per_vol):
            total_days = TRAILING_WINDOW_DAYS + WINDOW_DAYS
            # Offsetting by path_idx alone made every vol level reuse the
            # SAME `paths_per_vol` calendar anchors (path_idx cycles 0..9
            # under each of 5 vol levels) -- 50 windows collapsed onto
            # just 10 distinct window_start dates. That's not a cosmetic
            # issue: score_frontier's split-sample bootstrap clusters by
            # window_start specifically to capture GENUINE same-date
            # cross-sectional dependence (see frontier.py) -- these paths
            # share no randomness across vol levels (fresh rng.normal()
            # draws per iteration) and have no real reason to be
            # calendar-coincident, so the collision only made an already-
            # independent control arm's CI needlessly conservative,
            # clustering unrelated paths together for no statistical
            # reason. The large per-vol-level offset below guarantees
            # every (vol, path_idx) combination gets its own distinct
            # anchor, restoring the genuine one-window-per-cluster
            # independence this arm is supposed to have.
            vol_offset_days = vol_idx * (paths_per_vol * 3 + total_days)
            dates = pd.bdate_range(anchor + pd.Timedelta(days=vol_offset_days + path_idx * 3),
                                    periods=total_days)
            if dt_mode == "calendar":
                day_gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
                dt_years = day_gaps / 365.0
            else:
                dt_years = np.full(total_days - 1, 1.0 / 252.0)

            z = rng.normal(size=len(dt_years))
            log_ret = (RATE - 0.5 * vol**2) * dt_years + vol * np.sqrt(dt_years) * z
            prices = SPOT0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
            full_series = pd.Series(prices, index=dates)

            # Strictly adjacent, non-overlapping split -- matches
            # hedging_policy_frontier_study.py's collect_windows exactly
            # (trailing ends at start_idx-1, window starts at start_idx).
            trailing = full_series.iloc[:TRAILING_WINDOW_DAYS]
            window = full_series.iloc[TRAILING_WINDOW_DAYS:]
            estimated_vol = _annualized_realized_vol(trailing)
            if not (0.02 < estimated_vol < 3.0):
                continue

            label = f"gbm_vol_{vol:.2f}"
            true_vol_windows.append({"label": label, "window": window, "hedge_vol": vol})
            estimated_vol_windows.append({"label": label, "window": window, "hedge_vol": estimated_vol})

    return true_vol_windows, estimated_vol_windows


def run_arm(windows, label, save_name):
    t0 = time.perf_counter()
    grid = run_policy_grid(windows, MULTIPLIERS, RISK_AVERSIONS, RATE, TRANSACTION_COST_BPS)
    grid_time = time.perf_counter() - t0
    GRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_OUTPUT_DIR / save_name, index=False)  # lets plot_hedging_frontier.py reuse this run
    # Independent simulated paths, no time overlap, no shared randomness --
    # unlike the real study's rolling windows there is no serial dependence
    # to preserve, so block length 1 (plain i.i.d. resampling) is correct,
    # not copied from the real study's overlap-derived length.
    t0 = time.perf_counter()
    findings = score_frontier(grid, MULTIPLIERS, RISK_AVERSIONS, block_len=1.0)
    score_time = time.perf_counter() - t0
    print_frontier_report(findings, MULTIPLIERS, label=label)
    print(f"  [{len(windows)} windows x {len(RISK_AVERSIONS)} lam0 x {len(MULTIPLIERS)} c grid: "
          f"{grid_time:.1f}s backtesting, {score_time:.1f}s split-sample bootstrap scoring]\n")
    return findings


def main():
    true_vol_windows, estimated_vol_windows = simulate_gbm_windows(VOL_LEVELS, PATHS_PER_VOL, SEED)
    print(f"{len(true_vol_windows)} simulated GBM windows "
          f"({len(VOL_LEVELS)} vol levels x {PATHS_PER_VOL} paths), {WINDOW_DAYS} business days "
          f"each, {TRAILING_WINDOW_DAYS}-day trailing estimation segment preceding each\n")

    true_vol_findings = run_arm(true_vol_windows, "gbm_true_vol (isolates discretization alone)",
                                 "gbm_true_vol_grid.csv")
    estimated_vol_findings = run_arm(
        estimated_vol_windows, "gbm_estimated_vol (isolates discretization + vol-estimation error)",
        "gbm_estimated_vol_grid.csv")

    print("Summary (theory predicts c*=1 in all arms):")
    for tv, ev in zip(true_vol_findings, estimated_vol_findings):
        assert tv.lam0 == ev.lam0
        print(f"  risk_aversion={tv.lam0}:")
        for f, tag in [(tv, "true_vol"), (ev, "estimated_vol")]:
            if f.at_boundary:
                print(f"    {tag}: INCONCLUSIVE -- objective still improving at the c={f.c_star} "
                      f"grid edge (cost-dominated regime)")
            else:
                print(f"    {tag}: c*={f.c_star}, gap={f.gap_pct:+.1f}%, "
                      f"distinguishable from theory: {f.distinguishable}")

    interior_pairs = [(tv, ev) for tv, ev in zip(true_vol_findings, estimated_vol_findings)
                       if not tv.at_boundary and not ev.at_boundary]
    if not interior_pairs:
        print("\nNo well-posed regime in both arms with this grid -- widen RISK_AVERSIONS "
              "before drawing conclusions.")
        return

    print("\nDecomposition (true_vol isolates discretization; estimated_vol adds vol-estimation "
          "error on top -- the estimated_vol/true_vol gap is what estimation error alone "
          "contributes):")
    for tv, ev in interior_pairs:
        print(f"  risk_aversion={tv.lam0}: discretization alone -> c*={tv.c_star}x theory; "
              f"+ vol-estimation error -> c*={ev.c_star}x theory")
    print(
        "\nCompare both against hedging_policy_frontier_study.py's real-data c* in the same "
        "well-posed regime: whichever GBM arm lands closer identifies the dominant driver -- "
        "if estimated_vol is much closer to the real-data result than true_vol is, "
        "vol-estimation error (not markets failing to be GBM) explains most of the gap; if "
        "both GBM arms undershoot the real-data result by similar amounts, real-market "
        "structure (fat tails, vol clustering) is still implicated for the remainder."
    )

    # Robustness check: gbm_true_vol above diffuses its simulated price
    # path over each date's REAL calendar-day gap (3 calendar days across
    # a weekend, same as HedgingBacktester's own ACT/365 clock) -- but
    # real markets realize roughly ONE trading day's worth of variance
    # over a weekend, not three. Given how central "gbm_true_vol
    # reproduces the real-data optimum almost exactly" is to the
    # discretization-not-vol-estimation-error conclusion above, this
    # tests whether that calendar-clock artifact in the PRICE DIFFUSION
    # itself (not the backtester's own financing/theta accounting, which
    # is unchanged here -- only the diffusion's per-step variance
    # differs) is doing any of the work: a trading-clock variant uses a
    # UNIFORM dt=1/252 per business-day step regardless of the real
    # calendar gap, removing the weekend-variance artifact entirely.
    print("\n=== Robustness: does the calendar-clock's weekend variance "
          "(3 days diffused over a weekend, vs. real markets' ~1 trading day) drive the "
          "gbm_true_vol match to real data? ===")
    trading_clock_windows, _ = simulate_gbm_windows(VOL_LEVELS, PATHS_PER_VOL, SEED, dt_mode="trading")
    trading_clock_findings = run_arm(
        trading_clock_windows, "gbm_true_vol_trading_clock (uniform dt=1/252, no weekend-variance artifact)",
        "gbm_true_vol_trading_clock_grid.csv")

    print("Comparison (theory predicts c*=1 in both):")
    for tv, tc in zip(true_vol_findings, trading_clock_findings):
        assert tv.lam0 == tc.lam0
        if tv.at_boundary or tc.at_boundary:
            print(f"  risk_aversion={tv.lam0}: INCONCLUSIVE (grid-boundary in at least one arm)")
            continue
        moved = "" if tv.c_star == tc.c_star else f"  <-- MOVED (calendar clock: {tv.c_star}x)"
        print(f"  risk_aversion={tv.lam0}: calendar-clock c*={tv.c_star}x, "
              f"trading-clock c*={tc.c_star}x{moved}")
    print(
        "If trading-clock c* matches calendar-clock c* (or is very close), the weekend-variance "
        "artifact is NOT what's driving the match to real data -- discretization alone, "
        "independent of how weekend time is treated, is doing the work. If trading-clock c* "
        "moves meaningfully AWAY from real data's c*, the calendar-clock's weekend variance was "
        "itself contributing to the match, and the 'discretization is sufficient on its own' "
        "conclusion above needs qualifying."
    )


if __name__ == "__main__":
    main()
