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

from pathlib import Path

import numpy as np
import pandas as pd

from bscpp.backtest.frontier import print_frontier_report, run_policy_grid, score_frontier

GRID_OUTPUT_DIR = Path(__file__).parent / "output"

# Identical to hedging_policy_frontier_study.py.
MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RISK_AVERSIONS = [0.03, 0.3, 3.0]
TRANSACTION_COST_BPS = 5.0
RATE = 0.05
WINDOW_DAYS = 45
TRAILING_WINDOW_DAYS = 45  # same trailing length real study uses for the vol estimate

VOL_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35]
PATHS_PER_VOL = 10
SPOT0 = 100.0
SEED = 20260813


def _annualized_realized_vol(closes: pd.Series) -> float:
    """Identical estimator to hedging_policy_frontier_study.py's
    annualized_realized_vol: calendar-clock, matching the backtester's own
    clock exactly (see that module and the CHANGELOG fix history)."""
    closes = closes.dropna()
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < 2:
        return float("nan")
    elapsed_years = (closes.index[-1] - closes.index[0]).days / 365.0
    if elapsed_years <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(log_returns.to_numpy() ** 2) / elapsed_years))


def simulate_gbm_windows(vol_levels, paths_per_vol, seed) -> list[dict]:
    """For each (vol level, path index), simulates a TRAILING_WINDOW_DAYS
    calibration segment immediately followed by the WINDOW_DAYS hedging
    window, both under the same true vol and the same continuous business-
    day path (so the trailing estimate is a genuine out-of-sample estimate
    of what the hedging window will realize, exactly mirroring how the
    real study estimates hedge_vol on data preceding, not overlapping,
    the hedged window). Step sizes use each date's real calendar-day gap
    (1 midweek, 3 over a weekend) -- the same clock HedgingBacktester.run
    uses for real price data.

    Returns two window lists (true_vol, estimated_vol) over the IDENTICAL
    simulated paths -- same random draws -- so any difference between the
    two arms' results is attributable only to hedge_vol being true vs.
    estimated, not to sampling noise from different paths.
    """
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp("2024-01-01")
    true_vol_windows, estimated_vol_windows = [], []

    for vol in vol_levels:
        for path_idx in range(paths_per_vol):
            total_days = TRAILING_WINDOW_DAYS + WINDOW_DAYS
            dates = pd.bdate_range(anchor + pd.Timedelta(days=path_idx * 3), periods=total_days)
            day_gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
            dt_years = day_gaps / 365.0

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
    grid = run_policy_grid(windows, MULTIPLIERS, RISK_AVERSIONS, RATE, TRANSACTION_COST_BPS)
    GRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_OUTPUT_DIR / save_name, index=False)  # lets plot_hedging_frontier.py reuse this run
    # Independent simulated paths, no time overlap, no shared randomness --
    # unlike the real study's rolling windows there is no serial dependence
    # to preserve, so block length 1 (plain i.i.d. resampling) is correct,
    # not copied from the real study's overlap-derived length.
    findings = score_frontier(grid, MULTIPLIERS, RISK_AVERSIONS, block_len=1.0)
    print_frontier_report(findings, MULTIPLIERS, label=label)
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


if __name__ == "__main__":
    main()
