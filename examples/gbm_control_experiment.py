"""Control experiment: does daily-rebalancing discretization alone explain
the empirically-wide Whalley-Wilmott band found on real data?

hedging_policy_frontier_study.py found that in the one well-posed
(risk-variance-balanced) regime, the empirically cost-risk-minimizing band
multiplier is c* != 1 -- roughly 2x wider than the WW (1997) asymptotic
theory predicts, a gap statistically distinguishable from zero. Two
candidate explanations were left open there:

  (a) Discretization: WW's derivation assumes continuous monitoring: real
      (and this project's) hedging only rebalances once a day, so some
      band-widening is mechanically expected even under the exact GBM
      dynamics the theory itself assumes -- the theory's own model,
      violated only in observation frequency.
  (b) Real-market structure: fat tails, volatility clustering, and
      autocorrelation that GBM doesn't have.

This script isolates (a) by rerunning the EXACT SAME frontier sweep
(same MULTIPLIERS, RISK_AVERSIONS, objective, per-window bootstrap
methodology) against simulated GBM paths instead of real prices, with:
  - hedge_vol set to the TRUE simulation volatility (not a trailing
    estimate), removing vol-estimation noise as a further confound --
    this isolates discretization specifically, not "any deviation from
    theory."
  - the same discrete rebalancing cadence as the real study: 45
    business-day windows (matching WINDOW_DAYS there), with calendar-day
    step sizes taken from the actual business-day gaps (1 day midweek, 3
    days over a weekend) -- literally the same clock the backtester uses
    for real price data, not a simplified every-day simulation.
  - 5 volatility levels (0.15-0.35, spanning the range this project's own
    examples treat as typical) x 10 independently-seeded paths each = 50
    simulated windows, matching the real study's ~50-window scale, with
    volatility level standing in for cross-sectional (ticker) variation.

Reading the result:
  - If c* reproduces the real study's ~2x gap here too, on data that is
    EXACTLY the model WW's own derivation assumes except for discrete
    rebalancing, the mechanism is identified: daily discretization is
    sufficient on its own, and real-market structure isn't needed to
    explain the finding.
  - If it doesn't reproduce (c* much closer to 1, or the gap is not
    statistically distinguishable from zero), discretization alone is
    NOT sufficient -- the real-data gap must come substantially from
    real-market structure GBM doesn't have (fat tails, vol clustering),
    which is the more interesting result: it means the theory's
    small-cost asymptotics hold up fine under daily rebalancing per se,
    and what breaks it is markets not being GBM.

No market data or API key needed -- this is the whole point of running it
this week rather than waiting on more real-data collection.

    python examples/gbm_control_experiment.py
"""

import numpy as np
import pandas as pd

from bscpp.backtest import HedgingBacktester
from bscpp.backtest.policies import WhalleyWilmottPolicy
from bscpp.stats import stationary_block_bootstrap

# Identical to hedging_policy_frontier_study.py -- the whole point of a
# control experiment is changing exactly one thing (the price-generating
# process), not re-deriving the analysis.
MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RISK_AVERSIONS = [0.001, 0.01, 0.1]
TRANSACTION_COST_BPS = 5.0
RATE = 0.05
WINDOW_DAYS = 45

VOL_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35]
PATHS_PER_VOL = 10
SPOT0 = 100.0
SEED = 20260813


def simulate_gbm_windows(vol_levels, paths_per_vol, seed) -> list[dict]:
    """WINDOW_DAYS-long business-day price paths under exact GBM at each
    vol level, using each step's REAL calendar-day gap (1 midweek, 3 over
    a weekend) for the drift/diffusion scaling -- the same clock
    HedgingBacktester.run uses internally for real price data, so
    swapping in these paths changes nothing about the discretization
    structure being tested, only the generating process.
    """
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp("2024-01-01")
    windows = []
    for vol in vol_levels:
        for path_idx in range(paths_per_vol):
            # stagger anchors so (ticker, window_start) pairs are unique in
            # the pivot below; paths are independent regardless of dates
            dates = pd.bdate_range(anchor + pd.Timedelta(days=path_idx * 3), periods=WINDOW_DAYS)
            day_gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
            dt_years = day_gaps / 365.0

            z = rng.normal(size=len(dt_years))
            log_ret = (RATE - 0.5 * vol**2) * dt_years + vol * np.sqrt(dt_years) * z
            prices = SPOT0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
            series = pd.Series(prices, index=dates)

            windows.append({
                "ticker": f"gbm_vol_{vol:.2f}", "window": series, "hedge_vol": vol,
            })
    return windows


def run_policy_grid(windows: list[dict]) -> pd.DataFrame:
    """Identical logic to hedging_policy_frontier_study.run_policy_grid."""
    rows = []
    for w in windows:
        window, hedge_vol = w["window"], w["hedge_vol"]
        spot0 = float(window.iloc[0])
        strike = round(spot0 / 5) * 5
        expiration = window.index[-1].date()

        for lam0 in RISK_AVERSIONS:
            for c in MULTIPLIERS:
                backtester = HedgingBacktester(rate=RATE, transaction_cost_bps=TRANSACTION_COST_BPS)
                policy = WhalleyWilmottPolicy(risk_aversion=lam0 / c ** 3)
                try:
                    result = backtester.run(window, strike=strike, expiration=expiration,
                                             hedge_vol=hedge_vol, option_type="call", policy=policy)
                except Exception:
                    continue
                attributed = backtester.attribute_pnl(result)
                rows.append({
                    "ticker": w["ticker"], "window_start": window.index[0].date(),
                    "lam0": lam0, "c": c,
                    "total_cost": result["transaction_cost"].sum(),
                    "pnl_variance": float(attributed["realized_pnl"].var(ddof=1)),
                })
    return pd.DataFrame(rows)


def main():
    windows = simulate_gbm_windows(VOL_LEVELS, PATHS_PER_VOL, SEED)
    print(f"{len(windows)} simulated GBM windows "
          f"({len(VOL_LEVELS)} vol levels x {PATHS_PER_VOL} paths), "
          f"{WINDOW_DAYS} business days each, hedge_vol = true simulation vol\n")

    grid = run_policy_grid(windows)
    # Paths are independently simulated with no time overlap and no shared
    # randomness -- unlike the real study's overlapping rolling windows,
    # there is no serial dependence to preserve here, so the block
    # bootstrap is deliberately run at block length 1 (plain i.i.d.
    # resampling), not copied from the real study's overlap-derived length.
    block_len = 1.0

    print(f"{'lam0':>8} {'c':>6} {'mean_cost':>11} {'mean_var':>11} {'objective':>11}")
    findings = []
    for lam0 in RISK_AVERSIONS:
        sub = grid[grid["lam0"] == lam0]
        objectives = {}
        for c in MULTIPLIERS:
            cell = sub[sub["c"] == c]
            mean_cost = cell["total_cost"].mean()
            mean_var = cell["pnl_variance"].mean()
            objective = mean_cost + lam0 * mean_var
            objectives[c] = objective
            print(f"{lam0:>8} {c:>6} {mean_cost:>11.4f} {mean_var:>11.4f} {objective:>11.4f}")

        c_star = min(objectives, key=objectives.get)
        at_boundary = c_star in (MULTIPLIERS[0], MULTIPLIERS[-1])
        gap_pct = (objectives[1.0] - objectives[c_star]) / objectives[c_star] * 100

        pivot = sub.pivot_table(index=["ticker", "window_start"], columns="c",
                                 values=["total_cost", "pnl_variance"])
        per_window_gap = (
            (pivot[("total_cost", 1.0)] + lam0 * pivot[("pnl_variance", 1.0)])
            - (pivot[("total_cost", c_star)] + lam0 * pivot[("pnl_variance", c_star)])
        ).to_numpy()
        boot = stationary_block_bootstrap(per_window_gap, avg_block_len=block_len)

        findings.append((lam0, c_star, gap_pct, boot, at_boundary))
        boundary_note = "  [GRID-BOUNDARY -- not a well-posed interior optimum]" if at_boundary else ""
        print(f"  -> empirical optimum c*={c_star} (theory predicts c=1); "
              f"objective gap {gap_pct:+.1f}% ; {boot}{boundary_note}\n")

    print("Summary:")
    for lam0, c_star, gap_pct, boot, at_boundary in findings:
        distinguishable = not (boot.ci_low <= 0.0 <= boot.ci_high)
        if at_boundary:
            print(f"  risk_aversion={lam0}: INCONCLUSIVE -- objective still improving at the "
                  f"c={c_star} grid edge (cost-dominated regime), same as the real-data study.")
        else:
            verdict = "distinguishable from theory" if distinguishable else "not distinguishable from theory"
            print(f"  risk_aversion={lam0}: c*={c_star} (theory=1), gap={gap_pct:+.1f}%, "
                  f"bootstrap CI excludes 0: {distinguishable} -> {verdict}")

    interior = [f for f in findings if not f[4]]
    real_c_star, real_gap_pct = 2.0, 7.0  # hedging_policy_frontier_study.py's well-posed-regime result
    print(
        f"\nControl-experiment verdict (real-data well-posed regime: c*~{real_c_star:g}x theory, "
        f"gap ~+{real_gap_pct:.0f}%, bootstrap CI excluded 0):"
    )
    if not interior:
        print(
            "  No well-posed (interior) regime here either -- every tested risk-aversion "
            "is cost-dominated under pure GBM at this cost/variance scale, so this control "
            "can't speak to the mechanism at all with this grid. Re-run with a wider "
            "RISK_AVERSIONS grid before concluding anything."
        )
        return

    distinguishable_interior = [f for f in interior if not (f[3].ci_low <= 0.0 <= f[3].ci_high)]
    if not distinguishable_interior:
        print(
            "  NOT REPRODUCED: the well-posed regime(s) here are statistically consistent "
            "with c*=1 under pure GBM with daily rebalancing. Discretization alone is NOT "
            "sufficient to explain the real-data finding -- the widening found on real data "
            "implicates real-market structure GBM doesn't have (fat tails, volatility "
            "clustering, autocorrelation), a stronger and more interesting result than 'the "
            "theory just needs a discretization correction'."
        )
        return

    lam0, c_star, gap_pct, boot, _ = min(distinguishable_interior, key=lambda f: f[0])
    print(
        f"  Discretization alone DOES produce a real, statistically significant, same-"
        f"direction widening effect: c*={c_star:g}x theory, gap={gap_pct:+.1f}% (risk_aversion="
        f"{lam0}), bootstrap CI excludes 0. Pure GBM with only daily (not continuous) "
        f"monitoring is enough to move the empirical optimum away from c=1 -- WW's "
        f"continuous-monitoring assumption is a real, non-negligible source of the gap, not "
        f"a negligible technicality."
    )
    if c_star > real_c_star * 1.3:
        print(
            f"  But the magnitude OVERSHOOTS the real-data result ({c_star:g}x vs ~{real_c_star:g}x "
            f"theory; {gap_pct:+.1f}% vs ~+{real_gap_pct:.0f}%): pure-GBM discretization alone "
            f"predicts MORE band-widening than is actually observed on real data. That rules "
            f"out 'discretization explains none of it,' but it also means real-market "
            f"structure isn't simply ADDING extra widening on top of a GBM baseline -- "
            f"something about real dynamics (fat tails, vol clustering, autocorrelation) "
            f"appears to pull the empirical optimum back TOWARD theory relative to what "
            f"pure discretization alone would predict. Worth a follow-up: is that a genuine "
            f"dynamical effect, or a scale mismatch between this script's vol/cost grid and "
            f"the real study's actual realized vols?"
        )
    elif c_star < real_c_star / 1.3:
        print(
            f"  The magnitude UNDERSHOOTS the real-data result ({c_star:g}x vs ~{real_c_star:g}x "
            f"theory; {gap_pct:+.1f}% vs ~+{real_gap_pct:.0f}%): discretization is a real "
            f"contributor but not the whole story -- real-market structure likely accounts "
            f"for the remaining gap between this control's {c_star:g}x and the real study's "
            f"~{real_c_star:g}x."
        )
    else:
        print(
            f"  And the magnitude ROUGHLY MATCHES the real-data result ({c_star:g}x vs "
            f"~{real_c_star:g}x theory) -- discretization looks like a sufficient explanation "
            f"on its own; real-market structure isn't needed as an additional cause."
        )


if __name__ == "__main__":
    main()
