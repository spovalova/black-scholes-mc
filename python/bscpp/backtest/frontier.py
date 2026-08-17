"""Shared band-multiplier x risk-aversion frontier sweep, usable across any
collection of price windows -- real market data, simulated GBM with true
vol, simulated GBM with trailing-estimated vol, or any future arm -- scored
with a single SCALE-INVARIANT objective so results are directly comparable
across configurations, not just individually plausible.

Why scale-invariant: dollar transaction cost scales with spot*turnover and
dollar P&L variance scales with spot^2*sigma^2*T, so a fixed lambda in a
raw-dollar mean-variance objective means something different for a ~$580
SPY window than a $100 GBM path -- exactly the scale confound a
cross-configuration comparison (real vs. simulated, cheap vs. expensive
names) cannot afford to have baked into it. Normalizing cost and variance
by each window's OWN option premium (not a global constant) fixes this:
cost-as-a-fraction-of-premium and variance-as-a-fraction-of-premium^2 are
dimensionless and put every window on equal footing before pooling,
independent of the underlying's price level or the option's absolute
dollar value. (The alternative -- normalizing by S^2*sigma^2*T -- is also
defensible; premium is preferred here because it ties the objective
directly to the economic stake of the specific position being hedged,
which is what a mean-variance trade-off on THIS trade should be measured
against.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bscpp.backtest.hedging import HedgingBacktester
from bscpp.backtest.policies import WhalleyWilmottPolicy
from bscpp.stats import (
    BootstrapResult,
    cluster_bootstrap_indices,
    effective_sample_size,
)

_PREMIUM_FLOOR = 1e-6  # guards against division blowups on near-worthless options


def run_policy_grid(windows: list[dict], multipliers: list[float], risk_aversions: list[float],
                     rate: float, transaction_cost_bps: float, option_type: str = "call") -> pd.DataFrame:
    """windows: each a dict with "label" (str, e.g. ticker or arm tag), "window"
    (pd.Series of prices), "hedge_vol" (float or pd.Series), and optionally
    "strike"/"expiration" (default: true ATM at window start / window's
    last date).

    Returns one row per (window, lam0, c) with raw total_cost, pnl_variance,
    and premium0 (the option's day-0 price) -- normalization happens in
    score_frontier, not here, so this table still supports raw-dollar
    inspection if that's ever useful. The returned DataFrame's `.attrs`
    dict carries `expected_cells`/`dropped_by_reason`/`dropped_pct`: cells
    are dropped (backtester exception, or a near-worthless premium
    guarding against a division blowup downstream) SILENTLY into the row
    list otherwise -- if drops concentrate in particular (c, lam0) cells
    or tickers, those cells are differentially thinned in a way that
    would otherwise be invisible. A warnings.warn() fires when the drop
    rate exceeds 1% of the expected grid, so a real problem doesn't have
    to be discovered by manually inspecting .attrs after the fact.
    """
    rows = []
    dropped_by_reason: dict[str, int] = {}
    expected_cells = len(windows) * len(risk_aversions) * len(multipliers)
    for w in windows:
        window, hedge_vol = w["window"], w["hedge_vol"]
        # True ATM, not rounded to a $5 strike grid: these are SYNTHETIC
        # options with no listed-strike constraint to respect, so rounding
        # only injects unnecessary moneyness noise -- up to ~4% on a
        # sub-$100 name -- heterogeneous across tickers of different price
        # levels, for no offsetting benefit.
        strike = w["strike"] if "strike" in w else float(window.iloc[0])
        expiration = w["expiration"] if "expiration" in w else window.index[-1].date()

        for lam0 in risk_aversions:
            for c in multipliers:
                backtester = HedgingBacktester(rate=rate, transaction_cost_bps=transaction_cost_bps)
                policy = WhalleyWilmottPolicy(risk_aversion=lam0 / c ** 3)
                try:
                    result = backtester.run(window, strike=strike, expiration=expiration,
                                             hedge_vol=hedge_vol, option_type=option_type,
                                             policy=policy)
                except ValueError as exc:
                    dropped_by_reason[f"backtester_error: {exc}"] = (
                        dropped_by_reason.get(f"backtester_error: {exc}", 0) + 1)
                    continue
                premium0 = float(result["option_value"].iloc[0])
                if premium0 < _PREMIUM_FLOOR:
                    dropped_by_reason["premium_below_floor"] = (
                        dropped_by_reason.get("premium_below_floor", 0) + 1)
                    continue
                # realized_pnl's variance == portfolio_value.diff().var() --
                # attribute_pnl builds 6 full attribution series (financing,
                # gamma, theta, vega, delta-gap, transaction-cost P&L) per
                # call specifically to explain WHERE P&L comes from, none of
                # which this grid-scoring loop uses; computing it here just
                # to extract one variance was the single most expensive line
                # in this loop for zero additional information.
                rows.append({
                    "label": w["label"], "window_start": window.index[0].date(),
                    "lam0": lam0, "c": c,
                    "total_cost": result["transaction_cost"].sum(),
                    "pnl_variance": float(result["portfolio_value"].diff().var(ddof=1)),
                    "premium0": premium0,
                })

    grid = pd.DataFrame(rows)
    n_dropped = sum(dropped_by_reason.values())
    grid.attrs["expected_cells"] = expected_cells
    grid.attrs["dropped_by_reason"] = dropped_by_reason
    grid.attrs["dropped_pct"] = 100.0 * n_dropped / expected_cells if expected_cells else 0.0
    if n_dropped > 0.01 * expected_cells:
        warnings.warn(
            f"run_policy_grid dropped {n_dropped}/{expected_cells} cells "
            f"({grid.attrs['dropped_pct']:.1f}%) -- see grid.attrs['dropped_by_reason']: "
            f"{dropped_by_reason}", stacklevel=2)
    return grid


@dataclass
class FrontierRegime:
    """One risk-aversion regime's result: the full objective(c) curve plus
    the empirical optimum and its OUT-OF-SAMPLE-VALIDATED statistical
    significance vs. c=1.

    `objectives`/`c_star`/`gap_pct` are point estimates on the FULL sample
    (most precise -- uses all the data). `.boot`/`split_sample_c_star` are
    a SEPARATE, deliberately more conservative split-sample validation
    (see _split_sample_bootstrap) -- c* is selected on the first half of
    calendar time and tested, unselected, on the second half, specifically
    to avoid post-selection inference bias (picking the best of several
    noisy candidates and testing best-vs-baseline on the SAME data that
    did the picking is a textbook winner's-curse setup). If
    `split_sample_c_star != c_star`, the two halves disagree on which c is
    best -- itself an important, explicitly surfaced instability signal,
    not something to average away.
    """
    lam0: float
    objectives: dict  # c -> normalized objective J(c), full sample
    c_star: float  # full-sample argmin -- descriptive, NOT what .boot tests
    gap_pct: float  # % improvement of c_star's objective over c=1's, full sample
    boot: BootstrapResult  # split-sample, held-out-half, cluster-bootstrapped CI
    split_sample_c_star: float  # c* selected on the FIRST HALF of calendar periods only
    at_boundary: bool
    per_period_gap: np.ndarray = field(repr=False)  # held-out-half diagnostic series

    @property
    def distinguishable(self) -> bool:
        return not (self.boot.ci_low <= 0.0 <= self.boot.ci_high)

    @property
    def split_sample_agrees(self) -> bool:
        """False means the train half and the full sample picked a
        different c* -- the split-sample CI is still valid (it tests
        WHATEVER c* the train half picked), but this flags that the
        "headline" full-sample c* itself may not be a stable estimate."""
        return self.split_sample_c_star == self.c_star


def _split_sample_bootstrap(sub: pd.DataFrame, multipliers: list[float], lam0: float,
                             block_len: float, n_boot: int = 2000, level: float = 0.95,
                             seed: int | None = 0, train_frac: float = 0.5):
    """Split-sample CI for the c*-vs-c=1 objective gap: c* is selected on
    a TRAINING slice of calendar periods (the chronologically FIRST
    `train_frac` of window_start values) and tested, without reselecting,
    on the held-out remainder -- avoiding post-selection inference by
    construction rather than by correcting for it after the fact.

    This replaced an earlier version of this function that instead
    RE-RAN the argmin selection inside every bootstrap resample (the
    "cheapest fix" for post-selection inference suggested by an external
    review) -- that version was IMPLEMENTED, then TESTED via a Monte
    Carlo coverage simulation (many fresh synthetic samples where every
    candidate c has the identical true objective, checking how often a
    nominal-95% CI wrongly excludes zero), and found not to reliably fix
    the miscalibration -- in one tested scenario it was WORSE than doing
    nothing (20% false-positive rate vs. the naive version's 12.5%,
    against a 5% target). This is a known subtlety: percentile bootstrap
    CIs are not generally valid for argmax/argmin-based statistics (the
    resampled statistic's own distribution is itself selection-biased,
    the same problem one level up). Split-sample selection avoids the
    issue entirely -- verified via the same coverage simulation at
    realistic scale (42 periods, 20 tickers): 3.3% false-positive rate,
    close to the 5% nominal target, vs. 8% for the no-split-at-all
    baseline at the same scale. See test_frontier.py for both checks,
    kept in the test suite specifically so this doesn't regress silently.

    CROSS-SECTIONAL (same-date, many-tickers) DEPENDENCE is handled
    together with the above, not separately: this project's real-data
    study pools ~20 co-moving tickers' rolling windows, and a block
    bootstrap over rows ordered (ticker, date) -- even with blocks long
    enough to preserve WITHIN-ticker serial dependence -- is blind to
    windows from DIFFERENT tickers sharing the same window_start being
    contemporaneously correlated (same market vol regime). Fixed by
    resampling CALENDAR PERIODS as the bootstrap unit via
    cluster_bootstrap_indices (bscpp.stats) on the held-out half -- every
    ticker's rows for a resampled period move together.

    Returns (BootstrapResult, c_star_train, gap_on_test_half,
    per_period_gap) -- the last a genuinely period-level (not row-level),
    held-out-half-only diagnostic series, at the resolution the CI's own
    n/n_effective describe.
    """
    sub = sub.reset_index(drop=True)
    periods = np.sort(sub["window_start"].unique())
    n_periods = len(periods)
    if n_periods < 10:
        raise ValueError(f"need at least 10 distinct window_start periods for a 50/50 "
                          f"split-sample test, got {n_periods}")
    n_train = max(1, round(n_periods * train_frac))
    train_periods, test_periods = set(periods[:n_train]), set(periods[n_train:])

    c_arr = sub["c"].to_numpy()
    cost_arr = sub["norm_cost"].to_numpy()
    var_arr = sub["norm_variance"].to_numpy()
    window_start_arr = sub["window_start"].to_numpy()

    def objective_at(idx: np.ndarray) -> dict:
        c_sel, cost_sel, var_sel = c_arr[idx], cost_arr[idx], var_arr[idx]
        objs = {}
        for c in multipliers:
            mask = c_sel == c
            objs[c] = (cost_sel[mask].mean() + lam0 * var_sel[mask].mean()) if mask.any() else np.inf
        return objs

    train_idx = np.flatnonzero(np.isin(window_start_arr, list(train_periods)))
    train_objs = objective_at(train_idx)
    c_star_train = min(train_objs, key=train_objs.get)

    test_periods_sorted = np.array(sorted(test_periods))
    cluster_row_idx = [np.flatnonzero(window_start_arr == p) for p in test_periods_sorted]
    test_idx = np.concatenate(cluster_row_idx)
    test_objs = objective_at(test_idx)
    gap = test_objs[1.0] - test_objs[c_star_train]

    # Held-out-half, period-level diagnostic series (one value per test
    # period, at the TRAIN-selected c_star) -- the honest resolution to
    # report n/n_effective at.
    per_period_gap = np.array([
        objective_at(idx)[1.0] - objective_at(idx)[c_star_train] for idx in cluster_row_idx
    ])

    rng = np.random.default_rng(seed)
    boot_gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = cluster_bootstrap_indices(cluster_row_idx, block_len, rng)
        objs_b = objective_at(idx)
        boot_gaps[b] = objs_b[1.0] - objs_b[c_star_train]  # c_star FIXED from training -- no reselection

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot_gaps, [alpha, 1.0 - alpha])
    boot = BootstrapResult(estimate=gap, ci_low=float(lo), ci_high=float(hi), level=level,
                            n_boot=n_boot, avg_block_len=float(block_len), n=len(test_periods_sorted),
                            n_effective=effective_sample_size(per_period_gap))
    return boot, c_star_train, gap, per_period_gap


def score_frontier(grid: pd.DataFrame, multipliers: list[float], risk_aversions: list[float],
                    block_len: float, seed: int | None = 0) -> list[FrontierRegime]:
    """Scores an already-run grid (see run_policy_grid) with the
    normalized objective J(c) = mean(cost/premium0) + lam0 * mean(variance/premium0^2),
    normalizing EACH WINDOW by its own premium before pooling across
    windows -- not pooling raw dollars then dividing by an average premium,
    which would still leave dispersion from mixed price levels inside the
    pooled mean.

    c_star/gap_pct are point estimates on the full sample. The CI in
    `.boot` is a SPLIT-SAMPLE, PERIOD-CLUSTERED bootstrap (see
    _split_sample_bootstrap) -- both post-selection inference and
    cross-ticker dependence are addressed there. `block_len` is the
    average number of CONSECUTIVE CALENDAR PERIODS resampled as one block
    within the held-out half (not the earlier per-window-row block
    length) -- pass the same overlap-derived value callers already used;
    the unit changed, not the intent.
    """
    grid = grid.copy()
    grid["norm_cost"] = grid["total_cost"] / grid["premium0"]
    grid["norm_variance"] = grid["pnl_variance"] / grid["premium0"] ** 2

    findings = []
    for lam0 in risk_aversions:
        sub = grid[grid["lam0"] == lam0]
        objectives = {}
        for c in multipliers:
            cell = sub[sub["c"] == c]
            objectives[c] = cell["norm_cost"].mean() + lam0 * cell["norm_variance"].mean()

        c_star = min(objectives, key=objectives.get)
        at_boundary = c_star in (multipliers[0], multipliers[-1])
        gap_pct = (objectives[1.0] - objectives[c_star]) / objectives[c_star] * 100

        boot, split_sample_c_star, _gap, per_period_gap = _split_sample_bootstrap(
            sub, multipliers, lam0, block_len, seed=seed)

        findings.append(FrontierRegime(lam0=lam0, objectives=objectives, c_star=c_star,
                                        gap_pct=gap_pct, boot=boot,
                                        split_sample_c_star=split_sample_c_star,
                                        at_boundary=at_boundary, per_period_gap=per_period_gap))
    return findings


def print_frontier_report(findings: list[FrontierRegime], multipliers: list[float], label: str = ""):
    if label:
        print(f"=== {label} ===")
    print(f"{'lam0':>10} {'c':>6} {'norm_objective':>14}")
    for f in findings:
        for c in multipliers:
            print(f"{f.lam0:>10} {c:>6} {f.objectives[c]:>14.6f}")
        boundary_note = "  [GRID-BOUNDARY]" if f.at_boundary else ""
        agree_note = "" if f.split_sample_agrees else \
            f"  [SPLIT DISAGREES: train-half picked c*={f.split_sample_c_star}]"
        print(f"  -> c*={f.c_star} (theory=1); gap {f.gap_pct:+.1f}%{boundary_note}\n"
              f"     split-sample validation (train=first half, test=second half): "
              f"{f.boot}{agree_note}\n")
