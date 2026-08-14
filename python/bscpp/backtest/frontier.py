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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bscpp.backtest.hedging import HedgingBacktester
from bscpp.backtest.policies import WhalleyWilmottPolicy
from bscpp.stats import BootstrapResult, stationary_block_bootstrap

_PREMIUM_FLOOR = 1e-6  # guards against division blowups on near-worthless options


def run_policy_grid(windows: list[dict], multipliers: list[float], risk_aversions: list[float],
                     rate: float, transaction_cost_bps: float, option_type: str = "call") -> pd.DataFrame:
    """windows: each a dict with "label" (str, e.g. ticker or arm tag), "window"
    (pd.Series of prices), "hedge_vol" (float or pd.Series), and optionally
    "strike"/"expiration" (default: ATM-at-window-start / window's last date).

    Returns one row per (window, lam0, c) with raw total_cost, pnl_variance,
    and premium0 (the option's day-0 price) -- normalization happens in
    score_frontier, not here, so this table still supports raw-dollar
    inspection if that's ever useful.
    """
    rows = []
    for w in windows:
        window, hedge_vol = w["window"], w["hedge_vol"]
        spot0 = float(window.iloc[0])
        strike = w.get("strike") or round(spot0 / 5) * 5
        expiration = w.get("expiration") or window.index[-1].date()

        for lam0 in risk_aversions:
            for c in multipliers:
                backtester = HedgingBacktester(rate=rate, transaction_cost_bps=transaction_cost_bps)
                policy = WhalleyWilmottPolicy(risk_aversion=lam0 / c ** 3)
                try:
                    result = backtester.run(window, strike=strike, expiration=expiration,
                                             hedge_vol=hedge_vol, option_type=option_type,
                                             policy=policy)
                except Exception:
                    continue
                attributed = backtester.attribute_pnl(result)
                premium0 = float(result["option_value"].iloc[0])
                if premium0 < _PREMIUM_FLOOR:
                    continue
                rows.append({
                    "label": w["label"], "window_start": window.index[0].date(),
                    "lam0": lam0, "c": c,
                    "total_cost": result["transaction_cost"].sum(),
                    "pnl_variance": float(attributed["realized_pnl"].var(ddof=1)),
                    "premium0": premium0,
                })
    return pd.DataFrame(rows)


@dataclass
class FrontierRegime:
    """One risk-aversion regime's result: the full objective(c) curve plus
    the empirical optimum and its statistical significance vs. c=1."""
    lam0: float
    objectives: dict  # c -> normalized objective J(c)
    c_star: float
    gap_pct: float  # % improvement of c_star's objective over c=1's
    boot: BootstrapResult
    at_boundary: bool
    per_window_gap: np.ndarray = field(repr=False)

    @property
    def distinguishable(self) -> bool:
        return not (self.boot.ci_low <= 0.0 <= self.boot.ci_high)


def score_frontier(grid: pd.DataFrame, multipliers: list[float], risk_aversions: list[float],
                    block_len: float) -> list[FrontierRegime]:
    """Scores an already-run grid (see run_policy_grid) with the
    normalized objective J(c) = mean(cost/premium0) + lam0 * mean(variance/premium0^2),
    normalizing EACH WINDOW by its own premium before pooling across
    windows -- not pooling raw dollars then dividing by an average premium,
    which would still leave dispersion from mixed price levels inside the
    pooled mean.
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

        pivot = sub.pivot_table(index=["label", "window_start"], columns="c",
                                 values=["norm_cost", "norm_variance"])
        per_window_gap = (
            (pivot[("norm_cost", 1.0)] + lam0 * pivot[("norm_variance", 1.0)])
            - (pivot[("norm_cost", c_star)] + lam0 * pivot[("norm_variance", c_star)])
        ).to_numpy()
        boot = stationary_block_bootstrap(per_window_gap, avg_block_len=block_len)

        findings.append(FrontierRegime(lam0=lam0, objectives=objectives, c_star=c_star,
                                        gap_pct=gap_pct, boot=boot, at_boundary=at_boundary,
                                        per_window_gap=per_window_gap))
    return findings


def print_frontier_report(findings: list[FrontierRegime], multipliers: list[float], label: str = ""):
    if label:
        print(f"=== {label} ===")
    print(f"{'lam0':>10} {'c':>6} {'norm_objective':>14}")
    for f in findings:
        for c in multipliers:
            print(f"{f.lam0:>10} {c:>6} {f.objectives[c]:>14.6f}")
        boundary_note = "  [GRID-BOUNDARY]" if f.at_boundary else ""
        print(f"  -> c*={f.c_star} (theory=1); gap {f.gap_pct:+.1f}% ; {f.boot}{boundary_note}\n")
