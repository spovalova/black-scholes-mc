"""matplotlib visualization for frontier-sweep results (see
bscpp.backtest.frontier). Kept in a separate module so `import
bscpp.backtest` doesn't require matplotlib as a hard dependency -- only
`pip install bscpp[plots]` / scripts that actually plot need it.
"""

from __future__ import annotations

import numpy as np

from bscpp.stats import cluster_bootstrap_indices


def objective_curve_with_ci(grid, multipliers, lam0, block_len, level=0.95, n_boot=2000, seed=0):
    """Per-c bootstrap CI on the mean normalized objective J(c) for one
    risk-aversion regime -- the whole curve's sampling uncertainty, not
    just the gap between c* and c=1 that FrontierRegime reports.

    Clusters by window_start (calendar period), not individual (ticker,
    window) rows, for the same reason bscpp.backtest.frontier's
    _split_sample_bootstrap does: rows from DIFFERENT tickers sharing the
    same window_start are contemporaneously correlated (same market vol
    regime), which a plain row-level block bootstrap is blind to -- see
    frontier.py's docstring for the full derivation. This function isn't
    testing a selected c* against a baseline, so it doesn't need the
    split-sample half of that fix, only the clustering half.
    """
    sub = grid[grid["lam0"] == lam0].copy()
    sub["objective"] = sub["total_cost"] / sub["premium0"] + \
        lam0 * sub["pnl_variance"] / sub["premium0"] ** 2

    means, los, his = [], [], []
    for c in multipliers:
        cell = sub[sub["c"] == c]
        vals = cell["objective"].to_numpy()
        periods = np.sort(cell["window_start"].unique())
        cluster_row_idx = [np.flatnonzero(cell["window_start"].to_numpy() == p) for p in periods]

        rng = np.random.default_rng(seed)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = cluster_bootstrap_indices(cluster_row_idx, block_len, rng)
            boot_means[b] = vals[idx].mean()
        alpha = (1.0 - level) / 2.0
        lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])

        means.append(float(vals.mean()))
        los.append(float(lo))
        his.append(float(hi))
    return np.array(means), np.array(los), np.array(his)


def plot_frontier(grids: dict, multipliers: list[float], risk_aversions: list[float],
                   block_lens: dict, out_path, title: str | None = None):
    """grids: {arm_label: grid DataFrame} as returned by run_policy_grid
    (already run, not re-run here). block_lens: {arm_label: avg_block_len}
    -- overlap-derived for real rolling-window data, 1.0 for independent
    simulated paths (see each study script for why).

    One subplot per risk-aversion regime; one line + shaded 95% CI band
    per arm; c=1 (the WW theoretical band) marked with a vertical line;
    each arm's empirical optimum c* marked with a star.
    """
    import matplotlib.pyplot as plt

    arms = [a for a in grids if not grids[a].empty]
    fig, axes = plt.subplots(1, len(risk_aversions), figsize=(5.5 * len(risk_aversions), 4.5))
    axes = np.atleast_1d(axes)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, lam0 in zip(axes, risk_aversions):
        for color, arm in zip(colors, arms):
            grid = grids[arm]
            if lam0 not in grid["lam0"].unique():
                continue
            means, los, his = objective_curve_with_ci(grid, multipliers, lam0, block_lens[arm])
            ax.plot(multipliers, means, marker="o", color=color, label=arm)
            ax.fill_between(multipliers, los, his, color=color, alpha=0.15)
            c_star = multipliers[int(np.argmin(means))]
            ax.plot(c_star, means[int(np.argmin(means))], marker="*", color=color,
                    markersize=16, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(multipliers)
        ax.set_xticklabels([f"{c:g}" for c in multipliers])
        ax.set_xlabel("band multiplier c  (theory = 1, dashed)")
        ax.set_ylabel("normalized objective J(c)")
        ax.set_title(f"risk_aversion = {lam0}")

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title or "Cost-risk objective vs. Whalley-Wilmott band multiplier "
                          "(shaded = 95% bootstrap CI, star = empirical optimum)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig
