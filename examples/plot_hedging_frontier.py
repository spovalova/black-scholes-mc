"""Renders the headline hedging-policy frontier plot: normalized objective
J(c) vs. band multiplier c, one panel per risk-aversion regime, one line
per arm (real market data, GBM with true vol, GBM with estimated vol),
shaded 95% bootstrap CI, theory (c=1) marked, each arm's empirical optimum
starred.

Loads the grid CSVs saved by hedging_policy_frontier_study.py and
gbm_control_experiment.py -- run those first. This script does no new
hedge simulation itself, so it's fast to iterate on layout/styling without
re-fetching market data or re-running thousands of simulated hedges.

    python examples/hedging_policy_frontier_study.py   # writes output/real_data_grid.csv
    python examples/gbm_control_experiment.py           # writes output/gbm_*_grid.csv
    python examples/plot_hedging_frontier.py             # writes assets/frontier.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from bscpp.backtest.frontier_plot import plot_frontier

OUTPUT_DIR = Path(__file__).parent / "output"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

MULTIPLIERS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0]
RISK_AVERSIONS = [0.03, 0.3, 3.0]
WINDOW_DAYS = 45
STRIDE_DAYS = 20


def main():
    paths = {
        "real_data": OUTPUT_DIR / "real_data_grid.csv",
        "gbm_true_vol": OUTPUT_DIR / "gbm_true_vol_grid.csv",
        "gbm_estimated_vol": OUTPUT_DIR / "gbm_estimated_vol_grid.csv",
    }
    available = {name: p for name, p in paths.items() if p.exists()}
    missing = [name for name in paths if name not in available]
    if not available:
        raise SystemExit(
            "No grid CSVs found. Run hedging_policy_frontier_study.py and "
            "gbm_control_experiment.py first (see this script's docstring)."
        )
    if missing:
        print(f"Warning: missing grid CSV(s) for {missing} -- plotting only {list(available)}.")

    grids = {name: pd.read_csv(p) for name, p in available.items()}
    # Real-data windows overlap in time (stride < window), inducing serial
    # dependence the bootstrap must respect; GBM arms are independent
    # simulated paths with no such dependence -- see each study script.
    block_lens = {
        "real_data": max(int(np.ceil(WINDOW_DAYS / STRIDE_DAYS)), 2),
        "gbm_true_vol": 1.0,
        "gbm_estimated_vol": 1.0,
    }

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_DIR / "frontier.png"
    plot_frontier(grids, MULTIPLIERS, RISK_AVERSIONS, block_lens, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
