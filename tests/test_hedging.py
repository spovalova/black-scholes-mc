import datetime as dt

import numpy as np
import pandas as pd

from bscpp.backtest.hedging import HedgingBacktester, realized_vs_implied_experiment


def test_hedging_pnl_zero_at_inception():
    dates = pd.date_range(dt.date.today(), periods=30, freq="D")
    prices = pd.Series(100.0 + np.zeros(len(dates)), index=dates)
    backtester = HedgingBacktester(rate=0.05)
    result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=0.2,
                             option_type="call")
    assert abs(result["portfolio_value"].iloc[0]) < 1e-9


def test_hedging_pnl_flat_path_is_near_zero_net_of_rates():
    # A perfectly flat underlying: no gamma P&L to speak of, so hedging P&L
    # should stay small (the position is essentially theta-neutral once
    # financing is accounted for, since hedge_vol matches "realized" (0) only
    # loosely here -- we just check it doesn't blow up).
    dates = pd.date_range(dt.date.today(), periods=30, freq="D")
    prices = pd.Series(100.0, index=dates)
    backtester = HedgingBacktester(rate=0.05)
    result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=0.2,
                             option_type="call")
    assert abs(result["portfolio_value"].iloc[-1]) < 5.0


def test_realized_vs_implied_pnl_sign_matches_theory():
    # Selling/hedging at hedge_vol while the world realizes a *lower* vol
    # should be profitable for the seller; a *higher* realized vol should
    # lose money. This is the central intuition behind vol risk premium
    # harvesting, and it should show up cleanly with enough paths.
    df = realized_vs_implied_experiment(
        hedge_vol=0.30, realized_vols=[0.15, 0.45], spot=100, strike=100, rate=0.05,
        t_days=45, n_paths_per_vol=250, seed=42,
    )
    low = df[df["realized_vol"] == 0.15]["mean_hedging_pnl"].iloc[0]
    high = df[df["realized_vol"] == 0.45]["mean_hedging_pnl"].iloc[0]
    assert low > 0
    assert high < 0
    assert low > high
