import datetime as dt
import math

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


def test_pnl_attribution_sums_to_realized_and_is_a_good_approximation():
    rng = np.random.default_rng(11)
    dates = pd.date_range(dt.date.today(), periods=60, freq="D")
    vol = 0.35
    dt_frac = 1 / 365
    rets = rng.normal((0.05 - 0.5 * vol**2) * dt_frac, vol * np.sqrt(dt_frac), size=59)
    path = 100 * np.exp(np.concatenate([[0], np.cumsum(rets)]))
    prices = pd.Series(path, index=dates)

    backtester = HedgingBacktester(rate=0.05)
    result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=vol,
                             option_type="call")
    attributed = backtester.attribute_pnl(result)

    # the decomposition is an exact accounting identity by construction:
    # sum of daily realized P&L must equal the final cumulative P&L.
    assert math.isclose(attributed["realized_pnl"].sum(), result["portfolio_value"].iloc[-1],
                         abs_tol=1e-6)

    # the Taylor-expansion approximation (financing + gamma + theta) should
    # explain most of the realized P&L, with a modest higher-order residual.
    total_abs_realized = attributed["realized_pnl"].abs().sum()
    total_abs_error = attributed["attribution_error"].abs().sum()
    assert total_abs_error < 0.5 * total_abs_realized


def test_dividend_yield_does_not_bias_hedging_pnl():
    # With hedge_vol == the path's own realized vol, hedging P&L should stay
    # near zero regardless of dividend_yield -- the stock leg must collect
    # its dividend income, or the hedge is inconsistent with using
    # dividend-adjusted Black-Scholes deltas and picks up a spurious bias.
    rng = np.random.default_rng(5)
    dates = pd.date_range(dt.date.today(), periods=90, freq="D")
    vol, rate, q = 0.25, 0.05, 0.03
    dt_frac = 1 / 365
    rets = rng.normal((rate - q - 0.5 * vol**2) * dt_frac, vol * np.sqrt(dt_frac), size=89)
    path = 100 * np.exp(np.concatenate([[0], np.cumsum(rets)]))
    prices = pd.Series(path, index=dates)

    backtester = HedgingBacktester(rate=rate, dividend_yield=q)
    result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=vol,
                             option_type="call")
    assert abs(result["portfolio_value"].iloc[-1]) < 5.0


def test_pnl_attribution_matches_carr_madan_closed_form():
    # Carr & Madan (2002), "Towards a Theory of Volatility Trading": the
    # P&L of delta-hedging at hedge_vol against a path with realized_vol
    # should satisfy, in the continuous-time limit,
    #   financing + gamma_pnl + theta_pnl ~= 0.5*Gamma*S^2*(hedge_vol^2 - realized_vol^2)*dt
    # (this is the fully-combined identity -- gamma_pnl + theta_pnl ALONE do
    # not equal this; the r*(S*delta - V) term embedded in theta is exactly
    # what financing_pnl cancels).
    #
    # Uses a deterministic alternating-sign path with an EXACT per-step
    # log-return magnitude (c = realized_vol*sqrt(dt)) rather than a random
    # GBM draw, so dS_i^2 ~= S_prev^2*realized_vol^2*dt very precisely and
    # the comparison isn't muddied by single-path Monte Carlo sampling noise.
    n_days, realized_vol, hedge_vol, rate = 120, 0.20, 0.35, 0.05
    dt_frac = 1 / 365
    c = realized_vol * math.sqrt(dt_frac)
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n_days - 1)])
    path = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(signs * c)]))
    dates = pd.date_range(dt.date.today(), periods=n_days, freq="D")
    prices = pd.Series(path, index=dates)

    backtester = HedgingBacktester(rate=rate)
    result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=hedge_vol,
                             option_type="call")
    attributed = backtester.attribute_pnl(result)

    predicted_total = attributed[["financing_pnl", "gamma_pnl", "theta_pnl"]].to_numpy().sum()

    gamma_prev = attributed["gamma"].shift(1).to_numpy()[1:]
    spot_prev = attributed["spot"].shift(1).to_numpy()[1:]
    closed_form_total = 0.5 * np.sum(
        gamma_prev * spot_prev ** 2 * (hedge_vol ** 2 - realized_vol ** 2) * dt_frac
    )

    assert math.isclose(predicted_total, closed_form_total, rel_tol=0.05)


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
