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


def test_transaction_costs_reduce_pnl_and_scale_with_bps():
    rng = np.random.default_rng(11)
    dates = pd.date_range(dt.date.today(), periods=60, freq="D")
    vol = 0.35
    dt_frac = 1 / 365
    rets = rng.normal((0.05 - 0.5 * vol**2) * dt_frac, vol * np.sqrt(dt_frac), size=59)
    path = 100 * np.exp(np.concatenate([[0], np.cumsum(rets)]))
    prices = pd.Series(path, index=dates)

    pnls, costs = [], []
    for bps in (0, 5, 20):
        backtester = HedgingBacktester(rate=0.05, transaction_cost_bps=bps)
        result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=vol,
                                 option_type="call")
        costs.append(result["transaction_cost"].sum())
        pnls.append(result["portfolio_value"].iloc[-1])

    assert costs[0] == 0.0
    assert costs[1] < costs[2]  # more bps -> more cost
    # cost should scale ~linearly with bps (same trade sizes, just pricier)
    assert math.isclose(costs[2] / costs[1], 4.0, rel_tol=0.05)
    assert pnls[0] > pnls[1] > pnls[2]  # more cost -> strictly worse P&L


def test_transaction_cost_attribution_is_exact_not_approximate():
    # Unlike gamma_pnl/theta_pnl (Taylor approximations), transaction costs
    # are a direct, exactly-known cash flow -- adding them to
    # predicted_pnl should not change the attribution_error residual AT
    # ALL, regardless of the cost level.
    rng = np.random.default_rng(11)
    dates = pd.date_range(dt.date.today(), periods=60, freq="D")
    vol = 0.35
    dt_frac = 1 / 365
    rets = rng.normal((0.05 - 0.5 * vol**2) * dt_frac, vol * np.sqrt(dt_frac), size=59)
    path = 100 * np.exp(np.concatenate([[0], np.cumsum(rets)]))
    prices = pd.Series(path, index=dates)

    errors = []
    for bps in (0, 5, 20):
        backtester = HedgingBacktester(rate=0.05, transaction_cost_bps=bps)
        result = backtester.run(prices, strike=100, expiration=dates[-1].date(), hedge_vol=vol,
                                 option_type="call")
        attributed = backtester.attribute_pnl(result)
        errors.append(attributed["attribution_error"].abs().sum())

    assert math.isclose(errors[0], errors[1], abs_tol=1e-9)
    assert math.isclose(errors[0], errors[2], abs_tol=1e-9)


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


def test_realized_vs_implied_experiment_transaction_costs_reduce_every_pnl():
    frictionless = realized_vs_implied_experiment(
        hedge_vol=0.30, realized_vols=[0.15, 0.30, 0.45], spot=100, strike=100, rate=0.05,
        t_days=45, n_paths_per_vol=250, seed=42, transaction_cost_bps=0,
    )
    with_cost = realized_vs_implied_experiment(
        hedge_vol=0.30, realized_vols=[0.15, 0.30, 0.45], spot=100, strike=100, rate=0.05,
        t_days=45, n_paths_per_vol=250, seed=42, transaction_cost_bps=10,
    )
    shift = frictionless["mean_hedging_pnl"].to_numpy() - with_cost["mean_hedging_pnl"].to_numpy()
    assert (shift > 0).all()  # costs strictly reduce P&L at every realized-vol level


def test_vega_pnl_zero_with_constant_vol_and_active_with_remarked_vol():
    # Constant hedge_vol: vega_pnl must be identically zero and the
    # attribution must match the historical (pre-vega) behavior. Re-marked
    # (time-varying) hedge_vol: the option's day-over-day re-marking P&L
    # must be captured by an explicit vega term, not dumped into the
    # residual -- on a real book re-marking is often the DOMINANT term.
    rng = np.random.default_rng(21)
    dates = pd.date_range(dt.date.today(), periods=40, freq="D")
    vol = 0.30
    dt_frac = 1 / 365
    rets = rng.normal((0.05 - 0.5 * vol**2) * dt_frac, vol * np.sqrt(dt_frac), size=39)
    prices = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(rets)])), index=dates)

    bt = HedgingBacktester(rate=0.05)

    flat = bt.attribute_pnl(bt.run(prices, 100, dates[-1].date(), 0.30))
    assert float(flat["vega_pnl"].abs().sum()) == 0.0

    # a vol path that drifts up then down around 0.30
    vol_path = pd.Series(0.30 + 0.06 * np.sin(np.linspace(0, 3.0, len(dates))), index=dates)
    marked = bt.run(prices, 100, dates[-1].date(), vol_path)
    attr = bt.attribute_pnl(marked)

    assert attr["vega_pnl"].abs().sum() > 0.1  # genuinely active
    # accounting identity is untouched
    assert math.isclose(attr["realized_pnl"].sum(), marked["portfolio_value"].iloc[-1],
                         abs_tol=1e-9)
    # with the vega term INCLUDED the decomposition must still explain most
    # of realized P&L; with it EXCLUDED the residual would blow up -- check
    # both directions to prove the term is doing real work.
    err_with = attr["attribution_error"].abs().sum()
    err_without = (attr["realized_pnl"] - (attr["predicted_pnl"] - attr["vega_pnl"])).abs().sum()
    assert err_with < 0.5 * attr["realized_pnl"].abs().sum()
    assert err_without > 2.0 * err_with


def test_hedge_vol_series_must_cover_all_dates():
    dates = pd.date_range(dt.date.today(), periods=10, freq="D")
    prices = pd.Series(100.0, index=dates)
    partial_vol = pd.Series(0.2, index=dates[:5])
    bt = HedgingBacktester(rate=0.05)
    try:
        bt.run(prices, 100, dates[-1].date(), partial_vol)
        assert False, "expected ValueError for partial hedge_vol coverage"
    except ValueError:
        pass
