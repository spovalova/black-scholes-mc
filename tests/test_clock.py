import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from bscpp.clock import Clock


def test_default_convention_is_act_365():
    assert Clock().convention == "ACT/365"
    assert Clock().days_per_year == 365.0


def test_year_fraction_act_365():
    clock = Clock("ACT/365")
    start = dt.date(2026, 1, 1)
    end = dt.date(2026, 7, 1)  # 181 days
    assert math.isclose(clock.year_fraction(start, end), 181 / 365.0)


def test_year_fraction_trading_252():
    clock = Clock("TRADING/252")
    start = dt.date(2026, 1, 1)
    end = dt.date(2027, 1, 1)  # 365 calendar days
    # 365 * 5/7 ~= 260.7 trading days -> /252
    assert math.isclose(clock.year_fraction(start, end), 365 * (5.0 / 7.0) / 252.0)


def test_time_to_expiry_clamped_at_zero():
    clock = Clock()
    past = dt.date(2026, 1, 1)
    future = dt.date(2026, 1, 10)
    assert clock.time_to_expiry(future, past) == 0.0  # expiration before as_of -> clamped
    assert clock.time_to_expiry(past, future) > 0.0


def test_time_to_expiry_floor_at_one_day():
    clock = Clock()
    same_day = dt.date(2026, 1, 1)
    assert clock.time_to_expiry(same_day, same_day) == 0.0
    assert clock.time_to_expiry(same_day, same_day, floor_at_one_day=True) == 1.0 / 365.0


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        Clock("ACT/360")


def test_annualized_realized_vol_matches_known_gbm_scale():
    # Simulate GBM at a known vol and confirm the estimator recovers it
    # within normal sampling noise over many trading days.
    rng = np.random.default_rng(0)
    true_vol = 0.30
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    log_rets = true_vol * math.sqrt(1 / 365.0) * rng.normal(size=n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)]))
    closes = pd.Series(prices, index=dates)

    estimated = Clock("ACT/365").annualized_realized_vol(closes)
    assert 0.20 < estimated < 0.45  # loose band; this is a noisy single-path estimate


def test_annualized_realized_vol_nan_on_insufficient_data():
    clock = Clock()
    assert clock.annualized_realized_vol(pd.Series([100.0], index=[pd.Timestamp("2026-01-01")])) != \
        clock.annualized_realized_vol(pd.Series([100.0], index=[pd.Timestamp("2026-01-01")]))  # NaN != NaN


def test_annualized_realized_vol_matches_original_inline_formula_exactly():
    # Regression guard: this Clock method REPLACES an inline formula that
    # was duplicated in 3 example scripts (hedging_policy_frontier_study.py,
    # gbm_control_experiment.py, real_data_validation_study.py), whose
    # results are already committed/published (see README's "Research
    # finding"). Must match the old formula bit-for-bit under ACT/365 --
    # any drift here would silently invalidate already-published numbers.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-03-01", periods=60)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=60))), index=dates)

    def old_formula(closes):
        closes = closes.dropna()
        log_returns = np.log(closes / closes.shift(1)).dropna()
        if len(log_returns) < 2:
            return float("nan")
        elapsed_years = (closes.index[-1] - closes.index[0]).days / 365.0
        if elapsed_years <= 0:
            return float("nan")
        return float(np.sqrt(np.sum(log_returns.to_numpy() ** 2) / elapsed_years))

    assert Clock("ACT/365").annualized_realized_vol(prices) == old_formula(prices)
