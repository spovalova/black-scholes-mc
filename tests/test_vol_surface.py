import math

import numpy as np

from bscpp.backtest.vol_surface import (
    SVISlice,
    fit_svi_slice,
    svi_butterfly_arbitrage_check,
    svi_fit_rmse,
    svi_gatheral_jacquier_check,
    svi_min_total_variance,
)


def test_svi_recovers_a_known_smile():
    # Generate strikes/ivs from a *known* SVI slice, refit, and check we
    # recover it (up to fitting noise) -- the cleanest correctness check
    # for a nonlinear least-squares fit.
    spot, t_years, rate = 100.0, 0.5, 0.03
    forward = spot * np.exp(rate * t_years)
    true_params = dict(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15)

    strikes = np.linspace(70, 140, 25)
    k = np.log(strikes / forward)
    w = true_params["a"] + true_params["b"] * (
        true_params["rho"] * (k - true_params["m"])
        + np.sqrt((k - true_params["m"]) ** 2 + true_params["sigma"] ** 2)
    )
    ivs = np.sqrt(w / t_years)

    fitted = fit_svi_slice(strikes, ivs, spot=spot, t_years=t_years, rate=rate)
    rmse = svi_fit_rmse(fitted, strikes, ivs, spot=spot, rate=rate)

    assert rmse < 1e-4  # should recover a noiseless smile almost exactly


def test_svi_fit_handles_sparse_smile_gracefully():
    spot, t_years = 100.0, 0.25
    strikes = np.array([90, 95, 100, 105, 110, 115])
    ivs = np.array([0.24, 0.21, 0.20, 0.205, 0.22, 0.245])
    fitted = fit_svi_slice(strikes, ivs, spot=spot, t_years=t_years)
    rmse = svi_fit_rmse(fitted, strikes, ivs, spot=spot)
    assert rmse < 0.02


def test_well_behaved_svi_slice_is_arbitrage_free():
    spot, rate, t_years = 100.0, 0.03, 0.5
    svi = SVISlice(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15, t=t_years)

    assert svi_min_total_variance(svi) > 0
    result = svi_butterfly_arbitrage_check(svi, spot=spot, rate=rate)
    assert result["arbitrage_free"]
    assert result["min_density"] > -1e-6


def test_pathological_svi_slice_is_flagged_as_arbitrage_violating():
    # extreme rho + tiny sigma -> an unrealistically kinked smile that
    # should fail the Breeden-Litzenberger density-positivity check.
    spot, rate, t_years = 100.0, 0.03, 0.5
    svi = SVISlice(a=0.01, b=3.0, rho=-0.95, m=0.0, sigma=0.02, t=t_years)

    result = svi_butterfly_arbitrage_check(svi, spot=spot, rate=rate)
    assert not result["arbitrage_free"]
    assert result["min_density"] < 0


def test_closed_form_g_agrees_with_numerical_check_on_both_slices():
    # The closed-form Gatheral-Jacquier g(k) check and the numerical
    # Breeden-Litzenberger check are two independent routes to the same
    # no-butterfly-arbitrage condition -- they should agree directionally.
    spot, rate, t_years = 100.0, 0.03, 0.5
    good = SVISlice(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.15, t=t_years)
    bad = SVISlice(a=0.01, b=3.0, rho=-0.95, m=0.0, sigma=0.02, t=t_years)

    good_closed = svi_gatheral_jacquier_check(good)
    good_numeric = svi_butterfly_arbitrage_check(good, spot=spot, rate=rate)
    assert good_closed["arbitrage_free"]
    assert good_numeric["arbitrage_free"]

    bad_closed = svi_gatheral_jacquier_check(bad)
    bad_numeric = svi_butterfly_arbitrage_check(bad, spot=spot, rate=rate)
    assert not bad_closed["arbitrage_free"]
    assert not bad_numeric["arbitrage_free"]


def test_closed_form_g_resolves_short_dated_noise_floor():
    # A 1-day-maturity slice pushes the numerical density check's
    # finite-difference noise close to its fixed tolerance (observed as low
    # as ~1e-13 in stress testing). The closed-form check has no such
    # noise floor and should give an unambiguous, comfortably positive
    # verdict on a slice that is genuinely arbitrage-free.
    stress = SVISlice(a=0.001, b=0.1, rho=-0.3, m=0.0, sigma=0.1, t=1 / 365)
    result = svi_gatheral_jacquier_check(stress)
    assert result["arbitrage_free"]
    assert result["min_g"] > 1e-3  # unambiguous, not noise-floor-close to 0


def test_negative_total_variance_fails_both_checks():
    # Regression test for a real false-negative hole found in external
    # review: a slice with NEGATIVE total variance somewhere (min w = -0.048
    # here -- negative implied variance, an outright arbitrage) previously
    # PASSED svi_gatheral_jacquier_check, because g(k) >= 0 is only
    # meaningful conditional on w(k) > 0 and no precondition was enforced.
    # Both checks must now reject such a slice outright, with a reason.
    bad = SVISlice(a=-0.05, b=0.02, rho=0.0, m=0.0, sigma=0.1, t=0.5)
    assert svi_min_total_variance(bad) < 0.0  # confirms the pathology

    closed = svi_gatheral_jacquier_check(bad)
    numeric = svi_butterfly_arbitrage_check(bad, spot=100.0, rate=0.0)

    assert not closed["arbitrage_free"]
    assert closed["reason"] == "negative_total_variance"
    assert math.isnan(closed["min_g"])  # g values on w<0 are garbage: not reported

    assert not numeric["arbitrage_free"]
    assert numeric["reason"] == "negative_total_variance"
