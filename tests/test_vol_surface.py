import math

import numpy as np

from bscpp.backtest.vol_surface import (
    SVISlice,
    _svi_conditional_linear_fit,
    fit_svi_slice,
    fit_svi_slice_quasi_explicit,
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


def test_svi_conditional_linear_fit_matches_the_svi_formula_when_feasible():
    # Unit-level check of the reparametrization fit_svi_slice_quasi_explicit
    # is built on: for a KNOWN (a,b,rho,m,sigma), y=(k-m)/sigma and
    # c1=a, c2=b*rho*sigma, c3=b*sigma should make
    # c1 + c2*y + c3*sqrt(y^2+1) reproduce the exact same total variance
    # SVISlice.total_variance computes -- confirming the algebra, not just
    # the end-to-end fit (which could mask a compensating pair of bugs).
    a, b, rho, m, sigma = 0.02, 0.15, -0.4, 0.05, 0.12
    svi = SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma, t=0.5)

    k = np.linspace(-0.5, 0.5, 20)
    w_true = svi.total_variance(k)

    y = (k - m) / sigma
    weights = np.ones_like(y)
    c1, c2, c3, sse = _svi_conditional_linear_fit(y, w_true, weights)

    assert sse < 1e-16  # exact algebraic recovery, not a fit -- should be ~machine precision
    assert math.isclose(c1, a, abs_tol=1e-8)
    assert math.isclose(c2, b * rho * sigma, abs_tol=1e-8)
    assert math.isclose(c3, b * sigma, abs_tol=1e-8)


def test_svi_quasi_explicit_recovers_a_known_smile():
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

    fitted = fit_svi_slice_quasi_explicit(strikes, ivs, spot=spot, t_years=t_years, rate=rate)
    rmse = svi_fit_rmse(fitted, strikes, ivs, spot=spot, rate=rate)

    assert rmse < 1e-4  # same bar test_svi_recovers_a_known_smile holds fit_svi_slice to


def test_svi_quasi_explicit_robust_to_bad_naive_initial_guesses():
    # The actual point of the quasi-explicit method, not a nice-to-have:
    # fit_svi_slice's full 5D nonlinear search can land in a bad local
    # optimum depending on its starting point -- verified here with two
    # initial guesses that measurably degrade it (RMSE ~0.19-0.22, vs.
    # ~0.0015 for a good fit) on a short-dated, strongly-skewed, noisy
    # smile (the harder case where this actually bites -- the smooth,
    # long-dated smile in test_svi_quasi_explicit_recovers_a_known_smile
    # doesn't expose it). fit_svi_slice_quasi_explicit needs no initial
    # guess at all and should match the GOOD fits regardless.
    spot, t_years, rate = 100.0, 0.083, 0.03
    forward = spot * np.exp(rate * t_years)
    true_params = dict(a=0.003, b=0.4, rho=-0.85, m=-0.05, sigma=0.04)

    strikes = np.linspace(85, 115, 15)
    k = np.log(strikes / forward)
    w = true_params["a"] + true_params["b"] * (
        true_params["rho"] * (k - true_params["m"])
        + np.sqrt((k - true_params["m"]) ** 2 + true_params["sigma"] ** 2)
    )
    ivs_clean = np.sqrt(np.maximum(w, 1e-8) / t_years)
    rng = np.random.default_rng(42)
    ivs = ivs_clean + rng.normal(0, 0.002, size=ivs_clean.shape)

    bad_guesses = [
        [0.001, 0.001, 0.99, -3.0, 0.001],
        [1e-4, 2.0, 0.95, -2.5, 2.0],
    ]
    naive_rmses = []
    for guess in bad_guesses:
        fitted = fit_svi_slice(strikes, ivs, spot=spot, t_years=t_years, rate=rate,
                                initial_guess=guess)
        naive_rmses.append(svi_fit_rmse(fitted, strikes, ivs, spot=spot, rate=rate))

    # Confirms the setup: these guesses really do break the naive fit --
    # this test would be vacuous (asserting robustness against a problem
    # that doesn't exist) without this check.
    assert max(naive_rmses) > 0.05

    fitted_qe = fit_svi_slice_quasi_explicit(strikes, ivs, spot=spot, t_years=t_years, rate=rate)
    rmse_qe = svi_fit_rmse(fitted_qe, strikes, ivs, spot=spot, rate=rate)
    assert rmse_qe < 0.005


def test_svi_quasi_explicit_vega_weighting_changes_the_fit():
    # vega_weighted=True should give a genuinely different fit than
    # vega_weighted=False -- but only where there's real tension for the
    # weighting to resolve: on a NOISELESS smile that exactly matches the
    # SVI functional form, every point is consistent with every other one
    # regardless of weighting, so both converge to the identical (true)
    # answer and the test would be vacuous. Noise breaks that -- with
    # residuals that can't all be driven to zero at once, which points
    # get to "win" the fit genuinely depends on how they're weighted.
    spot, t_years, rate = 100.0, 0.25, 0.03
    forward = spot * np.exp(rate * t_years)
    true_params = dict(a=0.015, b=0.3, rho=-0.6, m=-0.1, sigma=0.08)
    strikes = np.linspace(50, 200, 30)
    k = np.log(strikes / forward)
    w = true_params["a"] + true_params["b"] * (
        true_params["rho"] * (k - true_params["m"])
        + np.sqrt((k - true_params["m"]) ** 2 + true_params["sigma"] ** 2)
    )
    ivs_clean = np.sqrt(np.maximum(w, 1e-8) / t_years)
    rng = np.random.default_rng(3)
    ivs = ivs_clean + rng.normal(0, 0.01, size=ivs_clean.shape)

    weighted = fit_svi_slice_quasi_explicit(strikes, ivs, spot=spot, t_years=t_years, rate=rate,
                                             vega_weighted=True)
    unweighted = fit_svi_slice_quasi_explicit(strikes, ivs, spot=spot, t_years=t_years, rate=rate,
                                               vega_weighted=False)

    params_differ = any(
        not math.isclose(getattr(weighted, f), getattr(unweighted, f), abs_tol=1e-6, rel_tol=1e-4)
        for f in ("a", "b", "rho", "m", "sigma")
    )
    assert params_differ


def test_svi_quasi_explicit_fit_is_arbitrage_free_on_well_behaved_data():
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

    fitted = fit_svi_slice_quasi_explicit(strikes, ivs, spot=spot, t_years=t_years, rate=rate)
    assert svi_gatheral_jacquier_check(fitted)["arbitrage_free"]
