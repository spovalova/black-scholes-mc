import numpy as np

from bscpp.backtest.vol_surface import (
    SVISlice,
    fit_svi_slice,
    svi_butterfly_arbitrage_check,
    svi_fit_rmse,
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
