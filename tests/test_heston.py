import math

import numpy as np

import datetime as dt

import bscpp
from bscpp.backtest import (
    MockProvider,
    StripPricer,
    calibrate_heston,
    calibrate_heston_with_stability,
    heston_fit_rmse,
)


def test_heston_collapses_to_black_scholes_as_vol_of_vol_shrinks():
    # As xi -> 0 with v0 == theta, the variance process becomes deterministic
    # (constant v0), so Heston degenerates to GBM at vol = sqrt(v0) and must
    # match Black-Scholes exactly in that limit. A sign/branch-cut bug in the
    # characteristic function would NOT produce this clean, monotonic
    # convergence -- it's the sharpest available check on the formula itself.
    spot, strike, rate, div, maturity = 100.0, 100.0, 0.05, 0.0, 1.0
    v0 = 0.04
    bs = bscpp.price(spot, strike, rate, math.sqrt(v0), maturity, "call", div)

    diffs = []
    for xi in (0.1, 0.01, 0.001, 1e-4):
        hp = bscpp.HestonParams(kappa=2.0, theta=v0, xi=xi, rho=-0.5, v0=v0)
        heston = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Call, hp)
        diffs.append(abs(heston - bs))

    # each halving-ish of xi should shrink the gap to BS, and the smallest
    # xi should be very close indeed
    assert diffs == sorted(diffs, reverse=True)
    assert diffs[-1] < 1e-3


def test_heston_analytic_accurate_in_extreme_feller_violating_regime():
    # xi=3.0 against kappa=2.0, theta=0.04 badly violates the Feller
    # condition (2*kappa*theta=0.16 vs xi^2=9) -- exactly the extreme
    # regime the adaptive-quadrature rewrite targets (the old fixed
    # phi_max=200/4000-point Simpson rule was never verified here).
    #
    # A naive MC comparison at a modest step count is MISLEADING: the
    # full-truncation Euler scheme has its own well-known discretization
    # bias that's large precisely when Feller is badly violated (variance
    # keeps hitting its floor). At 300 steps MC disagrees with the
    # analytic price by ~40 std errors; by 3000 steps it converges to
    # within <1 std error of the SAME analytic price -- confirming the
    # analytic pricer, not the coarse MC, was right. This test uses
    # enough steps to make that comparison fair.
    spot, strike, rate, div, maturity = 100.0, 100.0, 0.05, 0.0, 1.0
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)

    analytic = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Call, hp)
    mc = bscpp.HestonMCPricer(seed=1).price(spot, strike, rate, div, maturity,
                                             bscpp.OptionType.Call, hp, 150_000, 3000)

    assert abs(analytic - mc.price) < 5 * mc.std_error


def test_heston_analytic_stable_at_very_short_maturity():
    # 1-day maturity: the characteristic function's integrand decays more
    # slowly in phi at short T, which is exactly the regime the old fixed
    # phi_max=200 truncation was flagged as untested against.
    spot, strike, rate, div = 100.0, 100.0, 0.05, 0.0
    v0 = 0.04
    hp = bscpp.HestonParams(kappa=2.0, theta=v0, xi=0.4, rho=-0.7, v0=v0)

    price = bscpp.heston_price(spot, strike, rate, div, 1 / 365, bscpp.OptionType.Call, hp)
    bs_reference = bscpp.price(spot, strike, rate, math.sqrt(v0), 1 / 365, "call", div)
    assert math.isclose(price, bs_reference, abs_tol=5e-3)


def test_heston_matches_independent_monte_carlo():
    spot, strike, rate, div, maturity = 100.0, 100.0, 0.05, 0.0, 1.0
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)

    analytic = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Call, hp)
    mc = bscpp.HestonMCPricer(seed=1).price(spot, strike, rate, div, maturity,
                                             bscpp.OptionType.Call, hp, 200_000, 200)

    assert abs(analytic - mc.price) < 4 * mc.std_error + 0.02


def test_heston_matches_mc_across_strikes_and_types():
    spot, rate, div, maturity = 100.0, 0.05, 0.0, 1.0
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)

    for strike, otype in [(80, bscpp.OptionType.Call), (100, bscpp.OptionType.Put),
                           (120, bscpp.OptionType.Call)]:
        analytic = bscpp.heston_price(spot, strike, rate, div, maturity, otype, hp)
        mc = bscpp.HestonMCPricer(seed=3).price(spot, strike, rate, div, maturity, otype, hp,
                                                 150_000, 150)
        assert abs(analytic - mc.price) < 4 * mc.std_error + 0.05


def test_heston_put_call_parity():
    spot, strike, rate, div, maturity = 100.0, 105.0, 0.03, 0.01, 0.75
    hp = bscpp.HestonParams(kappa=1.5, theta=0.05, xi=0.3, rho=-0.6, v0=0.06)

    call = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Call, hp)
    put = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Put, hp)

    lhs = call - put
    rhs = spot * math.exp(-div * maturity) - strike * math.exp(-rate * maturity)
    assert math.isclose(lhs, rhs, abs_tol=1e-9)


def test_feller_condition():
    satisfied = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.2, rho=-0.5, v0=0.04)
    violated = bscpp.HestonParams(kappa=1.0, theta=0.02, xi=2.0, rho=-0.5, v0=0.02)
    assert bscpp.heston_satisfies_feller_condition(satisfied)
    assert not bscpp.heston_satisfies_feller_condition(violated)


def test_heston_calibration_recovers_a_known_smile():
    # Heston calibration has a known parameter-identifiability issue
    # (different parameter vectors can produce near-identical smiles), so
    # the honest bar is fit quality on a noiseless synthetic smile, not
    # exact parameter recovery.
    spot, rate, div, t_years = 100.0, 0.05, 0.0, 0.5
    true_params = bscpp.HestonParams(kappa=1.8, theta=0.045, xi=0.55, rho=-0.65, v0=0.05)

    strikes = np.linspace(75, 130, 16)
    option_types = ["put" if k < spot else "call" for k in strikes]
    otypes_enum = [bscpp.OptionType.Call if t == "call" else bscpp.OptionType.Put
                   for t in option_types]

    true_prices = [bscpp.heston_price(spot, float(k), rate, div, t_years, ot, true_params)
                   for k, ot in zip(strikes, otypes_enum)]
    seed_inputs = [bscpp.make_inputs(spot, float(k), rate, 0.2, t_years, t, div)
                   for k, t in zip(strikes, option_types)]
    market_ivs = bscpp.bs_implied_vol_batch(seed_inputs, true_prices)

    fitted = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate, div)
    rmse = heston_fit_rmse(fitted, strikes, option_types, market_ivs, spot, t_years, rate, div)

    assert rmse < 1e-3


def _short_dated_mock_chain():
    spot, rate = 450.0, 0.05
    provider = MockProvider(spot=spot, base_vol=0.18, smile_strength=0.40)
    pricer = StripPricer(provider, rate=rate, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=45)
    chain = pricer.price_strip("SPY", expiration, strike_range=(0.85, 1.15), use_mc=False)
    calls = chain[chain["type"] == "call"]
    t_years = float(chain["T"].iloc[0])
    return calls["strike"].to_numpy(), calls["type"].tolist(), calls["model_iv"].to_numpy(), spot, t_years, rate


def test_regularization_pulls_v0_out_of_degenerate_corner():
    # On a short-dated, mildly-curved smile, the unregularized fit is known
    # to drive v0 toward its lower bound (a real, observed failure mode --
    # see heston_calibration_demo.py). Regularization should pull it back
    # toward the ATM-variance-scale region without materially hurting fit
    # quality.
    strikes, option_types, market_ivs, spot, t_years, rate = _short_dated_mock_chain()
    atm_var = float(np.median(market_ivs)) ** 2

    unregularized = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate,
                                      regularization_weight=0.0)
    regularized = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate)

    assert unregularized.v0 < 0.2 * atm_var  # confirms the degenerate baseline actually occurs
    assert regularized.v0 > 0.5 * atm_var  # regularized fit lands in a sane region

    rmse_unreg = heston_fit_rmse(unregularized, strikes, option_types, market_ivs, spot, t_years,
                                  rate)
    rmse_reg = heston_fit_rmse(regularized, strikes, option_types, market_ivs, spot, t_years, rate)
    assert rmse_reg < rmse_unreg + 0.01  # doesn't meaningfully hurt fit quality


def test_stability_diagnostic_distinguishes_regularized_from_unregularized():
    strikes, option_types, market_ivs, spot, t_years, rate = _short_dated_mock_chain()

    stable = calibrate_heston_with_stability(strikes, option_types, market_ivs, spot, t_years,
                                              rate, n_starts=5, seed=1)
    unstable = calibrate_heston_with_stability(strikes, option_types, market_ivs, spot, t_years,
                                                rate, n_starts=5, seed=1, regularization_weight=0.0)

    assert stable["params_stable"]
    assert not unstable["params_stable"]
    # both should reach comparably good fits despite the parameter instability --
    # this is exactly the "low RMSE doesn't mean well-identified" lesson
    assert max(stable["all_rmse"]) < 0.02
    assert max(unstable["all_rmse"]) < 0.02
