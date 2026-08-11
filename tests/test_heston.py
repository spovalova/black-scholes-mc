import math

import numpy as np

import bscpp
from bscpp.backtest import calibrate_heston, heston_fit_rmse


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
