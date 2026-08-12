import math

import numpy as np

import bscpp


def _inputs(otype="call"):
    return bscpp.make_inputs(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0,
                              option_type=otype, dividend_yield=0.0)


def test_bs_call_put_parity():
    call = bscpp.bs_price(_inputs("call"))
    put = bscpp.bs_price(_inputs("put"))
    # C - P = S - K*exp(-rT)
    lhs = call - put
    rhs = 100 - 100 * math.exp(-0.05 * 1.0)
    assert math.isclose(lhs, rhs, abs_tol=1e-9)


def test_bs_price_matches_known_value():
    # Standard textbook case: S=K=100, r=5%, sigma=20%, T=1y -> call ~= 10.4506
    price = bscpp.bs_price(_inputs("call"))
    assert math.isclose(price, 10.4506, abs_tol=1e-3)


def test_mc_converges_to_bs():
    inputs = _inputs("call")
    bs = bscpp.bs_price(inputs)
    mc = bscpp.MonteCarloPricer(seed=123).price_european(inputs, 200_000, True)
    # allow ~5 std errors of slack
    assert abs(mc.price - bs) < 5 * mc.std_error + 0.05


def test_greeks_sane_signs():
    call_greeks = bscpp.bs_greeks(_inputs("call"))
    put_greeks = bscpp.bs_greeks(_inputs("put"))
    assert 0.0 < call_greeks.delta < 1.0
    assert -1.0 < put_greeks.delta < 0.0
    assert call_greeks.gamma > 0
    assert call_greeks.vega > 0
    assert call_greeks.theta < 0  # long call decays


def test_implied_vol_round_trip():
    inputs = _inputs("put")
    inputs.vol = 0.35
    price = bscpp.bs_price(inputs)
    iv = bscpp.bs_implied_vol(inputs, price, initial_guess=0.2)
    assert math.isclose(iv, 0.35, abs_tol=1e-4)


def test_mc_std_error_scales_as_inverse_sqrt_n():
    # Monte Carlo error should shrink as 1/sqrt(N): 16x the paths should
    # roughly quarter the standard error.
    inputs = _inputs("call")
    small = bscpp.MonteCarloPricer(seed=1).price_european(inputs, 10_000, True)
    large = bscpp.MonteCarloPricer(seed=1).price_european(inputs, 160_000, True)
    ratio = small.std_error / large.std_error
    assert 3.0 < ratio < 5.5  # sqrt(16) = 4, generous slack for sampling noise


def test_mc_greeks_close_to_bs():
    inputs = _inputs("call")
    bs = bscpp.bs_greeks(inputs)
    mc = bscpp.MonteCarloPricer(seed=7).greeks_european(inputs, 200_000, True)
    assert math.isclose(mc.delta, bs.delta, abs_tol=0.03)
    assert math.isclose(mc.vega, bs.vega, abs_tol=1.5)


def test_iv_solver_brent_fallback_handles_extreme_cases_without_nan():
    # Stress test across extreme moneyness/maturity/vol: the Brent's-method
    # fallback (replacing plain bisection) should never return NaN for a
    # genuinely solvable price -- Newton alone used to fail (flat vega)
    # deep ITM/OTM and near expiry, which is exactly the regime this
    # targets. Brent is bracket-guaranteed convergent given [1e-6, 5.0]
    # brackets the root, which it does for any economically meaningful price.
    rng = np.random.default_rng(0)
    n_nan = 0
    well_posed_errs = []
    n_total = 0

    for _ in range(5000):
        strike = rng.uniform(1, 1000)
        rate = rng.uniform(-0.02, 0.10)
        div = rng.uniform(0, 0.05)
        maturity = rng.uniform(1 / 365, 3.0)
        true_vol = rng.uniform(0.01, 3.0)
        otype = "call" if rng.random() < 0.5 else "put"

        inputs = bscpp.make_inputs(100.0, strike, rate, true_vol, maturity, otype, div)
        price = bscpp.bs_price(inputs)
        vega = bscpp.bs_greeks(inputs).vega
        if price <= 1e-12:
            continue
        n_total += 1

        iv = bscpp.bs_implied_vol(inputs, price)
        assert iv == iv, "solver returned NaN for a genuinely solvable price"  # NaN check
        if iv != iv:
            n_nan += 1
            continue

        # Only require accurate recovery where the inverse problem is
        # well-posed (vega meaningfully above zero relative to price scale).
        # Deep ITM + near-expiry contracts have price essentially flat in
        # vol -- no solver, Brent's or otherwise, can recover a generating
        # vol the price itself doesn't distinguish.
        if vega > 1e-3 * max(price, 1.0):
            well_posed_errs.append(abs(iv - true_vol))

    assert n_nan == 0
    well_posed_errs = np.array(well_posed_errs)
    assert len(well_posed_errs) > 0.5 * n_total  # most cases should be well-posed
    assert well_posed_errs.max() < 1e-3  # essentially exact recovery where it's solvable


def test_antithetic_std_error_is_calibrated():
    # Regression test for a real bug: antithetic samples were pooled into
    # the i.i.d. variance formula, overstating std_error by ~32% (antithetic
    # pairs are negatively correlated -- that's the point of the technique).
    # The fixed estimator computes variance over PAIR MEANS. This test checks
    # the reported std_error against the realized dispersion of the estimator
    # across independent seeds -- an error bar must describe the error.
    inputs = _inputs("call")
    estimates = []
    reported = []
    for seed in range(120):
        r = bscpp.MonteCarloPricer(seed=seed).price_european(inputs, 10_000, True)
        estimates.append(r.price)
        reported.append(r.std_error)

    realized_std = float(np.std(estimates, ddof=1))
    mean_reported = float(np.mean(reported))
    ratio = mean_reported / realized_std
    # calibrated error bars: reported within ~15% of realized (sampling
    # noise in realized_std across 120 seeds is a few percent itself)
    assert 0.85 < ratio < 1.15, f"std_error miscalibrated: ratio={ratio:.3f}"


def test_trading_greeks_unit_conversion():
    greeks = bscpp.bs_greeks(_inputs("call"))
    desk = bscpp.trading_greeks(greeks)
    assert desk["delta"] == greeks.delta
    assert desk["gamma"] == greeks.gamma
    assert math.isclose(desk["vega_per_vol_point"], greeks.vega / 100.0)
    assert math.isclose(desk["theta_per_day"], greeks.theta / 365.0)
    assert math.isclose(desk["rho_per_rate_point"], greeks.rho / 100.0)
    # sanity: daily theta should be a small fraction of the annualized figure
    assert abs(desk["theta_per_day"]) < abs(greeks.theta)
