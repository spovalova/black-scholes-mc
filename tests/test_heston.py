import math

import numpy as np
import pytest

import datetime as dt

import bscpp
from bscpp.backtest import (
    MockProvider,
    StripPricer,
    calibrate_heston,
    calibrate_heston_with_stability,
    heston_fit_rmse,
)
from bscpp.backtest.heston_calibration import (
    _batch_resolution_for_maturity,
    _heston_iv_jacobian,
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


@pytest.mark.slow
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


def test_heston_mc_qe_accurate_at_far_fewer_steps_than_euler_in_same_extreme_regime():
    # Andersen (2008) QE scheme (HestonMCPricer.price_qe): the whole point
    # is fixing full-truncation Euler's discretization bias -- see
    # test_heston_analytic_accurate_in_extreme_feller_violating_regime
    # directly above, which needs 3000 steps to get within 5 std errors of
    # the analytic price in exactly this regime (300 steps disagrees by
    # ~40 std errors). QE samples v(t+dt)|v(t) from a distribution moment-
    # matched to the true CIR conditional law (squared-Gaussian or
    # exponential-tailed, chosen per step by the local variance-to-mean
    # ratio) instead of an Euler step with v floored at 0, so v is exactly
    # non-negative by construction and the step-count needed for accuracy
    # doesn't blow up as Feller is violated. Verified here at a small
    # fraction of Euler's 3000 steps -- not asserted to be fast, MEASURED.
    spot, strike, rate, div, maturity = 100.0, 100.0, 0.05, 0.0, 1.0
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)

    analytic = bscpp.heston_price(spot, strike, rate, div, maturity, bscpp.OptionType.Call, hp)
    qe = bscpp.HestonMCPricer(seed=1).price_qe(spot, strike, rate, div, maturity,
                                                bscpp.OptionType.Call, hp, 150_000, 20)

    assert abs(analytic - qe.price) < 5 * qe.std_error


def test_heston_mc_qe_matches_analytic_across_stress_regimes():
    # Same stress regimes heston_price/heston_price_cos were both
    # validated against, all at the SAME low step count (20) the extreme-
    # Feller test above uses -- QE shouldn't need regime-specific tuning
    # to stay accurate.
    regimes = [
        ("normal ATM call", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("normal put", 1.0, 100.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("OTM call", 1.0, 120.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("deep OTM put", 1.0, 60.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("with dividend", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("short maturity", 0.1, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("Feller-violating put", 1.0, 100.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)),
    ]
    spot, rate = 100.0, 0.05

    for name, maturity, strike, opt_type, hp in regimes:
        div = 0.03 if name == "with dividend" else 0.0
        analytic = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type, hp)
        qe = bscpp.HestonMCPricer(seed=11).price_qe(spot, strike, rate, div, maturity, opt_type,
                                                      hp, 100_000, 20)
        assert abs(analytic - qe.price) < 5 * qe.std_error, name


def test_heston_mc_qe_forward_unbiased_without_martingale_correction():
    # price_qe deliberately skips Andersen's martingale-correction variant
    # of K0 (see heston.hpp) -- justified here, not just asserted: pricing
    # a call struck at 0 (payoff = S_T exactly, never clamped) isolates
    # E[discounted S_T] from any strike-dependent effect, and it should
    # match the theoretical forward spot*exp(-dividend_yield*maturity) to
    # within Monte Carlo noise despite the missing correction.
    spot, rate, div, maturity = 100.0, 0.05, 0.0, 1.0
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)

    qe = bscpp.HestonMCPricer(seed=5).price_qe(spot, 0.0, rate, div, maturity,
                                                bscpp.OptionType.Call, hp, 200_000, 20)
    expected = spot * math.exp(-div * maturity)
    assert abs(qe.price - expected) < 5 * qe.std_error


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


def test_batch_resolution_scales_up_below_fourteen_days_or_above_xi_one():
    # Both thresholds are load-bearing, not just the maturity one: below
    # 14 days, calibrate_heston relies on the fast (1500, 150) default
    # being accurate, and the maturity sweep behind this policy showed it
    # degrading sharply under ~5 days. Separately, a fixed-maturity sweep
    # over xi found the same default *also* breaks down as vol-of-vol
    # rises regardless of maturity -- a maturity-only policy would wrongly
    # call a long-dated, high-xi contract safe.
    assert _batch_resolution_for_maturity(45 / 365, xi=0.4) == (1500, 150.0)
    assert _batch_resolution_for_maturity(10 / 365, xi=0.4) == (8000, 800.0)
    assert _batch_resolution_for_maturity(13.9 / 365, xi=0.4) == (8000, 800.0)
    assert _batch_resolution_for_maturity(1 / 365, xi=0.4) == (8000, 800.0)
    assert _batch_resolution_for_maturity(1.0, xi=1.0) == (1500, 150.0)
    assert _batch_resolution_for_maturity(1.0, xi=1.01) == (8000, 800.0)
    assert _batch_resolution_for_maturity(1.0, xi=3.0) == (8000, 800.0)


def test_heston_price_batch_accurate_at_the_resolution_policy_actually_selects():
    # Directly exercises what calibrate_heston will actually call: for a
    # maturity on each side of the 7-day threshold, price_batch AT the
    # resolution _batch_resolution_for_maturity picks for it must stay
    # close to the trusted adaptive price -- not just "some resolution
    # exists that works," but the one the policy actually selects.
    spot, rate, div = 100.0, 0.05, 0.0
    strikes = [80, 90, 100, 110, 120]
    types = [bscpp.OptionType.Call] * 5

    for maturity, hp in [
        (45 / 365, bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)),
        (2 / 365, bscpp.HestonParams(kappa=5.0, theta=0.04, xi=3.0, rho=-0.5, v0=0.04)),
    ]:
        num_nodes, phi_max = _batch_resolution_for_maturity(maturity, hp.xi)
        batch = bscpp.heston_price_batch(spot, strikes, types, rate, div, maturity, hp,
                                          num_nodes, phi_max)
        single = [bscpp.heston_price(spot, k, rate, div, maturity, t, hp)
                  for k, t in zip(strikes, types)]
        for b, s in zip(batch, single):
            assert math.isclose(b, s, abs_tol=1e-3, rel_tol=1e-3)


def test_heston_price_batch_matches_single_price_across_stress_regimes():
    # price_batch shares characteristic-function evaluations across strikes
    # via a FIXED (not adaptive) quadrature grid -- verified here against
    # the trusted adaptive heston_price across the exact same stress cases
    # that pricer was itself validated against, not assumed to inherit
    # that accuracy for free. Each regime is given the quadrature
    # resolution _batch_resolution_for_maturity would actually select for
    # it -- the raw C++ default (1500, 150) is tuned for typical
    # calibration use and is deliberately NOT safe at short maturity (see
    # test_batch_resolution_scales_up_below_seven_days); this test checks
    # accuracy is achievable at adequate resolution, not that the default
    # happens to be enough everywhere.
    spot, rate, div = 100.0, 0.05, 0.0
    strikes = [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 150]
    types = [bscpp.OptionType.Call] * 6 + [bscpp.OptionType.Put] * 5

    regimes = [
        ("normal", 1.0, bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("1-day maturity", 1 / 365,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)),
        ("Feller-violating", 1.0,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)),
        ("worst case: 1-day + xi=3.0", 1 / 365,
         bscpp.HestonParams(kappa=5.0, theta=0.04, xi=3.0, rho=-0.5, v0=0.04)),
    ]

    for name, maturity, hp in regimes:
        num_nodes, phi_max = _batch_resolution_for_maturity(maturity, hp.xi)
        batch = bscpp.heston_price_batch(spot, strikes, types, rate, div, maturity, hp,
                                          num_nodes, phi_max)
        single = [bscpp.heston_price(spot, k, rate, div, maturity, t, hp)
                  for k, t in zip(strikes, types)]
        for b, s in zip(batch, single):
            assert math.isclose(b, s, abs_tol=1e-4, rel_tol=1e-4), name


def test_heston_price_cos_matches_adaptive_across_stress_regimes():
    # heston_price_cos (Fang & Oosterlee 2008 COS method) shares the same
    # char_function as heston_price but sums a fixed-node cosine series
    # instead of adaptively quadrature-integrating -- see heston.hpp for
    # why (closing the ~14x gap to QuantLib's AnalyticHestonEngine found
    # in benchmarks/test_heston_benchmark.py). It exists purely to be a
    # faster, cross-checked alternative to heston_price, so it's held to
    # the same stress regimes heston_price itself was validated against:
    # short maturity, badly Feller-violating vol-of-vol, and both together.
    #
    # An earlier draft of this method had a real bug caught only by this
    # kind of cross-check: it copied Fang & Oosterlee's payoff-coefficient
    # formula K*(chi_k-psi_k) verbatim, which assumes their paper's
    # log-moneyness convention x=ln(S_T/K) -- but this implementation uses
    # absolute log-price x=ln(S_T) (matching char_function), where K
    # multiplies only the psi (constant) term, not chi. The bug produced
    # prices off by 2-4 orders of magnitude, not a subtle drift, which is
    # exactly the kind of error a cross-check against a trusted reference
    # catches immediately and a "does it run" smoke test would not.
    spot, rate, div = 100.0, 0.05, 0.0
    regimes = [
        ("normal ATM call", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("normal OTM call", 1.0, 120.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("normal put", 1.0, 100.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("deep OTM put", 1.0, 60.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("1-day maturity", 1 / 365, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)),
        ("Feller-violating", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)),
        ("worst case: 1-day + xi=3.0", 1 / 365, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=5.0, theta=0.04, xi=3.0, rho=-0.5, v0=0.04)),
        ("5-year maturity", 5.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
    ]
    for name, maturity, strike, opt_type, hp in regimes:
        adaptive = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type, hp)
        cos = bscpp.heston_price_cos(spot, strike, rate, div, maturity, opt_type, hp)
        assert math.isclose(cos, adaptive, abs_tol=1e-3, rel_tol=1e-3), name


def test_heston_price_cos_matches_adaptive_across_random_stress_sweep():
    # A single hand-picked stress list can miss a parameter combination
    # that breaks the adaptive truncation-range search inside
    # heston_price_cos (this happened during development: a long-maturity,
    # badly Feller-violating case made the domain-widening loop land on
    # the no-arbitrage floor of 0.0 at two successive iterations by
    # coincidence, which looked like convergence but wasn't -- see
    # price_cos_raw in heston.cpp for why the fixed-point comparison uses
    # the unclamped price). This sweeps a wide, fixed (seeded, reproducible)
    # grid of maturities/params/strikes to catch that class of bug instead
    # of relying on hand-picked cases alone.
    rng = np.random.default_rng(1234)
    spot, rate, div = 100.0, 0.04, 0.01
    n = 60
    for _ in range(n):
        hp = bscpp.HestonParams(
            kappa=rng.uniform(0.5, 5.0),
            theta=rng.uniform(0.01, 0.1),
            xi=rng.uniform(0.1, 3.0),
            rho=rng.uniform(-0.9, 0.1),
            v0=rng.uniform(0.01, 0.1),
        )
        maturity = rng.uniform(1 / 365, 3.0)
        strike = rng.uniform(70, 140)
        opt_type = bscpp.OptionType.Call if rng.random() < 0.5 else bscpp.OptionType.Put

        adaptive = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type, hp)
        cos = bscpp.heston_price_cos(spot, strike, rate, div, maturity, opt_type, hp)
        assert abs(cos - adaptive) < max(1e-3, 1e-2 * abs(adaptive)), (hp, maturity, strike, opt_type)


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


def test_heston_price_jacobian_matches_finite_differences_across_stress_regimes():
    # heston_price_jacobian computes price() plus its exact partials
    # w.r.t. kappa/theta/xi/rho/v0 via forward-mode AD (see heston.hpp/
    # dual.hpp) -- cross-checked here against central finite differences
    # on heston_price itself, across the same stress regimes heston_price
    # and heston_price_cos were both validated against. Central
    # differences are only accurate to ~1e-6 relative themselves, so the
    # tolerance below is set by the REFERENCE's precision, not an
    # arbitrarily loose bar.
    regimes = [
        ("normal ATM call", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("normal OTM put", 1.0, 90.0, bscpp.OptionType.Put,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
        ("1-day maturity", 1 / 365, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)),
        ("Feller-violating", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=3.0, rho=-0.7, v0=0.04)),
        ("worst case: 1-day + xi=3.0", 1 / 365, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=5.0, theta=0.04, xi=3.0, rho=-0.5, v0=0.04)),
        ("with dividend", 1.0, 100.0, bscpp.OptionType.Call,
         bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)),
    ]
    spot, rate = 100.0, 0.05
    param_names = ["kappa", "theta", "xi", "rho", "v0"]

    for name, maturity, strike, opt_type, hp in regimes:
        div = 0.03 if name == "with dividend" else 0.0
        jac = bscpp.heston_price_jacobian(spot, strike, rate, div, maturity, opt_type, hp)
        adaptive = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type, hp)
        assert math.isclose(jac.price, adaptive, rel_tol=1e-8), name

        base = {"kappa": hp.kappa, "theta": hp.theta, "xi": hp.xi, "rho": hp.rho, "v0": hp.v0}
        analytic = [jac.d_kappa, jac.d_theta, jac.d_xi, jac.d_rho, jac.d_v0]
        for pname, deriv in zip(param_names, analytic):
            h = max(1e-6, abs(base[pname]) * 1e-6)
            bumped = dict(base)
            bumped[pname] = base[pname] + h
            p_plus = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type,
                                         bscpp.HestonParams(**bumped))
            bumped[pname] = base[pname] - h
            p_minus = bscpp.heston_price(spot, strike, rate, div, maturity, opt_type,
                                          bscpp.HestonParams(**bumped))
            fd = (p_plus - p_minus) / (2 * h)
            assert math.isclose(deriv, fd, abs_tol=2e-4, rel_tol=2e-3), f"{name} d/d{pname}"


def test_heston_price_jacobian_batch_matches_single_price_jacobian():
    # heston_price_jacobian_batch shares characteristic-function
    # evaluations (now including their derivatives) across strikes via a
    # fixed grid, exactly like heston_price_batch does for price() alone
    # -- verified here against the per-strike heston_price_jacobian, not
    # assumed to inherit its accuracy for free.
    spot, rate, div, maturity = 100.0, 0.05, 0.01, 0.75
    hp = bscpp.HestonParams(kappa=2.0, theta=0.045, xi=0.5, rho=-0.6, v0=0.05)
    strikes = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0]
    types = [bscpp.OptionType.Put] * 4 + [bscpp.OptionType.Call] * 5

    batch = bscpp.heston_price_jacobian_batch(spot, strikes, types, rate, div, maturity, hp)
    for k, t, b in zip(strikes, types, batch):
        single = bscpp.heston_price_jacobian(spot, k, rate, div, maturity, t, hp)
        assert math.isclose(b.price, single.price, abs_tol=1e-3, rel_tol=1e-3)
        for field in ("d_kappa", "d_theta", "d_xi", "d_rho", "d_v0"):
            assert math.isclose(getattr(b, field), getattr(single, field), abs_tol=1e-3,
                                 rel_tol=1e-3), field


def test_heston_iv_jacobian_matches_numerical_jacobian():
    # _heston_iv_jacobian converts the price-space Jacobian above into
    # IV-space via the implicit function theorem (d(iv)/d(param) =
    # dPrice/dparam / vega) -- cross-checked here at the level
    # calibrate_heston actually consumes it, against a plain central
    # difference on _heston_implied_vols itself (the function that
    # produces calibrate_heston's residuals).
    from bscpp.backtest.heston_calibration import _heston_implied_vols

    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    option_types = ["put", "put", "put", "call", "call", "call", "call"]
    spot, t_years, rate, div = 100.0, 0.75, 0.04, 0.01
    params = [2.0, 0.045, 0.5, -0.6, 0.05]

    analytic = _heston_iv_jacobian(params, strikes, option_types, spot, t_years, rate, div)

    numeric = np.zeros_like(analytic)
    for j in range(5):
        h = max(1e-6, abs(params[j]) * 1e-6)
        plus, minus = list(params), list(params)
        plus[j] += h
        minus[j] -= h
        iv_plus = _heston_implied_vols(plus, strikes, option_types, spot, t_years, rate, div)
        iv_minus = _heston_implied_vols(minus, strikes, option_types, spot, t_years, rate, div)
        numeric[:, j] = (iv_plus - iv_minus) / (2 * h)

    diff = np.abs(analytic - numeric)
    reldiff = diff / np.maximum(np.abs(numeric), 1e-6)
    assert np.all((diff < 1e-4) | (reldiff < 1e-2))


def test_calibrate_heston_analytic_jacobian_matches_finite_difference_fit():
    # The whole point of use_analytic_jacobian is to be a faster drop-in
    # for scipy's default finite-difference Jacobian, not a different
    # optimization problem -- both should converge to essentially the
    # same fit on the same data.
    strikes = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
    option_types = ["put"] * 6 + ["call"] * 7
    spot, t_years, rate, div = 100.0, 0.5, 0.04, 0.01
    true_hp = bscpp.HestonParams(kappa=2.0, theta=0.045, xi=0.5, rho=-0.6, v0=0.05)
    otypes = [bscpp.OptionType.Call if t == "call" else bscpp.OptionType.Put for t in option_types]
    prices = bscpp.heston_price_batch(spot, [float(k) for k in strikes], otypes, rate, div,
                                       t_years, true_hp)
    n = len(strikes)
    otype_arr = (np.asarray(option_types) != "call").astype(np.int32)
    market_ivs = bscpp.bs_implied_vol_batch_arrays(
        np.full(n, spot), np.asarray(strikes, dtype=float), np.full(n, rate), np.full(n, div),
        np.full(n, 0.2), np.full(n, t_years), otype_arr, np.asarray(prices))

    analytic_fit = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate, div,
                                     use_analytic_jacobian=True)
    fd_fit = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate, div,
                               use_analytic_jacobian=False)

    for field in ("kappa", "theta", "xi", "rho", "v0"):
        assert math.isclose(getattr(analytic_fit, field), getattr(fd_fit, field), abs_tol=1e-3,
                             rel_tol=1e-3), field

    analytic_rmse = heston_fit_rmse(analytic_fit, strikes, option_types, market_ivs, spot,
                                     t_years, rate, div)
    fd_rmse = heston_fit_rmse(fd_fit, strikes, option_types, market_ivs, spot, t_years, rate, div)
    assert math.isclose(analytic_rmse, fd_rmse, abs_tol=1e-6, rel_tol=1e-2)


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
    # 30 days (not 45): StripPricer now solves OTM-only (calls above the
    # implied forward, puts below -- see engine.extract_forward_and_carry),
    # so a calls-only slice would silently drop the whole ITM-call half of
    # the range. The genuine OTM-only smile keeps BOTH types and drops the
    # ITM-fallback NaNs, giving a full-width, properly-conditioned strike
    # range -- exactly how a real desk would build calibration inputs.
    spot, rate = 450.0, 0.05
    provider = MockProvider(rate=0.05, spot=spot, base_vol=0.18, smile_strength=0.40)
    pricer = StripPricer(provider, rate=rate, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=30)
    chain = pricer.price_strip("SPY", expiration, strike_range=(0.85, 1.15), use_mc=False)
    priced = chain.dropna(subset=["model_iv"])
    t_years = float(chain["T"].iloc[0])
    return (priced["strike"].to_numpy(), priced["type"].tolist(), priced["model_iv"].to_numpy(),
            spot, t_years, rate)


@pytest.mark.slow
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


@pytest.mark.slow
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


def test_heston_price_never_negative_deep_otm_short_maturity():
    # Regression test: deep-OTM short-maturity prices used to come back
    # slightly NEGATIVE (quadrature noise exceeding the tiny true price,
    # observed -2.3e-7 for the 5-day 200-strike call below). A negative
    # option price is itself an arbitrage and poisons IV solves, so the
    # pricer must clamp at the no-arbitrage floor of zero.
    hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)
    call = bscpp.heston_price(100.0, 200.0, 0.05, 0.0, 5 / 365, bscpp.OptionType.Call, hp)
    put = bscpp.heston_price(100.0, 40.0, 0.05, 0.0, 5 / 365, bscpp.OptionType.Put, hp)
    assert call >= 0.0
    assert put >= 0.0
    # and they should still be (near-)zero, not inflated by the clamp
    assert call < 1e-4
    assert put < 1e-4
