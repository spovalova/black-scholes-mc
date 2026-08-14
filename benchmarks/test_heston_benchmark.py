"""Heston semi-analytic pricing: bscpp vs. QuantLib's AnalyticHestonEngine,
on identical inputs. See benchmarks/README.md.

vollib doesn't price Heston, so there's no third reference here.
"""

import math

import QuantLib as ql
import pytest

import bscpp
from conftest import ql_force_recompute

S, K, RATE, DIV, T = 100.0, 100.0, 0.05, 0.0, 1.0
KAPPA, THETA, XI, RHO, V0 = 2.0, 0.04, 0.4, -0.7, 0.05


@pytest.fixture(scope="module")
def ql_heston_option():
    calc_date = ql.Date(15, 8, 2026)
    ql.Settings.instance().evaluationDate = calc_date
    day_count = ql.Actual365Fixed()
    maturity = calc_date + int(round(T * 365))

    spot_quote = ql.SimpleQuote(S)
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, RATE, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, DIV, day_count))
    process = ql.HestonProcess(rate_ts, div_ts, ql.QuoteHandle(spot_quote), V0, KAPPA, THETA, XI, RHO)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
    option.setPricingEngine(engine)
    return option, spot_quote


def test_heston_price_correctness(ql_heston_option):
    option, _ = ql_heston_option
    hp = bscpp.HestonParams(kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, v0=V0)
    bscpp_price = bscpp.heston_price(S, K, RATE, DIV, T, bscpp.OptionType.Call, hp)
    ql_price = option.NPV()
    # Two independent characteristic-function implementations; the
    # README already claims (and this confirms) they match essentially
    # exactly, not just "close."
    assert math.isclose(bscpp_price, ql_price, rel_tol=1e-6)


def test_heston_price_bscpp(benchmark):
    hp = bscpp.HestonParams(kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, v0=V0)
    benchmark(bscpp.heston_price, S, K, RATE, DIV, T, bscpp.OptionType.Call, hp)


def test_heston_price_quantlib(benchmark, ql_heston_option):
    option, spot_quote = ql_heston_option
    benchmark(ql_force_recompute(option, spot_quote, S))
