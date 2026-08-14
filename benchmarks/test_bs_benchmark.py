"""Black-Scholes price and implied-vol solve: bscpp vs. QuantLib vs.
vollib, on identical inputs. See benchmarks/README.md.
"""

import math

import QuantLib as ql
import pytest
from vollib.black_scholes_merton import black_scholes_merton
from vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_implied_vol

import bscpp
from conftest import ql_force_recompute

S, K, RATE, DIV, VOL, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0


@pytest.fixture(scope="module")
def ql_european_option():
    calc_date = ql.Date(15, 8, 2026)
    ql.Settings.instance().evaluationDate = calc_date
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    maturity = calc_date + int(round(T * 365))

    spot_quote = ql.SimpleQuote(S)
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, RATE, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, DIV, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calc_date, calendar, VOL, day_count))
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot_quote), div_ts, rate_ts, vol_ts)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option, process, spot_quote


@pytest.fixture(scope="module")
def bscpp_inputs():
    return bscpp.make_inputs(S, K, RATE, VOL, T, "call", DIV)


# --- correctness, checked before anything is timed ---

def test_bs_price_correctness(ql_european_option):
    option, _, _ = ql_european_option
    bscpp_price = bscpp.price(S, K, RATE, VOL, T, "call", DIV)
    ql_price = option.NPV()
    vollib_price = black_scholes_merton('c', S, K, T, RATE, VOL, DIV)
    assert math.isclose(bscpp_price, ql_price, rel_tol=1e-6)
    assert math.isclose(bscpp_price, vollib_price, rel_tol=1e-6)


def test_bs_implied_vol_correctness(ql_european_option, bscpp_inputs):
    option, process, _ = ql_european_option
    market_price = bscpp.price(S, K, RATE, VOL, T, "call", DIV)
    bscpp_iv = bscpp.bs_implied_vol(bscpp_inputs, market_price)
    ql_iv = option.impliedVolatility(market_price, process)
    vollib_iv = bsm_implied_vol(market_price, S, K, T, RATE, DIV, 'c')
    assert math.isclose(bscpp_iv, VOL, abs_tol=1e-4)
    assert math.isclose(ql_iv, VOL, abs_tol=1e-3)  # QuantLib's default solver tolerance is coarser
    assert math.isclose(vollib_iv, VOL, abs_tol=1e-4)


# --- timing ---

def test_bs_price_bscpp(benchmark, bscpp_inputs):
    benchmark(bscpp.bs_price, bscpp_inputs)


def test_bs_price_quantlib(benchmark, ql_european_option):
    option, _, spot_quote = ql_european_option
    benchmark(ql_force_recompute(option, spot_quote, S))


def test_bs_price_vollib(benchmark):
    benchmark(black_scholes_merton, 'c', S, K, T, RATE, VOL, DIV)


def test_bs_implied_vol_bscpp(benchmark, bscpp_inputs):
    market_price = bscpp.price(S, K, RATE, VOL, T, "call", DIV)
    benchmark(bscpp.bs_implied_vol, bscpp_inputs, market_price)


def test_bs_implied_vol_quantlib(benchmark, ql_european_option):
    # impliedVolatility() runs a genuine internal solve each call (verified
    # by direct measurement: ~5us/call, consistent with a real Newton/
    # Brent solve, not the ~0.2us a cached lookup would show) -- unlike
    # raw NPV(), it does NOT need the quote-perturbation forcing above.
    option, process, _ = ql_european_option
    market_price = option.NPV()
    benchmark(option.impliedVolatility, market_price, process)


def test_bs_implied_vol_vollib(benchmark):
    market_price = black_scholes_merton('c', S, K, T, RATE, VOL, DIV)
    benchmark(bsm_implied_vol, market_price, S, K, T, RATE, DIV, 'c')
