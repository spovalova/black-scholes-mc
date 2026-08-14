"""American option pricing: bscpp's CRR tree vs. QuantLib's CRR binomial
engine, at the same step count -- the same algorithm, two independent
implementations. See benchmarks/README.md.

vollib doesn't price American options, so there's no third reference here.
"""

import math

import QuantLib as ql
import pytest

import bscpp
from conftest import ql_force_recompute

# Longstaff & Schwartz (2001) headline example, also used in test_crr_tree.py
# and test_american.py.
S, K, RATE, DIV, VOL, T = 36.0, 40.0, 0.06, 0.0, 0.20, 1.0
NUM_STEPS = 500


@pytest.fixture(scope="module")
def ql_american_option():
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

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.AmericanExercise(calc_date, maturity))
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", NUM_STEPS))
    return option, spot_quote


def test_american_price_correctness(ql_american_option):
    option, _ = ql_american_option
    bscpp_price = bscpp.price_american_crr(S, K, RATE, VOL, T, "put", DIV, num_steps=NUM_STEPS)
    ql_price = option.NPV()
    # Same algorithm (CRR), same step count, two independent
    # implementations -- should agree tightly, not just both be "close to
    # 4.47-4.48" (the looser literature-benchmark tolerance used elsewhere).
    assert math.isclose(bscpp_price, ql_price, rel_tol=1e-3)


def test_american_price_bscpp(benchmark):
    benchmark(bscpp.price_american_crr, S, K, RATE, VOL, T, "put", DIV, NUM_STEPS)


def test_american_price_quantlib(benchmark, ql_american_option):
    option, spot_quote = ql_american_option
    benchmark(ql_force_recompute(option, spot_quote, S))
