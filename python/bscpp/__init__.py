"""Black-Scholes analytic pricing + Monte Carlo simulation, backed by a C++ core."""

from bscpp._core import (
    AmericanPricer,
    Greeks,
    MarketInputs,
    MCResult,
    MonteCarloPricer,
    OptionType,
    PricingResult,
    bs_greeks,
    bs_implied_vol,
    bs_price,
    bs_price_with_greeks,
)

__all__ = [
    "AmericanPricer",
    "Greeks",
    "MarketInputs",
    "MCResult",
    "MonteCarloPricer",
    "OptionType",
    "PricingResult",
    "bs_greeks",
    "bs_implied_vol",
    "bs_price",
    "bs_price_with_greeks",
    "make_inputs",
    "price",
    "price_mc",
    "price_american",
]


def make_inputs(spot, strike, rate, vol, maturity, option_type="call", dividend_yield=0.0):
    """Build a MarketInputs from Pythonic kwargs (option_type as 'call'/'put' string)."""
    otype = OptionType.Call if str(option_type).lower().startswith("c") else OptionType.Put
    return MarketInputs(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        vol=vol,
        maturity=maturity,
        type=otype,
    )


def price(spot, strike, rate, vol, maturity, option_type="call", dividend_yield=0.0):
    """Analytic Black-Scholes price."""
    inputs = make_inputs(spot, strike, rate, vol, maturity, option_type, dividend_yield)
    return bs_price(inputs)


def price_mc(spot, strike, rate, vol, maturity, option_type="call", dividend_yield=0.0,
             num_paths=100_000, antithetic=True, seed=42):
    """Monte Carlo European price; returns an MCResult(price, std_error)."""
    inputs = make_inputs(spot, strike, rate, vol, maturity, option_type, dividend_yield)
    mc = MonteCarloPricer(seed=seed)
    return mc.price_european(inputs, num_paths, antithetic)


def price_american(spot, strike, rate, vol, maturity, option_type="put", dividend_yield=0.0,
                    num_paths=50_000, num_steps=50, poly_degree=2, seed=42):
    """American-style price via Longstaff-Schwartz LSM; returns MCResult(price, std_error).

    Early exercise only carries value for puts (or calls with dividends);
    an American call with no dividends will price essentially identically
    to the European Black-Scholes call -- that equivalence is itself a
    useful sanity check (see tests/test_american.py).
    """
    inputs = make_inputs(spot, strike, rate, vol, maturity, option_type, dividend_yield)
    lsm = AmericanPricer(seed=seed)
    return lsm.price(inputs, num_paths, num_steps, poly_degree)
