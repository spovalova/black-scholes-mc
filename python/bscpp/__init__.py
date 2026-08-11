"""Black-Scholes analytic pricing + Monte Carlo simulation, backed by a C++ core."""

from bscpp._core import (
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
    "OptionType",
    "make_inputs",
    "price",
    "price_mc",
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
    """Monte Carlo price; returns an MCResult(price, std_error)."""
    inputs = make_inputs(spot, strike, rate, vol, maturity, option_type, dividend_yield)
    mc = MonteCarloPricer(seed=seed)
    return mc.price_european(inputs, num_paths, antithetic)
