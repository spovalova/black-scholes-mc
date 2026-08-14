"""Black-Scholes analytic pricing + Monte Carlo simulation, backed by a C++ core."""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("bscpp")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "0.0.0.dev0"

from bscpp._core import (
    AmericanPricer,
    Greeks,
    HestonMCPricer,
    HestonParams,
    MarketInputs,
    MCResult,
    MonteCarloPricer,
    OptionType,
    PricingResult,
    bs_greeks,
    bs_implied_vol,
    bs_implied_vol_batch,
    bs_price,
    bs_price_with_greeks,
    bs_price_with_greeks_batch,
    crr_implied_vol,
    crr_price,
    heston_price,
    heston_price_batch,
    heston_satisfies_feller_condition,
)
from bscpp.curve import ZeroCurve, resolve_rate
from bscpp.risk import Breach, Position, PortfolioRiskManager, RiskLimits
from bscpp.strategies import (
    Leg,
    StrategyPricer,
    StrategyResult,
    butterfly,
    long_option,
    straddle,
    strangle,
    strap,
    strip,
    vertical_spread,
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
    "bs_implied_vol_batch",
    "bs_price",
    "bs_price_with_greeks",
    "bs_price_with_greeks_batch",
    "crr_price",
    "crr_implied_vol",
    "HestonMCPricer",
    "HestonParams",
    "heston_price",
    "heston_price_batch",
    "heston_satisfies_feller_condition",
    "make_inputs",
    "price",
    "price_mc",
    "price_american",
    "price_american_crr",
    "Leg",
    "StrategyPricer",
    "StrategyResult",
    "butterfly",
    "long_option",
    "straddle",
    "strangle",
    "strap",
    "strip",
    "vertical_spread",
    "trading_greeks",
    "Breach",
    "Position",
    "PortfolioRiskManager",
    "RiskLimits",
    "ZeroCurve",
    "resolve_rate",
]


def trading_greeks(greeks: Greeks) -> dict:
    """Convert calculus Greeks to trading-desk convention.

    `bs_greeks`/`bs_price_with_greeks` return the literal partial
    derivatives: vega and rho per 1.00 (a full 100 percentage points) of
    vol/rate, theta per YEAR. No desk quotes them that way -- a "vega of
    40" on a desk means $40 per 1-vol-point (1%) move, and theta is always
    read as dollars lost per calendar day. Silently mixing the two
    conventions is a classic, costly beginner mistake (a theta of -60/year
    looks alarming; the same position decaying at -0.16/day usually isn't).
    This is a pure unit conversion, nothing more:

        vega_per_vol_point  = vega / 100
        rho_per_rate_point  = rho  / 100
        theta_per_day       = theta / 365

    delta and gamma are unchanged -- both are already quoted per $1 of
    spot movement on both a textbook and a trading desk.
    """
    return {
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega_per_vol_point": greeks.vega / 100.0,
        "theta_per_day": greeks.theta / 365.0,
        "rho_per_rate_point": greeks.rho / 100.0,
    }


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
                    num_paths=50_000, num_steps=50, poly_degree=2, seed=42,
                    num_calibration_paths=0):
    """American-style price via Longstaff-Schwartz LSM; returns MCResult(price, std_error).

    Early exercise only carries value for puts (or calls with dividends);
    an American call with no dividends will price essentially identically
    to the European Black-Scholes call -- that equivalence is itself a
    useful sanity check (see tests/test_american.py).

    Regresses the exercise policy on an independently-seeded calibration
    path set (num_calibration_paths, defaults to num_paths) and applies it
    to a separate pricing path set -- avoids the small upward look-ahead
    bias of regressing and pricing on the same paths.
    """
    inputs = make_inputs(spot, strike, rate, vol, maturity, option_type, dividend_yield)
    lsm = AmericanPricer(seed=seed)
    return lsm.price(inputs, num_paths, num_steps, poly_degree, num_calibration_paths)


def price_american_crr(spot, strike, rate, vol, maturity, option_type="put", dividend_yield=0.0,
                        num_steps=200):
    """American-style price via a dividend-aware Cox-Ross-Rubinstein binomial
    tree. Deterministic (no std_error -- there's no simulation noise to
    report), and microseconds per call at the default num_steps, vs.
    price_american's Monte Carlo LSM.

    This, not price_american, is what StripPricer uses for American implied
    vol in the chain pipeline (see bscpp.backtest.engine): a real equity
    chain is American-style, LSM is overkill for a single-asset vanilla
    price with no path dependence, and its simulation noise would make an
    IV solve noisy for no benefit. price_american/AmericanPricer (LSM)
    remains this project's cross-check on CRR and its generalization to
    path-dependent/multi-factor cases CRR can't handle -- see crr_tree.hpp.
    """
    otype = OptionType.Call if str(option_type).lower().startswith("c") else OptionType.Put
    return crr_price(spot, strike, rate, dividend_yield, maturity, otype, vol, num_steps)
