"""
C++ Black-Scholes analytic pricer and Monte Carlo pricer, exposed to Python.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AmericanPricer', 'Greeks', 'HestonJacobian', 'HestonMCPricer', 'HestonParams', 'MCResult', 'MarketInputs', 'MonteCarloPricer', 'OptionType', 'PricingResult', 'bs_greeks', 'bs_implied_vol', 'bs_implied_vol_batch', 'bs_implied_vol_batch_arrays', 'bs_price', 'bs_price_with_greeks', 'bs_price_with_greeks_batch', 'bs_price_with_greeks_batch_arrays', 'crr_implied_vol', 'crr_price', 'heston_price', 'heston_price_batch', 'heston_price_cos', 'heston_price_jacobian', 'heston_price_jacobian_batch', 'heston_satisfies_feller_condition']
class AmericanPricer:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 42) -> None:
        ...
    def price(self, inputs: MarketInputs, num_paths: typing.SupportsInt | typing.SupportsIndex, num_steps: typing.SupportsInt | typing.SupportsIndex, poly_degree: typing.SupportsInt | typing.SupportsIndex = 2, num_calibration_paths: typing.SupportsInt | typing.SupportsIndex = 0) -> MCResult:
        """
        American-style option price via Longstaff-Schwartz least-squares Monte Carlo, using an independently-seeded calibration path set (num_calibration_paths, defaults to num_paths) separate from the pricing path set.
        """
class Greeks:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def delta(self) -> float:
        ...
    @delta.setter
    def delta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def theta(self) -> float:
        ...
    @theta.setter
    def theta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vega(self) -> float:
        ...
    @vega.setter
    def vega(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class HestonJacobian:
    def __repr__(self) -> str:
        ...
    @property
    def d_kappa(self) -> float:
        ...
    @d_kappa.setter
    def d_kappa(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_rho(self) -> float:
        ...
    @d_rho.setter
    def d_rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_theta(self) -> float:
        ...
    @d_theta.setter
    def d_theta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_v0(self) -> float:
        ...
    @d_v0.setter
    def d_v0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_xi(self) -> float:
        ...
    @d_xi.setter
    def d_xi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def price(self) -> float:
        ...
    @price.setter
    def price(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class HestonMCPricer:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 42) -> None:
        ...
    def price(self, spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, params: HestonParams, num_paths: typing.SupportsInt | typing.SupportsIndex, num_steps: typing.SupportsInt | typing.SupportsIndex) -> MCResult:
        ...
    def price_qe(self, spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, params: HestonParams, num_paths: typing.SupportsInt | typing.SupportsIndex, num_steps: typing.SupportsInt | typing.SupportsIndex) -> MCResult:
        """
        Andersen (2008) QE scheme -- see heston.hpp for why this needs far fewer num_steps than price() (full-truncation Euler) to reach comparable accuracy, especially when the Feller condition is badly violated.
        """
class HestonParams:
    def __init__(self, kappa: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex, xi: typing.SupportsFloat | typing.SupportsIndex, rho: typing.SupportsFloat | typing.SupportsIndex, v0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def kappa(self) -> float:
        ...
    @kappa.setter
    def kappa(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def theta(self) -> float:
        ...
    @theta.setter
    def theta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def v0(self) -> float:
        ...
    @v0.setter
    def v0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def xi(self) -> float:
        ...
    @xi.setter
    def xi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MCResult:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def price(self) -> float:
        ...
    @price.setter
    def price(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def std_error(self) -> float:
        ...
    @std_error.setter
    def std_error(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MarketInputs:
    type: OptionType
    # The C++ binding (cpp/src/bindings.cpp) declares dividend_yield's
    # default before vol/maturity's required (no-default) params -- valid
    # for pybind11's own keyword-argument matching at the C++ level, but
    # not expressible as a plain positional Python signature (a required
    # param can't follow a defaulted one). vol/maturity/type are marked
    # keyword-only here to stay valid Python while matching how this
    # constructor is actually called throughout this codebase already
    # (see bscpp.make_inputs) -- always by keyword, never positionally
    # past rate.
    def __init__(self, spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex = 0.0, *, vol: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType = OptionType.Call) -> None:
        ...
    @property
    def dividend_yield(self) -> float:
        ...
    @dividend_yield.setter
    def dividend_yield(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def maturity(self) -> float:
        ...
    @maturity.setter
    def maturity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rate(self) -> float:
        ...
    @rate.setter
    def rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def spot(self) -> float:
        ...
    @spot.setter
    def spot(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def strike(self) -> float:
        ...
    @strike.setter
    def strike(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vol(self) -> float:
        ...
    @vol.setter
    def vol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MonteCarloPricer:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 42) -> None:
        ...
    def greeks_european(self, inputs: MarketInputs, num_paths: typing.SupportsInt | typing.SupportsIndex, antithetic: bool = True, bump_frac: typing.SupportsFloat | typing.SupportsIndex = 0.01) -> Greeks:
        ...
    def price_european(self, inputs: MarketInputs, num_paths: typing.SupportsInt | typing.SupportsIndex, antithetic: bool = True) -> MCResult:
        ...
class OptionType:
    """
    Members:
    
      Call
    
      Put
    """
    Call: typing.ClassVar[OptionType]  # value = <OptionType.Call: 0>
    Put: typing.ClassVar[OptionType]  # value = <OptionType.Put: 1>
    __members__: typing.ClassVar[dict[str, OptionType]]  # value = {'Call': <OptionType.Call: 0>, 'Put': <OptionType.Put: 1>}
    @typing.overload
    def __eq__(self, other: OptionType) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: OptionType) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PricingResult:
    greeks: Greeks
    def __init__(self) -> None:
        ...
    @property
    def price(self) -> float:
        ...
    @price.setter
    def price(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def _philox_raw_draws(seed: typing.SupportsInt | typing.SupportsIndex, stream: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
    ...
def _philox_seek_draws(seed: typing.SupportsInt | typing.SupportsIndex, c0: typing.SupportsInt | typing.SupportsIndex, c1: typing.SupportsInt | typing.SupportsIndex, c2: typing.SupportsInt | typing.SupportsIndex, c3: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
    ...
def bs_greeks(inputs: MarketInputs) -> Greeks:
    ...
def bs_implied_vol(inputs: MarketInputs, market_price: typing.SupportsFloat | typing.SupportsIndex, initial_guess: typing.SupportsFloat | typing.SupportsIndex = 0.2, max_iter: typing.SupportsInt | typing.SupportsIndex = 100, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08) -> float:
    ...
def bs_implied_vol_batch(inputs: collections.abc.Sequence[MarketInputs], market_prices: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], initial_guess: typing.SupportsFloat | typing.SupportsIndex = 0.2, max_iter: typing.SupportsInt | typing.SupportsIndex = 100, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08) -> list[float]:
    ...
def bs_implied_vol_batch_arrays(spot: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], strike: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], rate: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dividend_yield: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], vol: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], maturity: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], type: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], market_price: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], initial_guess: typing.SupportsFloat | typing.SupportsIndex = 0.2, max_iter: typing.SupportsInt | typing.SupportsIndex = 100, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08) -> numpy.typing.NDArray[numpy.float64]:
    """
    Struct-of-arrays batch implied-vol solve: NumPy arrays in, NumPy array out. GIL released for the whole loop.
    """
def bs_price(inputs: MarketInputs) -> float:
    ...
def bs_price_with_greeks(inputs: MarketInputs) -> PricingResult:
    ...
def bs_price_with_greeks_batch(inputs: collections.abc.Sequence[MarketInputs]) -> list[PricingResult]:
    """
    Price + Greeks for a list of MarketInputs in one C++ call (avoids per-contract Python<->C++ crossing overhead when pricing a whole chain).
    """
def bs_price_with_greeks_batch_arrays(spot: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], strike: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], rate: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dividend_yield: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], vol: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], maturity: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], type: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Struct-of-arrays batch price+Greeks: NumPy arrays in, NumPy arrays out ((price, delta, gamma, vega, theta, rho)), type as 0=Call/1=Put. GIL released for the whole loop. See bs_price_with_greeks_batch's docstring for why this exists alongside the list-of-MarketInputs version.
    """
def crr_implied_vol(spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, market_price: typing.SupportsFloat | typing.SupportsIndex, num_steps: typing.SupportsInt | typing.SupportsIndex = 200, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-06, max_iter: typing.SupportsInt | typing.SupportsIndex = 100) -> float:
    """
    American implied vol via Brent's method against crr_price -- NaN if market_price isn't bracketed by [1e-6, 5.0] vol, matching bs_implied_vol's contract exactly.
    """
def crr_price(spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, vol: typing.SupportsFloat | typing.SupportsIndex, num_steps: typing.SupportsInt | typing.SupportsIndex = 200) -> float:
    """
    American-style price via a dividend-aware Cox-Ross-Rubinstein binomial tree -- see crr_tree.hpp for why this, not Longstaff-Schwartz MC, is the chain pipeline's American pricer.
    """
def heston_price(spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, params: HestonParams) -> float:
    ...
def heston_price_batch(spot: typing.SupportsFloat | typing.SupportsIndex, strikes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], types: collections.abc.Sequence[OptionType], rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, params: HestonParams, num_nodes: typing.SupportsInt | typing.SupportsIndex = 1500, phi_max: typing.SupportsFloat | typing.SupportsIndex = 150.0) -> list[float]:
    """
    Prices a whole strike grid in one call, sharing characteristic-function evaluations across strikes -- see heston.hpp for why this is faster, not just more convenient, than calling heston_price in a loop, and for when it isn't.
    """
def heston_price_cos(spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, params: HestonParams, num_terms: typing.SupportsInt | typing.SupportsIndex = 160) -> float:
    """
    Fang & Oosterlee (2008) COS-method price -- see heston.hpp for the fixed-node-vs-adaptive-quadrature tradeoff against heston_price, and its accuracy/speed profile (cross-checked to <0.02% relative error against heston_price across a 300-case random stress sweep; falls back to heston_price itself on the rare parameter combinations where its adaptive truncation search doesn't converge).
    """
def heston_price_jacobian(spot: typing.SupportsFloat | typing.SupportsIndex, strike: typing.SupportsFloat | typing.SupportsIndex, rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, type: OptionType, params: HestonParams) -> HestonJacobian:
    """
    price() plus its exact partials w.r.t. all 5 Heston parameters in one pass (forward-mode AD, not finite differences) -- see heston.hpp for why this needs a second, independent differentiation unit rather than literal complex-step, and bscpp.backtest.heston_calibration for how calibrate_heston uses it.
    """
def heston_price_jacobian_batch(spot: typing.SupportsFloat | typing.SupportsIndex, strikes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], types: collections.abc.Sequence[OptionType], rate: typing.SupportsFloat | typing.SupportsIndex, dividend_yield: typing.SupportsFloat | typing.SupportsIndex, maturity: typing.SupportsFloat | typing.SupportsIndex, params: HestonParams, num_nodes: typing.SupportsInt | typing.SupportsIndex = 1500, phi_max: typing.SupportsFloat | typing.SupportsIndex = 150.0) -> list[HestonJacobian]:
    """
    heston_price_jacobian, batched across a strike grid the same way heston_price_batch batches heston_price -- see heston.hpp for why the per-strike version alone is NOT a win over finite differences (measured ~3.6x slower).
    """
def heston_satisfies_feller_condition(params: HestonParams) -> bool:
    ...
