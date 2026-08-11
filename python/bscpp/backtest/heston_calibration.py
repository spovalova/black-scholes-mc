"""Calibrate the Heston stochastic-volatility model to a market chain.

Calibration is done in implied-vol space (Heston price -> BS-implied vol,
minimize vs. market implied vol) rather than raw price space. Price-space
least squares is dominated by deep ITM contracts, whose price is almost
pure intrinsic value and barely moves with the vol parameters -- fitting
there is trivial and uninformative, while the OTM wings (where the actual
vol-surface information lives) get underweighted. IV-space calibration
treats every strike comparably, the same reasoning behind `fit_svi_slice`
comparing total variance rather than raw price.

Heston has a well-known parameter identifiability issue: multiple (kappa,
theta, xi, rho, v0) combinations can produce near-identical smiles over a
finite strike range. This module (and its tests) therefore judge success
by fit quality (IV RMSE), not by recovering a specific known parameter
vector -- that's a more honest bar for what calibration can actually promise.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import bscpp

# kappa, theta, xi, rho, v0
_LOWER_BOUNDS = [1e-3, 1e-4, 1e-3, -0.999, 1e-4]
_UPPER_BOUNDS = [20.0, 4.0, 5.0, 0.999, 4.0]


def _heston_implied_vols(params, strikes, option_types, spot, t_years, rate, dividend_yield):
    kappa, theta, xi, rho, v0 = params
    hp = bscpp.HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

    otypes = [bscpp.OptionType.Call if t == "call" else bscpp.OptionType.Put for t in option_types]
    prices = [
        bscpp.heston_price(spot, float(k), rate, dividend_yield, t_years, ot, hp)
        for k, ot in zip(strikes, otypes)
    ]

    seed_inputs = [
        bscpp.make_inputs(spot, float(k), rate, 0.2, t_years, t, dividend_yield)
        for k, t in zip(strikes, option_types)
    ]
    ivs = bscpp.bs_implied_vol_batch(seed_inputs, prices)
    # a failed IV solve (NaN) means this parameter guess produced an
    # arbitrage-violating or otherwise unrecoverable price; push the
    # optimizer away from it with a large (but finite) residual instead of
    # propagating NaN into least_squares.
    return np.array([iv if iv == iv else 5.0 for iv in ivs])


def calibrate_heston(
    strikes,
    option_types,
    market_ivs,
    spot: float,
    t_years: float,
    rate: float,
    dividend_yield: float = 0.0,
    initial_guess: list[float] | None = None,
    max_nfev: int = 300,
):
    """Fit Heston params to (strike, option_type, market_iv) triples at one expiry.

    option_types: sequence of "call"/"put", same length as strikes/market_ivs.
    Returns a bscpp.HestonParams.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    option_types = list(option_types)
    valid = np.isfinite(market_ivs) & (market_ivs > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes = strikes[valid]
    market_ivs = market_ivs[valid]
    option_types = [t for t, v in zip(option_types, valid) if v]
    if strikes.size < 6:
        raise ValueError("need at least 6 valid (strike, iv) points to calibrate Heston")

    def residuals(params):
        return _heston_implied_vols(params, strikes, option_types, spot, t_years, rate,
                                     dividend_yield) - market_ivs

    if initial_guess is None:
        atm_var = float(np.median(market_ivs)) ** 2
        initial_guess = [2.0, atm_var, 0.5, -0.5, atm_var]

    result = least_squares(residuals, x0=initial_guess, bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS),
                            max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
    kappa, theta, xi, rho, v0 = result.x
    return bscpp.HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


def heston_fit_rmse(
    params, strikes, option_types, market_ivs, spot: float, t_years: float, rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """RMSE of the calibrated fit, in implied-vol units."""
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    option_types = list(option_types)
    valid = np.isfinite(market_ivs) & (market_ivs > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes, market_ivs = strikes[valid], market_ivs[valid]
    option_types = [t for t, v in zip(option_types, valid) if v]

    fitted_ivs = _heston_implied_vols(
        [params.kappa, params.theta, params.xi, params.rho, params.v0],
        strikes, option_types, spot, t_years, rate, dividend_yield,
    )
    return float(np.sqrt(np.mean((fitted_ivs - market_ivs) ** 2)))
