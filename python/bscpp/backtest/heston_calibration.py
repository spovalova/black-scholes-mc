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
    regularization_weight: float = 0.05,
):
    """Fit Heston params to (strike, option_type, market_iv) triples at one expiry.

    option_types: sequence of "call"/"put", same length as strikes/market_ivs.
    Returns a bscpp.HestonParams.

    `regularization_weight` adds Tikhonov-style penalty residuals pulling
    v0 and theta toward the ATM variance prior and kappa toward a
    market-typical order of magnitude (~2.0), scaled to roughly the same
    units as the IV residuals. This exists because the raw (unregularized)
    fit can drive v0 to its lower bound on short-dated/mildly-curved
    smiles -- a real, observed failure mode (see heston_calibration_demo.py)
    where many (v0, kappa, theta) combinations fit the data almost equally
    well. Regularization doesn't resolve the underlying identifiability
    issue (nothing can, without more data -- e.g. multiple expiries), but
    it keeps the *fitted* parameters in an economically sane region instead
    of an arbitrary corner of the flat direction, without materially
    hurting fit quality when the data DOES pin the parameters down (set
    regularization_weight=0 to recover the unregularized fit and compare).
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

    atm_var = float(np.median(market_ivs)) ** 2
    kappa_prior = 2.0

    def residuals(params):
        kappa, theta, xi, rho, v0 = params
        main = _heston_implied_vols(params, strikes, option_types, spot, t_years, rate,
                                     dividend_yield) - market_ivs
        if regularization_weight <= 0.0:
            return main
        reg = np.sqrt(regularization_weight) * np.array([
            v0 - atm_var,
            theta - atm_var,
            (kappa - kappa_prior) / kappa_prior,  # relative scale, kappa's units differ from vol
        ])
        return np.concatenate([main, reg])

    if initial_guess is None:
        initial_guess = [kappa_prior, atm_var, 0.5, -0.5, atm_var]

    result = least_squares(residuals, x0=initial_guess, bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS),
                            max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
    kappa, theta, xi, rho, v0 = result.x
    return bscpp.HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


def calibrate_heston_with_stability(
    strikes,
    option_types,
    market_ivs,
    spot: float,
    t_years: float,
    rate: float,
    dividend_yield: float = 0.0,
    n_starts: int = 6,
    seed: int = 0,
    **calibrate_kwargs,
) -> dict:
    """Run calibration from several randomly-perturbed initial guesses and
    report whether the result is stable -- both in fit quality (do all
    starts reach a similarly good RMSE) and in the parameters themselves
    (how much does each parameter vary across starts).

    This exists because a single calibrate_heston() call reports exactly
    one point estimate with no indication of whether that estimate is
    well-identified or an arbitrary point along a flat direction in the
    loss surface (the v0 issue documented in calibrate_heston's
    docstring). A desk re-hedging off tomorrow's re-calibration needs to
    know which of these it's looking at.

    Returns a dict: best_params (lowest-RMSE result), best_rmse,
    all_rmse (one per start), param_std (per-parameter std dev across
    starts), fit_quality_stable (bool: all starts reached within 0.5 vol
    points of the best RMSE -- the fit itself is stable even if individual
    parameters are not), params_stable (bool: every parameter's std dev
    across starts is small relative to its typical scale).
    """
    rng = np.random.default_rng(seed)
    atm_var = float(np.median(np.asarray(market_ivs, dtype=float))) ** 2

    results = []
    for _ in range(n_starts):
        guess = [
            2.0 * rng.uniform(0.4, 2.5),
            atm_var * rng.uniform(0.4, 2.5),
            0.5 * rng.uniform(0.4, 2.5),
            float(np.clip(-0.5 * rng.uniform(0.3, 1.8), -0.95, 0.95)),
            atm_var * rng.uniform(0.4, 2.5),
        ]
        params = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate,
                                   dividend_yield, initial_guess=guess, **calibrate_kwargs)
        rmse = heston_fit_rmse(params, strikes, option_types, market_ivs, spot, t_years, rate,
                                dividend_yield)
        results.append((params, rmse))

    best_params, best_rmse = min(results, key=lambda r: r[1])
    all_rmse = np.array([r[1] for r in results])

    param_names = ["kappa", "theta", "xi", "rho", "v0"]
    param_arrays = {name: np.array([getattr(p, name) for p, _ in results]) for name in param_names}
    param_std = {name: float(arr.std()) for name, arr in param_arrays.items()}
    # "small relative to typical scale": std < 20% of the mean absolute value
    params_stable = all(
        param_std[name] < 0.2 * max(abs(param_arrays[name].mean()), 1e-6) for name in param_names
    )

    return {
        "best_params": best_params,
        "best_rmse": float(best_rmse),
        "all_rmse": all_rmse.tolist(),
        "param_std": param_std,
        "fit_quality_stable": bool(all_rmse.max() - all_rmse.min() < 0.005),
        "params_stable": params_stable,
    }


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
