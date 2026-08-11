"""Implied volatility smile fitting via Gatheral's SVI parameterization.

SVI ("Stochastic Volatility Inspired") models total implied variance as a
function of log-moneyness k = ln(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where w(k) = sigma_impl(k)^2 * T. It's a standard industry parameterization
(Gatheral 2004) because five parameters capture level (a), angle/slope of
the wings (b, rho), and the location/curvature of the smile minimum (m,
sigma) while staying well-behaved (arbitrage-free in the "no calendar/
butterfly arbitrage" sense) over a much wider strike range than a raw
polynomial fit in strike.

Fitting one slice (one expiration) at a time here -- a full surface is a
sequence of per-expiry SVISlice fits, one per available expiration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVISlice:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    t: float  # time to expiry this slice was fit at, in years

    def total_variance(self, log_moneyness):
        k = np.asarray(log_moneyness, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, log_moneyness):
        w = np.maximum(self.total_variance(log_moneyness), 0.0)
        return np.sqrt(w / self.t)


def fit_svi_slice(
    strikes,
    market_ivs,
    spot: float,
    t_years: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> SVISlice:
    """Least-squares fit an SVI slice to (strike, implied vol) pairs at one expiry."""
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    valid = np.isfinite(market_ivs) & (market_ivs > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes, market_ivs = strikes[valid], market_ivs[valid]
    if strikes.size < 6:
        raise ValueError("need at least 6 valid (strike, iv) points to fit an SVI slice")

    forward = spot * np.exp((rate - dividend_yield) * t_years)
    k = np.log(strikes / forward)
    w_obs = market_ivs ** 2 * t_years

    def residuals(params):
        a, b, rho, m, sigma = params
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        return model - w_obs

    x0 = [max(w_obs.min(), 1e-4), 0.1, -0.3, 0.0, 0.1]
    bounds = ([-np.inf, 0.0, -0.999, -np.inf, 1e-4], [np.inf, np.inf, 0.999, np.inf, np.inf])
    result = least_squares(residuals, x0=x0, bounds=bounds, max_nfev=5000)

    a, b, rho, m, sigma = result.x
    return SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma, t=t_years)


def svi_fit_rmse(
    svi: SVISlice, strikes, market_ivs, spot: float, rate: float = 0.0, dividend_yield: float = 0.0
) -> float:
    """Root-mean-square error of the fit, in implied-vol (not variance) units."""
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    valid = np.isfinite(market_ivs) & (market_ivs > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes, market_ivs = strikes[valid], market_ivs[valid]

    forward = spot * np.exp((rate - dividend_yield) * svi.t)
    k = np.log(strikes / forward)
    fitted = svi.implied_vol(k)
    return float(np.sqrt(np.mean((fitted - market_ivs) ** 2)))


def svi_min_total_variance(svi: SVISlice) -> float:
    """w(k) attains its minimum at k* = m + rho*sigma/sqrt(1-rho^2), value below.

    A negative minimum means the slice implies a negative total variance
    somewhere -- an immediate (necessary-condition) arbitrage violation, and
    a fast check before bothering with the full density scan below.
    """
    return svi.a + svi.b * svi.sigma * np.sqrt(max(1.0 - svi.rho ** 2, 0.0))


def svi_butterfly_arbitrage_check(
    svi: SVISlice,
    spot: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    k_range: tuple[float, float] = (-1.5, 1.5),
    n_points: int = 300,
) -> dict:
    """No-butterfly-arbitrage check via Breeden-Litzenberger (1978).

    The risk-neutral density implied by a strike-continuum of call prices is
    q(K) = e^{rT} * d^2C/dK^2; absence of butterfly (calendar-preserving,
    single-expiry) arbitrage requires q(K) >= 0 everywhere. Rather than
    trusting a memorized closed-form arbitrage condition on the SVI
    parameters directly, this prices calls off the fitted smile with our own
    (already-tested) Black-Scholes pricer across a strike grid and takes a
    finite-difference second derivative -- directly checking the thing that
    actually matters (price-curve convexity), reusing machinery we've
    already validated elsewhere in this project.
    """
    import bscpp  # local import: avoids vol_surface.py depending on bscpp at module load time

    forward = spot * np.exp((rate - dividend_yield) * svi.t)
    k = np.linspace(k_range[0], k_range[1], n_points)
    strikes = forward * np.exp(k)
    ivs = svi.implied_vol(k)

    inputs = [
        bscpp.make_inputs(spot, float(K), rate, float(iv), svi.t, "call", dividend_yield)
        for K, iv in zip(strikes, ivs)
    ]
    prices = np.array([r.price for r in bscpp.bs_price_with_greeks_batch(inputs)])

    # non-uniform-grid second derivative (strikes are uniform in log-space,
    # not in strike-space, since k is the uniform variable above)
    density = np.full(n_points, np.nan)
    for i in range(1, n_points - 1):
        h1 = strikes[i] - strikes[i - 1]
        h2 = strikes[i + 1] - strikes[i]
        d2c_dk2 = 2.0 * (
            prices[i - 1] / (h1 * (h1 + h2)) - prices[i] / (h1 * h2) + prices[i + 1] / (h2 * (h1 + h2))
        )
        density[i] = np.exp(rate * svi.t) * d2c_dk2

    interior = density[1:-1]
    min_density = float(np.min(interior))
    return {
        "min_density": min_density,
        "arbitrage_free": bool(min_density >= -1e-6),  # small numerical tolerance
        "strikes": strikes,
        "density": density,
    }
