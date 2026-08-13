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

Known limitation: fitting slices independently gives NO guarantee they're
jointly consistent across expiries (no calendar-spread arbitrage). That
guarantee is exactly what Gatheral & Jacquier's SSVI (surface) extension of
this same paper exists to provide -- via a condition on how ATM total
variance and the smile's curvature must co-move across maturities. This
project implements single-slice SVI only; two slices fit independently by
`fit_svi_slice` at different expiries could still imply a calendar
arbitrage between them even if each individually passes
`svi_gatheral_jacquier_check`/`svi_butterfly_arbitrage_check`.
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

    def total_variance_derivative(self, log_moneyness):
        """w'(k) = b*(rho + u/s), u = k-m, s = sqrt(u^2+sigma^2)."""
        k = np.asarray(log_moneyness, dtype=float)
        u = k - self.m
        s = np.sqrt(u ** 2 + self.sigma ** 2)
        return self.b * (self.rho + u / s)

    def total_variance_second_derivative(self, log_moneyness):
        """w''(k) = b*sigma^2 / s^3."""
        k = np.asarray(log_moneyness, dtype=float)
        u = k - self.m
        s = np.sqrt(u ** 2 + self.sigma ** 2)
        return self.b * self.sigma ** 2 / s ** 3


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


def svi_g_function(svi: SVISlice, log_moneyness) -> np.ndarray:
    """Gatheral & Jacquier (2013) g(k): a slice is butterfly-arbitrage-free
    iff g(k) >= 0 for all real k (plus a tail condition standard SVI with
    b>0 satisfies automatically -- see svi_gatheral_jacquier_check).

        g(k) = (1 - k*w'(k)/(2*w(k)))^2 - (w'(k)^2/4)*(1/w(k) + 1/4) + w''(k)/2

    This is the closed-form counterpart to svi_butterfly_arbitrage_check's
    numerical Breeden-Litzenberger scan: same underlying no-arbitrage
    condition (Gatheral-Jacquier derive g(k) by reparametrizing the exact
    same density-positivity requirement from (strike, price) into
    (log-moneyness, total-variance) coordinates), but here it's ~2 orders of
    magnitude cheaper to evaluate and has no finite-difference noise floor --
    at the cost of trusting this formula's transcription is correct, which
    is exactly why both checks exist side by side rather than picking one.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = svi.total_variance(k)
    wp = svi.total_variance_derivative(k)
    wpp = svi.total_variance_second_derivative(k)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp ** 2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def svi_gatheral_jacquier_check(
    svi: SVISlice, k_range: tuple[float, float] = (-4.0, 4.0), n_points: int = 2000
) -> dict:
    """Closed-form no-butterfly-arbitrage check via g(k) >= 0.

    Cheap enough to scan finely (default 2000 points) for the true global
    minimum, unlike the numerical density check which is limited by pricer
    calls per grid point. Standard SVI with b > 0 automatically satisfies
    the paper's tail condition (lim_{k->inf} d+(k) = -inf), so g(k) >= 0
    everywhere is the complete butterfly-arbitrage-free criterion here.
    """
    k = np.linspace(k_range[0], k_range[1], n_points)

    # The g(k) >= 0 criterion is only meaningful where total variance
    # w(k) > 0: g(k) divides by w, so its value on a negative-variance
    # slice is garbage. A slice with negative total variance ANYWHERE is
    # already an outright arbitrage (negative implied variance) regardless
    # of what g(k) evaluates to, so check the closed-form global minimum
    # of w first.
    min_w = svi_min_total_variance(svi)
    w_grid = svi.total_variance(k)
    min_w_on_grid = float(np.min(w_grid))
    if min_w < 0.0 or min_w_on_grid <= 0.0:
        return {
            "min_g": float("nan"),
            "arbitrage_free": False,
            "reason": "negative_total_variance",
            "min_total_variance": float(min(min_w, min_w_on_grid)),
            "k_grid": k,
            "g_values": np.full_like(k, np.nan),
        }

    g = svi_g_function(svi, k)
    min_g = float(np.min(g))
    return {"min_g": min_g, "arbitrage_free": bool(min_g >= 0.0),
            "min_total_variance": float(min_w), "k_grid": k, "g_values": g}


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

    Cross-referenced against svi_gatheral_jacquier_check's closed-form g(k)
    (added after both were independently verified to agree on known
    arbitrage-free and arbitrage-violating test slices): this numerical
    check's tolerance is on absolute density, which is scale-dependent --
    on very short-dated slices it can register a technically-negative value
    of ~1e-13 that's pure finite-difference noise, saved only by the fixed
    -1e-6 tolerance below happening to cover it. Prefer
    svi_gatheral_jacquier_check for a precise verdict; keep this one as an
    independent cross-check (same reasoning as HestonMCPricer existing
    alongside HestonPricer) rather than the sole source of truth.
    """
    import bscpp  # local import: avoids vol_surface.py depending on bscpp at module load time

    forward = spot * np.exp((rate - dividend_yield) * svi.t)
    k = np.linspace(k_range[0], k_range[1], n_points)

    # Same precondition as svi_gatheral_jacquier_check: negative total
    # variance anywhere is already an arbitrage (and implied_vol() would
    # silently clamp it to 0, masking the violation from the density scan).
    min_w = svi_min_total_variance(svi)
    if min_w < 0.0 or float(np.min(svi.total_variance(k))) <= 0.0:
        return {
            "min_density": float("nan"),
            "arbitrage_free": False,
            "reason": "negative_total_variance",
            "min_total_variance": float(min_w),
            "strikes": forward * np.exp(k),
            "density": np.full(n_points, np.nan),
        }

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
