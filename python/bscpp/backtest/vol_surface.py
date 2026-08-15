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

Two fitters: `fit_svi_slice` (plain 5-parameter nonlinear least squares --
simple, but a 5D nonlinear search can land in a bad local optimum
depending on its initial guess) and `fit_svi_slice_quasi_explicit`
(Zeliade Systems 2009's method: reduces the search to 2 nonlinear
parameters (m, sigma) by solving the remaining three (a, b, rho) exactly
-- a closed-form linear system -- at every candidate, with optional
vega weighting so the fit reflects which strikes actually matter for
pricing/hedging). Kept as a separate function, not a replacement,
matching this project's established pattern (HestonPricer::price vs.
price_cos, HestonMCPricer.price vs. price_qe) of adding a cross-checked
alternative rather than swapping the existing one out from under callers.

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
from scipy.optimize import least_squares, minimize


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
    initial_guess: list[float] | None = None,
) -> SVISlice:
    """Least-squares fit an SVI slice to (strike, implied vol) pairs at one expiry.

    `initial_guess` (a,b,rho,m,sigma) overrides the default starting point
    -- exposed mainly so fit_svi_slice_quasi_explicit's test suite can
    demonstrate, not just assert, that THIS full 5-parameter nonlinear
    fit is sensitive to where it starts (a real local-minima risk any
    5D nonlinear least-squares carries), in exactly the way the quasi-
    explicit method's 2D-outer/convex-inner split isn't.
    """
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

    x0 = initial_guess if initial_guess is not None else [max(w_obs.min(), 1e-4), 0.1, -0.3, 0.0, 0.1]
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


_SVI_SIGMA_FLOOR = 1e-4


def _svi_conditional_linear_fit(y, w_obs, weights):
    """Given y=(k-m)/sigma for a FIXED (m, sigma), solve the weighted
    linear least-squares sub-problem for (c1, c2, c3) in

        w(k) = c1 + c2*y + c3*sqrt(y^2+1)

    the reparametrization fit_svi_slice_quasi_explicit uses to turn SVI's
    5-parameter fit into a 2-parameter outer search: a=c1, b=c3/sigma,
    rho=c2/c3 once sigma is reintroduced by the caller. Linear in
    (c1,c2,c3) for any fixed (m,sigma), so this has a closed-form (matrix
    pseudo-inverse) solution -- no iteration, no initial-guess sensitivity.

    That closed-form answer isn't always a valid SVI slice, though (needs
    c3>=0, |c2|<=c3 i.e. |rho|<=1, and non-negative total variance at the
    minimum, a+b*sigma*sqrt(1-rho^2) = c1+sqrt(c3^2-c2^2) >= 0 -- see
    svi_min_total_variance for the same condition in (a,b,rho) form).
    Falls back to a constrained convex optimization when it isn't --
    "quasi" explicit, not always fully explicit, the same qualification
    the Zeliade paper itself gives the method. The fallback is still
    initial-guess-insensitive in the sense that matters (feasible region
    and objective are both convex, so any reasonable start converges to
    the same global optimum) even though it's technically iterative.

    Returns (c1, c2, c3, weighted_sse).
    """
    X = np.column_stack([np.ones_like(y), y, np.sqrt(y ** 2 + 1.0)])
    sqrt_w = np.sqrt(weights)
    coeffs, *_ = np.linalg.lstsq(X * sqrt_w[:, None], w_obs * sqrt_w, rcond=None)
    c1, c2, c3 = coeffs

    feasible = c3 >= 0.0 and abs(c2) <= c3 and c1 + np.sqrt(max(c3 ** 2 - c2 ** 2, 0.0)) >= 0.0
    if not feasible:
        def objective(p):
            resid = X @ p - w_obs
            return float(np.sum(weights * resid ** 2))

        c3_guess = max(abs(c3), 1e-3)
        constraints = [
            {"type": "ineq", "fun": lambda p: p[2]},                                  # c3 >= 0
            {"type": "ineq", "fun": lambda p: p[2] - p[1]},                            # c3 >= c2
            {"type": "ineq", "fun": lambda p: p[2] + p[1]},                            # c3 >= -c2
            {"type": "ineq", "fun": lambda p: p[0] +
                np.sqrt(np.maximum(p[2] ** 2 - p[1] ** 2, 0.0))},                      # min w >= 0
        ]
        result = minimize(objective, x0=[c1, np.clip(c2, -c3_guess, c3_guess), c3_guess],
                           method="SLSQP", constraints=constraints,
                           options={"maxiter": 200, "ftol": 1e-14})
        c1, c2, c3 = result.x

    weighted_sse = float(np.sum(weights * (X @ np.array([c1, c2, c3]) - w_obs) ** 2))
    return c1, c2, c3, weighted_sse


def fit_svi_slice_quasi_explicit(
    strikes,
    market_ivs,
    spot: float,
    t_years: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    vega_weighted: bool = True,
    m_grid_size: int = 9,
    sigma_grid_size: int = 9,
) -> SVISlice:
    """Zeliade Systems' "quasi-explicit" SVI calibration (Zeliade Systems,
    2009, "Quasi-Explicit Calibration of Gatheral's SVI Model") -- reduces
    fit_svi_slice's 5-parameter nonlinear least-squares to a 2-parameter
    (m, sigma) search, with (a, b, rho) solved EXACTLY (a closed-form
    linear system, not iterated) at every candidate -- see
    _svi_conditional_linear_fit for the reparametrization this relies on.
    A 5D nonlinear search can get stuck in a bad local optimum depending
    on where it starts; the inner (a,b,rho) problem here can't, in the
    same way, since it's convex and doesn't depend on an initial guess.
    Only the OUTER (m,sigma) search is still nonlinear, but 2-dimensional
    instead of 5, and every candidate it tries is scored by the inner
    problem's exact optimum rather than a partial descent step -- a grid
    search over (m,sigma) followed by a local (Nelder-Mead) refine from
    the best grid point.

    `vega_weighted` (default True): weights each strike's (variance-
    space) residual by vega^2 -- vega being d(BS price)/d(vol), the
    actual price sensitivity to an IV error at that strike -- so the fit
    reflects which strikes' IV errors matter most for pricing/hedging
    downstream. An unweighted total-variance fit treats a 1-vol-point
    error at a near-zero-vega deep OTM strike the same as at the vega-
    heavy ATM strike, which is backwards for anything that prices off the
    fitted smile afterward. Set False to recover a plain unweighted fit,
    e.g. to isolate the weighting's effect from the reparametrization's.

    `m_grid_size`/`sigma_grid_size` (default 9x9=81 candidates): measured,
    not assumed, that the final RMSE this function returns is INSENSITIVE
    to grid density from 5x5 up through 21x21 on every scenario tested --
    the local (Nelder-Mead) refine step from the best grid point does
    essentially all of the real work, so a dense grid mostly just adds
    cost (a 21x21 grid measured up to ~30x slower than a 5x5 one for
    identical RMSE on one test scenario). 9x9 is a deliberately modest
    default kept above the measured-sufficient floor as a safety margin
    for real market data less well-behaved than what was tested here, not
    a value that was itself found necessary.

    Verified in test_vol_surface.py against fit_svi_slice (this module's
    existing full 5D nonlinear fit) on the same synthetic and realistic
    data: matching or better RMSE, and -- the actual point of this method,
    not just a nice-to-have -- INSENSITIVE to a battery of deliberately
    bad fit_svi_slice initial guesses that measurably degrade the 5D fit's
    quality.
    """
    import bscpp  # local import, matches svi_butterfly_arbitrage_check above

    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    valid = np.isfinite(market_ivs) & (market_ivs > 0) & np.isfinite(strikes) & (strikes > 0)
    strikes, market_ivs = strikes[valid], market_ivs[valid]
    if strikes.size < 6:
        raise ValueError("need at least 6 valid (strike, iv) points to fit an SVI slice")

    forward = spot * np.exp((rate - dividend_yield) * t_years)
    k = np.log(strikes / forward)
    w_obs = market_ivs ** 2 * t_years

    if vega_weighted:
        n = strikes.size
        otype_arr = np.zeros(n, dtype=np.int32)  # vega is call/put-symmetric; type doesn't matter
        _, _, _, vega, _, _ = bscpp.bs_price_with_greeks_batch_arrays(
            np.full(n, spot), strikes, np.full(n, rate), np.full(n, dividend_yield), market_ivs,
            np.full(n, t_years), otype_arr)
        weights = np.maximum(vega, 1e-12) ** 2
        weights = weights / np.mean(weights)  # keep the SSE scale comparable across slices
    else:
        weights = np.ones_like(k)

    def inner_sse(m: float, sigma: float):
        sigma = max(sigma, _SVI_SIGMA_FLOOR)
        y = (k - m) / sigma
        c1, c2, c3, sse = _svi_conditional_linear_fit(y, w_obs, weights)
        return sse, (c1, c2, c3)

    k_range = float(k.max() - k.min()) or 1.0
    m_candidates = np.linspace(k.min() - 0.25 * k_range, k.max() + 0.25 * k_range, m_grid_size)
    sigma_candidates = np.geomspace(_SVI_SIGMA_FLOOR, max(2.0 * k_range, 0.5), sigma_grid_size)

    best_sse, best_m, best_sigma = np.inf, float(m_candidates[0]), float(sigma_candidates[0])
    for m in m_candidates:
        for sigma in sigma_candidates:
            sse, _ = inner_sse(float(m), float(sigma))
            if sse < best_sse:
                best_sse, best_m, best_sigma = sse, float(m), float(sigma)

    # Local refine in log(sigma) (sigma is a positive scale parameter --
    # searching its log keeps step sizes meaningful across orders of
    # magnitude) via Nelder-Mead: derivative-free, since the constrained
    # fallback inside _svi_conditional_linear_fit can make inner_sse's
    # dependence on (m,sigma) non-smooth right at the constraint boundary.
    refined = minimize(lambda p: inner_sse(p[0], float(np.exp(p[1])))[0],
                        x0=[best_m, np.log(best_sigma)], method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500})
    m_star = float(refined.x[0])
    sigma_star = max(float(np.exp(refined.x[1])), _SVI_SIGMA_FLOOR)

    _, (c1, c2, c3) = inner_sse(m_star, sigma_star)
    b = c3 / sigma_star
    rho = float(np.clip(c2 / c3, -0.999, 0.999)) if c3 > 1e-10 else 0.0
    return SVISlice(a=c1, b=b, rho=rho, m=m_star, sigma=sigma_star, t=t_years)
