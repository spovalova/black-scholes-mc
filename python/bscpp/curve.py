"""Minimal piecewise-flat zero-rate discount curve.

Every pricer in this project's C++ core ultimately needs a single scalar
risk-free rate for one specific maturity -- Black-Scholes, Heston, and LSM
are all single-maturity, single-rate closed-form/semi-analytic formulas,
with no term structure inside them by design, matching how those models
are actually specified in the literature. A real desk doesn't get that
scalar from one hardcoded constant, though -- it comes from a curve built
off observable instruments (SOFR/OIS, T-bills) at the SPECIFIC maturity
being priced, and different maturities in the same chain can legitimately
see different rates.

This module is the minimal version of that gap, not a curve-bootstrapping
engine: given a handful of (tenor, zero_rate) pillars, `ZeroCurve.zero_rate
(t)` / `.df(t)` return the maturity-appropriate rate/discount factor.
Building a curve FROM real market instruments (bootstrapping off SOFR
futures, bond yields, etc.) is a materially larger, different undertaking
-- out of scope here, see the README's scope statement. What this DOES
fix: every pricer/hedger/calibration entry point in this project now
accepts a ZeroCurve (or a bare float, still supported, for an explicitly
flat curve) via `resolve_rate` instead of a silent `rate=0.05` default --
the point isn't sophisticated curve construction, it's that "0.05" stops
being able to hide as an unexamined default no caller had to think about.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ZeroCurve:
    """Piecewise-flat, continuously-compounded zero-rate curve.

    tenors: strictly increasing maturities in years. rates: the
    continuously-compounded zero rate effective at and after each tenor,
    up to (not including) the next one -- a STEP function, not
    interpolated, deliberately: with only a handful of real pillars (a
    few OIS/SOFR tenors), interpolating between them implies more
    precision than the inputs actually carry. zero_rate(t) for t before
    the first tenor uses the first rate; for t after the last tenor, the
    last rate (flat extrapolation at both ends).
    """

    tenors: tuple
    rates: tuple

    def __post_init__(self):
        if len(self.tenors) != len(self.rates):
            raise ValueError("tenors and rates must be the same length")
        if len(self.tenors) == 0:
            raise ValueError("ZeroCurve needs at least one (tenor, rate) pillar")
        if list(self.tenors) != sorted(self.tenors):
            raise ValueError("tenors must be strictly increasing")

    @classmethod
    def flat(cls, rate: float) -> "ZeroCurve":
        """A single-rate curve -- the explicit way to say 'deliberately
        not modeling a term structure here', as opposed to a silent 0.05
        default nobody had to acknowledge choosing."""
        return cls(tenors=(0.0,), rates=(float(rate),))

    def zero_rate(self, t: float) -> float:
        if t <= self.tenors[0]:
            return self.rates[0]
        idx = bisect.bisect_right(self.tenors, t) - 1
        return self.rates[idx]

    def df(self, t: float) -> float:
        """Discount factor to maturity t, exp(-zero_rate(t) * t)."""
        return math.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate between t1 and t2 implied
        by this curve's own discount factors -- df(t2)/df(t1) = exp(-f*(t2-t1))."""
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        return -math.log(self.df(t2) / self.df(t1)) / (t2 - t1)


def resolve_rate(rate, t_years: float) -> float:
    """Accepts either a bare float (a flat rate, kept for convenience and
    backward compatibility) or a ZeroCurve, returning the scalar zero
    rate applicable at this specific maturity either way. Every rate=
    parameter across StripPricer, HedgingBacktester, and calibrate_heston
    goes through this before reaching the (single-rate) C++ pricers.
    """
    if isinstance(rate, ZeroCurve):
        return rate.zero_rate(t_years)
    return float(rate)
