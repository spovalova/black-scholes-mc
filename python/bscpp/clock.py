"""Explicit day-count/time convention, so "365" or "252" never appears as
an unexamined magic number scattered across the codebase.

Two conventions: ACT/365 (calendar days elapsed / 365, this project's
default) and TRADING/252 (trading days elapsed / 252, the other common
industry convention). Which is "right" depends on what's being measured:
option time decay runs on calendar time (an option loses value over a
weekend even though nothing trades), which is why ACT/365 is the default
for time-to-expiry and for the hedging backtester's financing/theta
accrual; realized vol is sometimes instead annualized on a trading-day
count, on the theory that volatility is driven by trading activity, not
calendar time.

This project's own history has a concrete cautionary tale for why picking
a convention implicitly per call site, rather than once and explicitly,
is a real risk: a validation study once scaled trading-day return std by
sqrt(365) while the backtester it was validating accrued calendar time,
silently inflating a reported result ~20% (see CHANGELOG). One Clock
object, threaded explicitly through every vol estimator and every time-
to-expiry calculation, is how that class of bug gets structurally
prevented rather than caught by luck a second time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_CONVENTIONS = ("ACT/365", "TRADING/252")


@dataclass(frozen=True)
class Clock:
    """convention: "ACT/365" (calendar days / 365, default) or
    "TRADING/252" (trading days / 252)."""

    convention: str = "ACT/365"

    def __post_init__(self):
        if self.convention not in _CONVENTIONS:
            raise ValueError(f"unknown clock convention: {self.convention!r}, "
                              f"must be one of {_CONVENTIONS}")

    def year_fraction(self, start, end) -> float:
        """Time between two dates/Timestamps, in years, under this
        convention. TRADING/252 approximates trading days elapsed as 5/7
        of calendar days -- a full trading-day calendar (weekends,
        exchange holidays) is out of scope here, see the README's scope
        statement; this is the same approximation ACT/365 vs. trading-day
        annualization comparisons in this project have always implicitly
        made, just named and made explicit rather than assumed."""
        calendar_days = (end - start).days
        if self.convention == "ACT/365":
            return calendar_days / 365.0
        return calendar_days * (5.0 / 7.0) / 252.0

    def elapsed(self, start, end, floor_at_one_day: bool = False) -> float:
        """max(0, year_fraction(start, end)), optionally floored at one
        day if that would otherwise be exactly zero -- e.g. duplicate
        consecutive dates in a price series, or an expiration falling on
        the pricing date itself, both of which need a strictly-positive
        step to price/accrue against."""
        t = max(self.year_fraction(start, end), 0.0)
        if floor_at_one_day and t <= 0.0:
            return 1.0 / self.days_per_year
        return t

    def time_to_expiry(self, as_of, expiration, floor_at_one_day: bool = False) -> float:
        """elapsed(as_of, expiration, ...) under a name that self-
        documents intent at call sites that specifically mean "time to
        this option's expiry."""
        return self.elapsed(as_of, expiration, floor_at_one_day)

    @property
    def days_per_year(self) -> float:
        return 365.0 if self.convention == "ACT/365" else 252.0

    def annualized_realized_vol(self, closes: pd.Series) -> float:
        """Realized vol from a price series, annualized under this
        clock's convention: sqrt(sum(log_returns^2) / elapsed_years).
        NaN if fewer than 2 usable observations or zero/negative elapsed
        time."""
        closes = closes.dropna()
        log_returns = np.log(closes / closes.shift(1)).dropna()
        if len(log_returns) < 2:
            return float("nan")
        elapsed_years = self.year_fraction(closes.index[0], closes.index[-1])
        if elapsed_years <= 0:
            return float("nan")
        return math.sqrt(float(np.sum(log_returns.to_numpy() ** 2)) / elapsed_years)
