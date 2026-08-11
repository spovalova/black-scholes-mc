"""Multi-leg options strategy construction and analysis.

A "strip" in options trading has two meanings, and this project now covers
both:

  1. A slice of a chain across strikes at one expiry -- what
     `backtest.StripPricer` prices.
  2. The literal strategy: long 1 call + long 2 puts at the same strike and
     expiry (a bearish-biased volatility play, twin to the "strap": long 2
     calls + 1 put, bullish-biased). Built here alongside straddles,
     strangles, and vertical spreads -- the standard combinations that make
     "pricing a strip" mean something beyond a single-leg quote.

Every constructor returns a plain list[Leg]; `StrategyPricer` is what turns
a leg list into net Greeks, entry cost, and an expiration payoff diagram
with exact breakevens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import bscpp


@dataclass
class Leg:
    instrument: str  # "call", "put", or "stock"
    quantity: float  # positive = long, negative = short
    strike: float | None = None  # None for a stock leg


# --- strategy constructors -------------------------------------------------

def long_option(option_type: str, strike: float, quantity: float = 1.0) -> list[Leg]:
    return [Leg(option_type, quantity, strike)]


def straddle(strike: float, quantity: float = 1.0) -> list[Leg]:
    return [Leg("call", quantity, strike), Leg("put", quantity, strike)]


def strangle(call_strike: float, put_strike: float, quantity: float = 1.0) -> list[Leg]:
    if not put_strike < call_strike:
        raise ValueError("a strangle needs put_strike < call_strike")
    return [Leg("call", quantity, call_strike), Leg("put", quantity, put_strike)]


def vertical_spread(
    option_type: str, long_strike: float, short_strike: float, quantity: float = 1.0
) -> list[Leg]:
    return [Leg(option_type, quantity, long_strike), Leg(option_type, -quantity, short_strike)]


def strip(strike: float, quantity: float = 1.0) -> list[Leg]:
    """Long 1 call + long 2 puts at the same strike -- bearish-biased vol play."""
    return [Leg("call", quantity, strike), Leg("put", 2.0 * quantity, strike)]


def strap(strike: float, quantity: float = 1.0) -> list[Leg]:
    """Long 2 calls + long 1 put at the same strike -- bullish-biased vol play."""
    return [Leg("call", 2.0 * quantity, strike), Leg("put", quantity, strike)]


def butterfly(option_type: str, low_strike: float, mid_strike: float, high_strike: float,
              quantity: float = 1.0) -> list[Leg]:
    if not low_strike < mid_strike < high_strike:
        raise ValueError("butterfly needs low_strike < mid_strike < high_strike")
    return [
        Leg(option_type, quantity, low_strike),
        Leg(option_type, -2.0 * quantity, mid_strike),
        Leg(option_type, quantity, high_strike),
    ]


# --- pricing / analysis ----------------------------------------------------

@dataclass
class StrategyResult:
    net_price: float
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    net_rho: float
    legs: pd.DataFrame = field(repr=False)


class StrategyPricer:
    """Prices a leg list today (net Greeks) and its expiration payoff."""

    def __init__(self, rate: float, dividend_yield: float = 0.0):
        self.rate = rate
        self.dividend_yield = dividend_yield

    def price(self, legs: list[Leg], spot: float, vol, maturity: float) -> StrategyResult:
        """vol is either a flat float, or a {strike: vol} dict for a skew-aware price
        (e.g. sourced from a fitted SVI slice's implied_vol at each leg's strike)."""
        rows = []
        for leg in legs:
            if leg.instrument == "stock":
                rows.append({
                    "instrument": "stock", "strike": None, "quantity": leg.quantity,
                    "price": spot * leg.quantity, "delta": leg.quantity,
                    "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
                })
                continue

            leg_vol = vol[leg.strike] if isinstance(vol, dict) else vol
            inputs = bscpp.make_inputs(spot, leg.strike, self.rate, leg_vol, maturity,
                                        leg.instrument, self.dividend_yield)
            result = bscpp.bs_price_with_greeks(inputs)
            rows.append({
                "instrument": leg.instrument, "strike": leg.strike, "quantity": leg.quantity,
                "price": result.price * leg.quantity,
                "delta": result.greeks.delta * leg.quantity,
                "gamma": result.greeks.gamma * leg.quantity,
                "vega": result.greeks.vega * leg.quantity,
                "theta": result.greeks.theta * leg.quantity,
                "rho": result.greeks.rho * leg.quantity,
            })

        legs_df = pd.DataFrame(rows)
        return StrategyResult(
            net_price=legs_df["price"].sum(),
            net_delta=legs_df["delta"].sum(),
            net_gamma=legs_df["gamma"].sum(),
            net_vega=legs_df["vega"].sum(),
            net_theta=legs_df["theta"].sum(),
            net_rho=legs_df["rho"].sum(),
            legs=legs_df,
        )

    @staticmethod
    def payoff_at_expiration(legs: list[Leg], spot_at_expiry: float) -> float:
        total = 0.0
        for leg in legs:
            if leg.instrument == "stock":
                total += leg.quantity * spot_at_expiry
            elif leg.instrument == "call":
                total += leg.quantity * max(spot_at_expiry - leg.strike, 0.0)
            else:
                total += leg.quantity * max(leg.strike - spot_at_expiry, 0.0)
        return total

    def payoff_diagram(
        self, legs: list[Leg], spot: float, vol, maturity: float,
        spot_range: tuple[float, float] = (0.5, 1.5), n_points: int = 200,
    ) -> tuple[pd.DataFrame, list[float]]:
        """Returns (plot_df, breakevens). Breakevens are computed exactly: the
        expiration payoff of a portfolio of vanilla options/stock is piecewise
        LINEAR in spot with kinks only at each leg's strike, so evaluating at
        the strikes plus the range endpoints and linearly interpolating for
        sign changes gives exact roots -- no numerical root-finder needed.
        """
        entry_cost = self.price(legs, spot, vol, maturity).net_price

        lo, hi = spot * spot_range[0], spot * spot_range[1]
        plot_grid = np.linspace(lo, hi, n_points)
        payoffs = np.array([self.payoff_at_expiration(legs, s) for s in plot_grid])
        pnl = payoffs - entry_cost
        plot_df = pd.DataFrame({"spot": plot_grid, "payoff_at_expiry": payoffs, "pnl_at_expiry": pnl})

        strikes = sorted({leg.strike for leg in legs if leg.strike is not None})
        kink_points = sorted(set([lo, hi] + [k for k in strikes if lo <= k <= hi]))
        kink_pnl = [self.payoff_at_expiration(legs, s) - entry_cost for s in kink_points]

        breakevens = []
        for i in range(len(kink_points) - 1):
            p0, p1 = kink_pnl[i], kink_pnl[i + 1]
            if p0 == 0:
                breakevens.append(kink_points[i])
            elif p0 * p1 < 0:
                s0, s1 = kink_points[i], kink_points[i + 1]
                frac = -p0 / (p1 - p0)
                breakevens.append(s0 + frac * (s1 - s0))
        if kink_pnl[-1] == 0:
            breakevens.append(kink_points[-1])

        return plot_df, breakevens
