"""Rebalancing policies for the delta-hedging backtester.

Where the hedge *ratio* comes from Black-Scholes, the hedge *timing* is a
policy choice with real P&L consequences once transaction costs exist:
rebalancing daily to the exact delta pays the spread on every small move,
while never rebalancing carries uncompensated gamma risk. The transaction-
cost literature closed this question asymptotically decades ago -- and
essentially no open implementation exists -- so this module provides the
standard ladder of policies, from the naive baseline up to the
Whalley-Wilmott asymptotically-optimal band:

- `DeltaPolicy`: rebalance to the exact BS delta on every observation
  (the textbook baseline every study should report but not stop at).
- `BandPolicy`: rebalance only when |delta - held| exceeds a FIXED band;
  on breach, trade to the nearest band edge (not to the center -- trading
  to the edge is the cost-minimizing correction, and matches the optimal
  policies' qualitative shape).
- `WhalleyWilmottPolicy`: the small-cost asymptotic optimum of the
  Hodges-Neuberger utility framework (Whalley & Wilmott 1997, Mathematical
  Finance 7(3)): a no-trade band around the BS delta with half-width

      H = ( (3/2) * cost_frac * S * Gamma^2 * exp(-r*(T-t)) / lam )^(1/3)

  where cost_frac is the proportional cost (transaction_cost_bps/1e4) and
  lam ("risk aversion") sets the risk/cost trade-off. The famous
  qualitative content: the band scales with Gamma^(2/3) (hedge more
  tightly where convexity is high) and with cost^(1/3) (costs push you to
  tolerate more delta drift, but only with a cube-root sensitivity).

Every policy exposes one method:

    target_shares(held, state) -> float

returning the number of shares to hold AFTER this observation (equal to
`held` when the policy chooses not to trade). `state` is a HedgeState with
the current Greeks and market snapshot, so custom/learned policies can be
plugged in with the same interface (any object with `target_shares` works;
see CallablePolicy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HedgeState:
    """Market/position snapshot handed to a policy at one observation."""
    t: float            # time to expiry, years
    spot: float
    delta: float        # current BS delta at the marking vol
    gamma: float
    vega: float
    rate: float
    cost_frac: float    # proportional transaction cost (bps / 1e4)


class DeltaPolicy:
    """Rebalance to the exact model delta at every observation (baseline)."""

    def target_shares(self, held: float, state: HedgeState) -> float:
        return state.delta


class BandPolicy:
    """Fixed no-trade band: trade to the nearest band edge on breach.

    band: half-width in delta units (e.g. 0.05 = tolerate |drift| <= 5 deltas
    per unit option). A width of 0 reduces to DeltaPolicy.
    """

    def __init__(self, band: float):
        if band < 0:
            raise ValueError("band must be >= 0")
        self.band = band

    def target_shares(self, held: float, state: HedgeState) -> float:
        drift = held - state.delta
        if drift > self.band:
            return state.delta + self.band   # trade DOWN to the upper edge
        if drift < -self.band:
            return state.delta - self.band   # trade UP to the lower edge
        return held                          # inside the band: do nothing


class WhalleyWilmottPolicy:
    """Whalley-Wilmott (1997) asymptotically-optimal no-trade band.

    risk_aversion (lam): larger = tighter bands = closer tracking, more
    trading. The asymptotic optimum trades to the nearest band edge, never
    to the center. With zero transaction cost the band collapses and this
    reduces to DeltaPolicy, as it must.

    The formula's exp(-r*(T-t)) factor is retained from the paper; for
    short-dated hedges it is within a few percent of 1 and its omission is
    a common (harmless) simplification -- kept here so the implementation
    matches the published result rather than a folklore version of it.
    """

    def __init__(self, risk_aversion: float = 1.0):
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be > 0")
        self.risk_aversion = risk_aversion

    def band_half_width(self, state: HedgeState) -> float:
        if state.cost_frac <= 0.0:
            return 0.0
        return (
            1.5 * state.cost_frac * state.spot * state.gamma ** 2
            * math.exp(-state.rate * state.t) / self.risk_aversion
        ) ** (1.0 / 3.0)

    def target_shares(self, held: float, state: HedgeState) -> float:
        h = self.band_half_width(state)
        drift = held - state.delta
        if drift > h:
            return state.delta + h
        if drift < -h:
            return state.delta - h
        return held


class CallablePolicy:
    """Adapter: wrap any fn(held, state) -> target_shares as a policy.

    This is the plug-in point for custom or learned (e.g. RL) policies --
    they receive the same HedgeState the classical policies do, so
    comparisons against the classical ladder are apples-to-apples.
    """

    def __init__(self, fn):
        self.fn = fn

    def target_shares(self, held: float, state: HedgeState) -> float:
        return float(self.fn(held, state))
