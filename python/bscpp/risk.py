"""Portfolio-level Greeks aggregation and limit checking.

Scope, stated plainly: this is net-Greeks aggregation across positions with
configurable limits and breach flags. It is NOT a production risk system --
there is no real-time market data feed, no margin/capital model, no kill
switch, no scenario/stress-testing engine, and no persistence. Building
those is a materially different (and much larger) undertaking than a
personal pricing/research project should claim to have solved. What this
module *does* do honestly: given a list of positions (each with its own
underlying, spot, and market inputs -- a real portfolio spans more than one
name), it computes true DOLLAR delta and gamma per position and in
aggregate, groups by underlying, and flags configurable limit breaches.
That is a genuine, useful building block; it is not a trading desk's risk
system.

Dollar, not share-equivalent: 100 shares of raw delta means something very
different on a $20 stock than a $2000 one, so a cross-underlying limit
checked in raw delta*quantity units (this project's own
StrategyPricer/StrategyResult convention, correct there because a single
strategy never mixes underlyings) doesn't actually aggregate risk
meaningfully across names -- exactly the case this module exists for. See
Position.greeks for the exact convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import bscpp


@dataclass
class Position:
    label: str
    instrument: str  # "call", "put", or "stock"
    quantity: float  # positive = long, negative = short
    underlying: str  # grouping key, e.g. a ticker
    spot: float
    rate: float
    dividend_yield: float = 0.0
    strike: float | None = None  # required for call/put
    vol: float | None = None  # required for call/put
    maturity: float | None = None  # required for call/put

    def greeks(self) -> dict:
        """True dollar Greeks for this position -- comparable across
        underlyings of different spot prices, which is the entire point
        of a portfolio-level limit (100 shares of raw delta is a very
        different risk on a $20 stock than a $2000 one).

        dollar_delta = delta * quantity * spot -- the $ notional of stock
        this position currently behaves like. For a stock position delta
        is identically 1, so this collapses to quantity * spot, i.e. its
        own market value, exactly as it should.

        dollar_gamma = gamma * quantity * spot^2 / 100 -- the standard
        desk convention for "the $ change in dollar delta for a 1% move
        in the underlying," NOT raw gamma*quantity (this project's
        StrategyPricer uses the raw, per-share convention instead,
        correctly, since a single strategy never mixes underlyings and
        so has no cross-name scale mismatch to correct for).

        vega/theta/rho need no such correction: each is already a $
        price sensitivity per contract (per 1.00 vol, per year, per 1.00
        rate), so quantity alone dollarizes them correctly regardless of
        the underlying's spot price.
        """
        if self.instrument == "stock":
            return {
                "price": self.spot * self.quantity,
                "delta": self.spot * self.quantity,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }
        if self.strike is None or self.vol is None or self.maturity is None:
            raise ValueError(f"position {self.label!r}: strike/vol/maturity required for options")

        inputs = bscpp.make_inputs(self.spot, self.strike, self.rate, self.vol, self.maturity,
                                    self.instrument, self.dividend_yield)
        result = bscpp.bs_price_with_greeks(inputs)
        return {
            "price": result.price * self.quantity,
            "delta": result.greeks.delta * self.quantity * self.spot,
            "gamma": result.greeks.gamma * self.quantity * self.spot ** 2 / 100.0,
            "vega": result.greeks.vega * self.quantity,
            "theta": result.greeks.theta * self.quantity,
            "rho": result.greeks.rho * self.quantity,
        }


@dataclass
class RiskLimits:
    """Any field left as None is not checked. Portfolio-level limits apply
    to the sum across all positions; per_underlying_* limits apply to each
    underlying's own subtotal independently."""
    max_abs_delta: float | None = None
    max_abs_gamma: float | None = None
    max_abs_vega: float | None = None
    max_abs_theta: float | None = None
    per_underlying_max_abs_delta: float | None = None
    per_underlying_max_abs_vega: float | None = None


@dataclass
class Breach:
    scope: str  # "portfolio" or an underlying label
    metric: str
    value: float
    limit: float

    def __repr__(self) -> str:
        return f"Breach({self.scope}: |{self.metric}|={abs(self.value):.4f} > {self.limit:.4f})"


class PortfolioRiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def position_greeks(self, positions: list[Position]) -> pd.DataFrame:
        rows = []
        for pos in positions:
            g = pos.greeks()
            rows.append({"label": pos.label, "underlying": pos.underlying,
                         "instrument": pos.instrument, "quantity": pos.quantity, **g})
        return pd.DataFrame(rows)

    def net_greeks(self, positions: list[Position]) -> dict:
        df = self.position_greeks(positions)
        if df.empty:
            return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        return {k: float(df[k].sum()) for k in ["price", "delta", "gamma", "vega", "theta", "rho"]}

    def greeks_by_underlying(self, positions: list[Position]) -> pd.DataFrame:
        df = self.position_greeks(positions)
        if df.empty:
            return df
        return df.groupby("underlying")[["price", "delta", "gamma", "vega", "theta", "rho"]].sum().reset_index()

    def check_limits(self, positions: list[Position]) -> list[Breach]:
        breaches: list[Breach] = []
        net = self.net_greeks(positions)
        checks = [
            ("max_abs_delta", "delta"), ("max_abs_gamma", "gamma"),
            ("max_abs_vega", "vega"), ("max_abs_theta", "theta"),
        ]
        for limit_name, metric in checks:
            limit = getattr(self.limits, limit_name)
            if limit is not None and abs(net[metric]) > limit:
                breaches.append(Breach("portfolio", metric, net[metric], limit))

        if self.limits.per_underlying_max_abs_delta is not None or \
                self.limits.per_underlying_max_abs_vega is not None:
            by_name = self.greeks_by_underlying(positions)
            for _, row in by_name.iterrows():
                if self.limits.per_underlying_max_abs_delta is not None and \
                        abs(row["delta"]) > self.limits.per_underlying_max_abs_delta:
                    breaches.append(Breach(row["underlying"], "delta", row["delta"],
                                            self.limits.per_underlying_max_abs_delta))
                if self.limits.per_underlying_max_abs_vega is not None and \
                        abs(row["vega"]) > self.limits.per_underlying_max_abs_vega:
                    breaches.append(Breach(row["underlying"], "vega", row["vega"],
                                            self.limits.per_underlying_max_abs_vega))
        return breaches
