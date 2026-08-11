"""Market data providers for the backtester.

A DataProvider's job is to hand back an underlying price and an option
chain slice as a plain pandas DataFrame with a fixed column contract:

    strike, type ('call'/'put'), bid, ask, last, volume, open_interest,
    implied_volatility (may be NaN)

`StripPricer` (see engine.py) only depends on that contract, so any
provider (Polygon, Tradier, a CSV file, ...) can be dropped in.
"""

from __future__ import annotations

import datetime as dt
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import requests

CHAIN_COLUMNS = [
    "strike",
    "type",
    "expiration",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
]


class DataProvider(ABC):
    @abstractmethod
    def get_underlying_price(self, ticker: str, as_of: dt.date | None = None) -> float:
        """Spot price of the underlying. as_of=None means the latest available price."""

    @abstractmethod
    def get_expirations(self, ticker: str) -> list[dt.date]:
        """Available option expiration dates for the underlying."""

    @abstractmethod
    def get_option_chain(
        self, ticker: str, expiration: dt.date, as_of: dt.date | None = None
    ) -> pd.DataFrame:
        """Option chain slice for one expiration, columns per CHAIN_COLUMNS."""

    @abstractmethod
    def get_price_history(self, ticker: str, start: dt.date, end: dt.date) -> pd.Series:
        """Daily close prices for the underlying over [start, end], indexed by date."""


class PolygonProvider(DataProvider):
    """Polygon.io-backed provider.

    Requires an "Options Starter" plan or higher for options chain data
    (the free tier only covers equities). Historical `as_of` snapshots
    require Polygon's paid historical options tier; on lower tiers,
    `as_of` is effectively ignored and you'll get the live/most-recent
    snapshot instead.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Polygon API key found. Set the POLYGON_API_KEY environment variable "
                "(or a .env file) or pass api_key= explicitly."
            )
        self.session = session or requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        resp = self.session.get(f"{self.BASE_URL}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_underlying_price(self, ticker: str, as_of: dt.date | None = None) -> float:
        if as_of is None:
            data = self._get(f"/v2/last/trade/{ticker}")
            return float(data["results"]["p"])
        data = self._get(f"/v1/open-close/{ticker}/{as_of.isoformat()}")
        return float(data["close"])

    def get_expirations(self, ticker: str) -> list[dt.date]:
        data = self._get(
            "/v3/reference/options/contracts",
            params={"underlying_ticker": ticker, "limit": 1000},
        )
        exps = sorted({c["expiration_date"] for c in data.get("results", [])})
        return [dt.date.fromisoformat(e) for e in exps]

    def get_option_chain(
        self, ticker: str, expiration: dt.date, as_of: dt.date | None = None
    ) -> pd.DataFrame:
        rows = []
        params = {"expiration_date": expiration.isoformat(), "limit": 250}
        data = self._get(f"/v3/snapshot/options/{ticker}", params=params)

        for contract in data.get("results", []):
            details = contract.get("details", {})
            quote = contract.get("last_quote") or {}
            last_trade = contract.get("last_trade") or {}
            day = contract.get("day") or {}
            rows.append(
                {
                    "strike": details.get("strike_price"),
                    "type": "call" if details.get("contract_type") == "call" else "put",
                    "expiration": details.get("expiration_date"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "last": last_trade.get("price"),
                    "volume": day.get("volume"),
                    "open_interest": contract.get("open_interest"),
                    "implied_volatility": contract.get("implied_volatility"),
                }
            )

        if not rows:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        return pd.DataFrame(rows)[CHAIN_COLUMNS]

    def get_price_history(self, ticker: str, start: dt.date, end: dt.date) -> pd.Series:
        # Daily aggregates (/v2/aggs) are available on Polygon's base equities
        # tier -- unlike the options chain endpoints above, this does not
        # require "Options Starter" or higher.
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        data = self._get(path, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        results = data.get("results", [])
        if not results:
            return pd.Series(dtype=float, name="close")

        idx = [
            dt.datetime.fromtimestamp(r["t"] / 1000, tz=dt.timezone.utc).date() for r in results
        ]
        closes = [r["c"] for r in results]
        return pd.Series(closes, index=pd.DatetimeIndex(idx), name="close")


class MockProvider(DataProvider):
    """Synthetic provider for exercising the pipeline without a data key.

    Generates a chain around a configurable spot with a simple volatility
    "smile" (parabolic in strike), so backtests show non-trivial pricing
    error even though nothing was fetched over the network. Useful for
    testing the plumbing and for learning before you wire up a real feed.
    """

    def __init__(
        self,
        spot: float = 100.0,
        base_vol: float = 0.20,
        smile_strength: float = 0.35,
        rate: float = 0.05,
        dividend_yield: float = 0.0,
        realized_vol: float | None = None,
        seed: int = 7,
    ):
        self.spot = spot
        self.base_vol = base_vol
        self.smile_strength = smile_strength
        self.rate = rate
        self.dividend_yield = dividend_yield
        # The underlying's *actual* realized diffusion vol, independent of
        # the (smiled) implied vol quoted in the option chain. Set this
        # different from base_vol to see the realized-vs-implied vol gap
        # show up as delta-hedging P&L in HedgingBacktester.
        self.realized_vol = realized_vol if realized_vol is not None else base_vol
        self._rng = np.random.default_rng(seed)

    def get_underlying_price(self, ticker: str, as_of: dt.date | None = None) -> float:
        return self.spot

    def get_expirations(self, ticker: str) -> list[dt.date]:
        today = dt.date.today()
        return [today + dt.timedelta(days=d) for d in (30, 60, 90)]

    def get_option_chain(
        self, ticker: str, expiration: dt.date, as_of: dt.date | None = None
    ) -> pd.DataFrame:
        import bscpp  # local import to avoid a hard dependency at module import time

        as_of = as_of or dt.date.today()
        t_years = max((expiration - as_of).days, 1) / 365.0

        strikes = np.round(self.spot * np.arange(0.7, 1.31, 0.025), 1)
        rows = []
        for strike in strikes:
            moneyness = strike / self.spot - 1.0
            market_iv = max(self.base_vol + self.smile_strength * moneyness**2, 0.01)

            for otype in ("call", "put"):
                inputs = bscpp.make_inputs(
                    self.spot, strike, self.rate, market_iv, t_years, otype, self.dividend_yield
                )
                theo = bscpp.bs_price(inputs)
                spread = max(0.05, 0.02 * theo)
                noise = self._rng.normal(0, 0.01 * max(theo, 0.05))
                mid = max(theo + noise, 0.01)

                rows.append(
                    {
                        "strike": strike,
                        "type": otype,
                        "expiration": expiration.isoformat(),
                        "bid": round(max(mid - spread / 2, 0.01), 2),
                        "ask": round(mid + spread / 2, 2),
                        "last": round(mid, 2),
                        "volume": int(self._rng.integers(0, 500)),
                        "open_interest": int(self._rng.integers(0, 5000)),
                        "implied_volatility": np.nan,  # forces the engine to solve for IV itself
                    }
                )

        return pd.DataFrame(rows)[CHAIN_COLUMNS]

    def get_price_history(self, ticker: str, start: dt.date, end: dt.date) -> pd.Series:
        """Simulate a daily-close GBM path at `realized_vol`, ending at `spot`.

        Anchoring the path to end at self.spot (rather than starting there)
        keeps get_underlying_price() and this history mutually consistent
        for a HedgingBacktester run that treats "today" as as_of=end.
        """
        n_days = (end - start).days + 1
        if n_days < 2:
            return pd.Series([self.spot], index=pd.DatetimeIndex([end]), name="close")

        dt_frac = 1.0 / 365.0
        log_returns = self._rng.normal(
            (self.rate - 0.5 * self.realized_vol**2) * dt_frac,
            self.realized_vol * np.sqrt(dt_frac),
            size=n_days - 1,
        )
        # Build forward from an arbitrary start, then rescale so the path
        # lands exactly on self.spot at `end`.
        log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
        path = np.exp(log_path)
        path *= self.spot / path[-1]

        dates = pd.date_range(start, end, freq="D")
        return pd.Series(path, index=dates, name="close")
