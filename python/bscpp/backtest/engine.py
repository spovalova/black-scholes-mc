"""Pricing engine: pulls a chain slice from a DataProvider, prices it with the
C++ BS/MC pricers, and compares to observed market prices.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import bscpp
from bscpp.backtest.data_provider import DataProvider


def _time_to_expiry_years(expiration: dt.date, as_of: dt.date) -> float:
    return max((expiration - as_of).days, 0) / 365.0


class StripPricer:
    """Prices a slice of a live (or historical) option chain with BS + MC."""

    def __init__(
        self,
        provider: DataProvider,
        rate: float = 0.05,
        dividend_yield: float = 0.0,
        mc_paths: int = 50_000,
        mc_seed: int = 42,
    ):
        self.provider = provider
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.mc_paths = mc_paths
        self.mc = bscpp.MonteCarloPricer(seed=mc_seed)

    def price_strip(
        self,
        ticker: str,
        expiration: dt.date,
        as_of: dt.date | None = None,
        strike_range: tuple[float, float] = (0.8, 1.2),
        use_mc: bool = True,
    ) -> pd.DataFrame:
        """Price every contract in [strike_range * spot] at `expiration`.

        Returns the chain DataFrame augmented with: spot, T, model_iv,
        bs_price, mc_price, mc_std_error, delta/gamma/vega/theta/rho,
        bs_error_vs_market, bs_error_pct.
        """
        as_of = as_of or dt.date.today()
        spot = self.provider.get_underlying_price(ticker, as_of=as_of)
        chain = self.provider.get_option_chain(ticker, expiration, as_of=as_of)
        if chain.empty:
            return chain

        lo, hi = strike_range
        chain = chain[(chain["strike"] >= spot * lo) & (chain["strike"] <= spot * hi)].copy()
        if chain.empty:
            return chain

        t_years = _time_to_expiry_years(expiration, as_of)
        if t_years <= 0:
            raise ValueError(f"expiration {expiration} is not after as_of {as_of}")

        chain["mid"] = chain[["bid", "ask"]].mean(axis=1)
        chain["mid"] = chain["mid"].fillna(chain["last"])

        model_ivs, bs_prices = [], []
        deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
        mc_prices, mc_errs = [], []

        for _, row in chain.iterrows():
            otype = "call" if row["type"] == "call" else "put"
            iv = row.get("implied_volatility")

            if iv is None or pd.isna(iv) or iv <= 0:
                mid = row["mid"]
                if mid and mid > 0:
                    seed_inputs = bscpp.make_inputs(
                        spot, row["strike"], self.rate, 0.20, t_years, otype, self.dividend_yield
                    )
                    solved = bscpp.bs_implied_vol(seed_inputs, mid)
                    iv = solved if solved == solved else 0.20  # NaN check
                else:
                    iv = 0.20
            model_ivs.append(iv)

            inputs = bscpp.make_inputs(
                spot, row["strike"], self.rate, iv, t_years, otype, self.dividend_yield
            )
            result = bscpp.bs_price_with_greeks(inputs)
            bs_prices.append(result.price)
            deltas.append(result.greeks.delta)
            gammas.append(result.greeks.gamma)
            vegas.append(result.greeks.vega)
            thetas.append(result.greeks.theta)
            rhos.append(result.greeks.rho)

            if use_mc:
                mc_result = self.mc.price_european(inputs, self.mc_paths, True)
                mc_prices.append(mc_result.price)
                mc_errs.append(mc_result.std_error)
            else:
                mc_prices.append(float("nan"))
                mc_errs.append(float("nan"))

        chain["spot"] = spot
        chain["T"] = t_years
        chain["model_iv"] = model_ivs
        chain["bs_price"] = bs_prices
        chain["mc_price"] = mc_prices
        chain["mc_std_error"] = mc_errs
        chain["delta"] = deltas
        chain["gamma"] = gammas
        chain["vega"] = vegas
        chain["theta"] = thetas
        chain["rho"] = rhos
        chain["bs_error_vs_market"] = chain["bs_price"] - chain["mid"]
        chain["bs_error_pct"] = chain["bs_error_vs_market"] / chain["mid"]

        return chain.reset_index(drop=True)


class Backtester:
    """Runs StripPricer over a sequence of as-of dates and aggregates results.

    Note: pulling a *historical* chain snapshot for an arbitrary past date
    depends on the DataProvider supporting it (Polygon's historical options
    snapshots require a paid tier above "Options Starter"). With MockProvider,
    or a live-only provider, every as_of effectively prices off the current
    chain — fine for exercising the pipeline, not a real historical backtest.
    """

    def __init__(self, pricer: StripPricer):
        self.pricer = pricer

    def run(
        self,
        ticker: str,
        expiration: dt.date,
        as_of_dates: list[dt.date],
        strike_range: tuple[float, float] = (0.8, 1.2),
        use_mc: bool = False,
    ) -> pd.DataFrame:
        frames = []
        for as_of in as_of_dates:
            df = self.pricer.price_strip(
                ticker, expiration, as_of=as_of, strike_range=strike_range, use_mc=use_mc
            )
            if df.empty:
                continue
            df = df.copy()
            df["as_of"] = as_of
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def summary(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results
        return (
            results.groupby("as_of")
            .agg(
                mean_abs_error=("bs_error_vs_market", lambda s: s.abs().mean()),
                mean_abs_pct_error=("bs_error_pct", lambda s: s.abs().mean()),
                n_contracts=("strike", "count"),
            )
            .reset_index()
        )
