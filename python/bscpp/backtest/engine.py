"""Pricing engine: pulls a chain slice from a DataProvider, prices it with the
C++ BS/MC pricers, and compares to observed market prices.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

import bscpp
from bscpp.backtest.data_provider import DataProvider
from bscpp.curve import resolve_rate


def _time_to_expiry_years(expiration: dt.date, as_of: dt.date) -> float:
    return max((expiration - as_of).days, 0) / 365.0


def extract_forward_and_carry(chain: pd.DataFrame, spot: float, t_years: float,
                               rate: float) -> tuple[float, float]:
    """Implied forward and cost-of-carry from put-call parity, at the
    strike minimizing |C-P| -- the standard desk recipe. Away from that
    strike, parity is contaminated by wider bid-ask spreads and (for
    American-style equity options) a growing early-exercise premium on
    whichever leg is deep ITM; the min-|C-P| strike is close to the
    forward itself, where both legs are near the money and neither effect
    is large.

    F = K* + e^{rT}(C-P) at that strike (mids). Returns (forward,
    implied_carry) where implied_carry b satisfies F = spot*e^{b*T} -- b
    bundles dividends, borrow cost, and any funding-spread noise into ONE
    number. It is NOT a separately-identified dividend yield or borrow
    rate (those aren't separable from option prices alone without further
    assumptions) -- but it sidesteps needing to know either separately to
    price at all, which is the actual point: this replaces an assumed
    dividend_yield with a market-implied number.

    Returns (nan, nan) if the chain has no strike with both a call and a
    put quote to pair.
    """
    mid = chain[["bid", "ask"]].mean(axis=1).fillna(chain["last"])
    otype = chain["type"].where(chain["type"] == "call", "put")
    pivot = pd.DataFrame({"strike": chain["strike"], "type": otype, "mid": mid}).pivot_table(
        index="strike", columns="type", values="mid")
    if "call" not in pivot.columns or "put" not in pivot.columns:
        return float("nan"), float("nan")
    paired = pivot.dropna(subset=["call", "put"])
    paired = paired[(paired["call"] > 0) & (paired["put"] > 0)]
    if paired.empty:
        return float("nan"), float("nan")

    k_star = (paired["call"] - paired["put"]).abs().idxmin()
    c, p = paired.loc[k_star, "call"], paired.loc[k_star, "put"]
    forward = float(k_star) + math.exp(rate * t_years) * float(c - p)
    if forward <= 0:
        return float("nan"), float("nan")
    implied_carry = math.log(forward / spot) / t_years
    return forward, implied_carry


class StripPricer:
    """Prices a slice of a live (or historical) option chain with BS + MC."""

    def __init__(
        self,
        provider: DataProvider,
        rate,
        dividend_yield: float = 0.0,
        mc_paths: int = 50_000,
        mc_seed: int = 42,
    ):
        """rate: a bare float (flat rate) or a bscpp.ZeroCurve -- resolved
        to the scalar rate at this chain's own maturity in price_strip.
        No default: a hardcoded rate is exactly the kind of assumption
        that shouldn't be silently inherited by every caller."""
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
        rate = resolve_rate(self.rate, t_years)

        chain["mid"] = chain[["bid", "ask"]].mean(axis=1)
        chain["mid"] = chain["mid"].fillna(chain["last"])
        otypes_series = chain["type"].where(chain["type"] == "call", "put")
        otypes = otypes_series.tolist()

        forward, implied_carry = extract_forward_and_carry(chain, spot, t_years, rate)
        chain["implied_forward"] = forward
        chain["implied_carry"] = implied_carry
        # F = S*e^{(r-q)T}, so the dividend yield consistent with a given
        # rate AND the market-implied forward is q = r - implied_carry.
        # Pricing off this instead of self.dividend_yield replaces an
        # ASSUMED dividend input with a market-implied one whenever the
        # chain has paired call/put quotes to extract a forward from --
        # eliminating the dividend-assumption problem, not just reporting
        # implied_carry as a side diagnostic. Falls back to
        # self.dividend_yield when forward extraction fails (e.g. a
        # calls-only chain).
        dividend_yield = rate - implied_carry if implied_carry == implied_carry else self.dividend_yield

        # --- resolve one implied vol per contract, batching the solver call ---
        # rows that already carry a usable quoted IV skip the solve entirely;
        # everything else is solved in a single C++ call rather than one
        # Python->C++ crossing per contract.
        given_iv = chain["implied_volatility"].to_numpy(dtype=float)
        mid = chain["mid"].to_numpy(dtype=float)
        needs_solve = ~np.isfinite(given_iv) | (given_iv <= 0)
        has_mid = np.isfinite(mid) & (mid > 0)
        # OTM-only solving (the standard desk recipe): calls above the
        # implied forward, puts below. Deep-ITM prices are almost pure
        # intrinsic value and barely move with vol -- solving IV from an
        # ITM mid is the classic ill-conditioned case (the solver has to
        # invert a nearly-flat price/vol relationship), and for American-
        # style equity options the ITM leg also carries an early-exercise
        # premium our European solver has no way to account for. Falls
        # back to solving every row (the old behavior) if the chain
        # doesn't have paired call/put quotes to extract a forward from.
        if forward == forward:  # not NaN
            is_call = np.array([t == "call" for t in otypes])
            strikes = chain["strike"].to_numpy(dtype=float)
            is_otm = np.where(is_call, strikes >= forward, strikes <= forward)
        else:
            is_otm = np.ones(len(chain), dtype=bool)
        solve_mask = needs_solve & has_mid & is_otm
        # rows in needs_solve & ~has_mid have neither a quote nor a usable
        # mid to solve from; rows in needs_solve & has_mid & ~is_otm are
        # ITM and deliberately not solved. Both keep the 0.20 placeholder
        # below and are marked "fallback" (NaN pricing outputs) -- see below.

        # iv_source per row: "quoted" (vendor IV used as-is), "solved" (we
        # solved it from the mid), or "fallback" (no usable quote OR the
        # solve failed). Fallback rows get model_iv = NaN and NaN pricing
        # outputs below -- the previous behavior silently priced them at an
        # invented 0.20 vol with no flag, contaminating every downstream
        # consumer (SVI fits, calibration, error stats) with fake IVs.
        iv_source = np.where(needs_solve, "fallback", "quoted").astype(object)
        model_ivs = np.where(needs_solve, np.nan, given_iv)
        if solve_mask.any():
            idx = np.flatnonzero(solve_mask)
            seed_inputs = [
                bscpp.make_inputs(spot, chain["strike"].iat[i], rate, 0.20, t_years,
                                   otypes[i], dividend_yield)
                for i in idx
            ]
            solved = bscpp.bs_implied_vol_batch(seed_inputs, [mid[i] for i in idx])
            for i, iv in zip(idx, solved):
                if iv == iv:  # solve succeeded
                    model_ivs[i] = iv
                    iv_source[i] = "solved"
                # else: leave NaN / "fallback" -- a failed solve means the
                # mid itself is outside what any vol could produce.

        # --- batch price + Greeks for the whole chain in one C++ call ---
        # Fallback rows have no real IV; price them at a harmless placeholder
        # then overwrite their outputs with NaN below so nothing downstream
        # can mistake them for data.
        priceable = model_ivs == model_ivs  # not-NaN mask
        safe_ivs = np.where(priceable, model_ivs, 0.20)
        inputs_list = [
            bscpp.make_inputs(spot, chain["strike"].iat[i], rate, safe_ivs[i], t_years,
                               otypes[i], dividend_yield)
            for i in range(len(chain))
        ]
        results = bscpp.bs_price_with_greeks_batch(inputs_list)

        if use_mc:
            mc_results = [
                self.mc.price_european(inp, self.mc_paths, True) if priceable[i] else None
                for i, inp in enumerate(inputs_list)
            ]
            mc_prices = [r.price if r else float("nan") for r in mc_results]
            mc_errs = [r.std_error if r else float("nan") for r in mc_results]
        else:
            mc_prices = [float("nan")] * len(chain)
            mc_errs = [float("nan")] * len(chain)

        nan_if_fallback = lambda vals: [  # noqa: E731
            v if priceable[i] else float("nan") for i, v in enumerate(vals)
        ]

        chain["spot"] = spot
        chain["T"] = t_years
        chain["model_iv"] = model_ivs
        chain["iv_source"] = iv_source
        chain["bs_price"] = nan_if_fallback([r.price for r in results])
        chain["mc_price"] = mc_prices
        chain["mc_std_error"] = mc_errs
        chain["delta"] = nan_if_fallback([r.greeks.delta for r in results])
        chain["gamma"] = nan_if_fallback([r.greeks.gamma for r in results])
        chain["vega"] = nan_if_fallback([r.greeks.vega for r in results])
        chain["theta"] = nan_if_fallback([r.greeks.theta for r in results])
        chain["rho"] = nan_if_fallback([r.greeks.rho for r in results])
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
