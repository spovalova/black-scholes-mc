"""Delta-hedging P&L simulation.

The core question this answers: if you sell an option and delta-hedge it
daily using Black-Scholes deltas computed at some `hedge_vol`, what P&L do
you actually realize once the underlying's *realized* path is known?

In continuous time with hedge_vol == the underlying's true instantaneous
vol, Black-Scholes replication is exact and the hedged position's P&L is
identically zero (net of the risk-free rate) -- selling the option and
running the hedge is a wash. Away from that idealization, two effects
create nonzero P&L:

  1. Discrete rehedging (only resetting delta once a day, not continuously)
     introduces gamma-driven hedging error even if hedge_vol is "correct".
  2. hedge_vol != realized vol: hedging at a vol higher than what actually
     realizes tends to make the option seller money (they collected a
     rich premium relative to the risk they ended up bearing), and vice
     versa. This is the basic mechanics behind "selling implied, buying
     realized" vol trading.

`realized_vs_implied_experiment` demonstrates effect (2) directly by
running many simulated paths per realized-vol level.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

import bscpp
from bscpp.backtest.policies import DeltaPolicy, HedgeState
from bscpp.curve import resolve_rate


class HedgingBacktester:
    """Simulates writing one option and delta-hedging it against a price path.

    transaction_cost_bps: cost of crossing the bid-ask spread on the stock
    hedge leg, in basis points of trade notional, charged on every trade
    (the initial hedge purchase, each daily rebalance, and the final
    liquidation). Defaults to 0 (frictionless), which is the biggest
    unrealism gap in treating this as a "backtest" rather than a pricing
    demo -- a real hedging book pays this on every single rebalance, and
    over many rebalances it accumulates into a real, not-negligible drag.
    A few bps (e.g. 5-10) is a reasonable rough estimate for a liquid
    single-name equity; illiquid names or wide markets would need more.
    """

    def __init__(self, rate, dividend_yield: float = 0.0, transaction_cost_bps: float = 0.0):
        """rate: a bare float (flat rate) or a bscpp.ZeroCurve. Resolved to
        the scalar rate at the option's OWN remaining maturity at each
        step (via bscpp.curve.resolve_rate) -- both for pricing the
        option and for the cash leg's financing accrual over that step.
        Using the option-maturity rate for financing too is a deliberate
        simplification (a real desk separates repo/overnight financing
        from the option's own discount rate); a genuine short-end curve
        is out of scope here, see the README's scope statement."""
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.transaction_cost_bps = transaction_cost_bps

    def run(
        self,
        price_path: pd.Series,
        strike: float,
        expiration: dt.date,
        hedge_vol,
        option_type: str = "call",
        policy=None,
    ) -> pd.DataFrame:
        """Replicate the standard "sell option, delta-hedge" accounting.

        Convention: we sell 1 option for its Black-Scholes premium at
        hedge_vol, and hold shares of the underlying (chosen by `policy`)
        financed out of a cash account that accrues at `rate`.
        `portfolio_value` is the hedged position's mark-to-market P&L since
        inception (cash + stock - option liability); it is exactly 0 on day
        0 by construction, so it doubles as the cumulative hedging P&L
        series.

        hedge_vol: a float (constant marking vol -- vega P&L is then zero
        by construction), or a pd.Series aligned/alignable to
        price_path.index (a re-marked vol path -- e.g. trailing realized or
        an implied-vol series). With a Series, the option is re-marked at
        each date's vol, and `attribute_pnl` reports the resulting vega
        P&L as an explicit term instead of silently burying the (often
        DOMINANT) re-marking P&L in the residual.

        policy: an object with `target_shares(held, state) -> float`
        deciding the post-observation stock position -- see
        bscpp.backtest.policies (DeltaPolicy, BandPolicy,
        WhalleyWilmottPolicy, CallablePolicy). Default: DeltaPolicy
        (rebalance to the exact delta every observation), which reproduces
        the classic daily-rebalanced backtest.
        """
        dates = list(price_path.index)
        if len(dates) < 2:
            raise ValueError("price_path needs at least 2 observations")

        policy = policy or DeltaPolicy()
        cost_frac = self.transaction_cost_bps / 10_000.0

        if isinstance(hedge_vol, pd.Series):
            vols = hedge_vol.reindex(price_path.index)
            if vols.isna().any():
                raise ValueError("hedge_vol series does not cover every price_path date")
            vols = vols.astype(float)
        else:
            vols = pd.Series(float(hedge_vol), index=price_path.index)

        spot0 = float(price_path.iloc[0])
        vol0 = float(vols.iloc[0])
        t0 = max((expiration - dates[0].date()).days, 1) / 365.0
        rate0 = resolve_rate(self.rate, t0)
        inputs0 = bscpp.make_inputs(spot0, strike, rate0, vol0, t0, option_type,
                                     self.dividend_yield)
        result0 = bscpp.bs_price_with_greeks(inputs0)

        state0 = HedgeState(t=t0, spot=spot0, delta=result0.greeks.delta,
                            gamma=result0.greeks.gamma, vega=result0.greeks.vega,
                            rate=rate0, cost_frac=cost_frac)
        shares = policy.target_shares(0.0, state0)
        cost0 = abs(shares) * spot0 * cost_frac  # crossing the spread to establish the hedge
        cash = result0.price - shares * spot0 - cost0
        spot_prev = spot0

        rows = [{
            "date": dates[0], "spot": spot0, "T": t0, "delta": result0.greeks.delta,
            "gamma": result0.greeks.gamma, "theta": result0.greeks.theta,
            "vega": result0.greeks.vega, "hedge_vol": vol0,
            "option_value": result0.price, "cash": cash, "shares": shares,
            "transaction_cost": cost0,
            "portfolio_value": cash + shares * spot0 - result0.price,
        }]

        for i in range(1, len(dates)):
            date = dates[i]
            spot = float(price_path.iloc[i])
            vol = float(vols.iloc[i])
            elapsed_days = (dates[i] - dates[i - 1]).days or 1
            dt_years = elapsed_days / 365.0
            t = max((expiration - date.date()).days, 0) / 365.0
            # Financing accrues over this step at the rate for the OPTION'S
            # remaining maturity as of this step (not a separate overnight/
            # repo rate -- see __init__'s docstring for why that's a
            # deliberate simplification, not an oversight).
            rate = resolve_rate(self.rate, t if t > 0.0 else t0)
            cash *= math.exp(rate * dt_years)
            # Dividend income on the stock leg carried into this interval
            # (held at `shares` since the last rebalance). Without this,
            # the hedge is inconsistent with using dividend_yield-adjusted
            # deltas from Black-Scholes, which price in the assumption that
            # the stock holder collects this yield.
            cash += shares * spot_prev * (math.exp(self.dividend_yield * dt_years) - 1.0)
            spot_prev = spot

            if t <= 0.0:
                payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
                cost = abs(shares) * spot * cost_frac  # crossing the spread to unwind the hedge
                cash += shares * spot - cost  # liquidate the hedge; the unified formula
                shares, option_value = 0.0, payoff  # below nets out -option_value (=payoff) once
                delta, gamma, theta, vega = 0.0, 0.0, 0.0, 0.0  # degenerate at expiry
                vol = float(vols.iloc[i - 1])  # no re-mark at expiry: payoff has no vol
            else:
                inputs = bscpp.make_inputs(spot, strike, rate, vol, t, option_type,
                                            self.dividend_yield)
                result = bscpp.bs_price_with_greeks(inputs)
                option_value = result.price
                delta, gamma = result.greeks.delta, result.greeks.gamma
                theta, vega = result.greeks.theta, result.greeks.vega
                state = HedgeState(t=t, spot=spot, delta=delta, gamma=gamma, vega=vega,
                                   rate=rate, cost_frac=cost_frac)
                new_shares = policy.target_shares(shares, state)
                cost = abs(new_shares - shares) * spot * cost_frac  # crossing the spread
                cash -= (new_shares - shares) * spot + cost  # trade the change, pay the spread
                shares = new_shares

            portfolio_value = cash + shares * spot - option_value
            rows.append({
                "date": date, "spot": spot, "T": t, "delta": delta,
                "gamma": gamma, "theta": theta, "vega": vega, "hedge_vol": vol,
                "option_value": option_value, "cash": cash, "shares": shares,
                "transaction_cost": cost,
                "portfolio_value": portfolio_value,
            })

        return pd.DataFrame(rows)

    def attribute_pnl(self, result: pd.DataFrame) -> pd.DataFrame:
        """Decompose each day's realized hedging P&L into financing + gamma + theta.

        Derivation (short 1 option, hedged with `shares` = option delta,
        cash growing at `rate`, rebalanced once per row): writing
        Pi = cash + shares*S - V and expanding the discrete update,

            dPi = cash_prev*(e^{r dt} - 1) + shares_prev*dS - dV

        Second-order Taylor expanding dV of the *same* pricing function
        (constant hedge_vol) around the prior state,

            dV ~= delta_prev*dS + 0.5*gamma_prev*dS^2 + theta_prev*dt

        and substituting (shares_prev == delta_prev by construction) gives

            dPi ~= cash_prev*(e^{r dt} - 1) - 0.5*gamma_prev*dS^2 - theta_prev*dt

        i.e. financing + a pure "gamma P&L" term + a pure "theta P&L" term.
        For a short option this is the textbook result: theta_prev < 0 for a
        long option means -theta_prev*dt > 0 -- the short seller collects
        time decay -- while -0.5*gamma_prev*dS^2 <= 0 always -- the short
        seller pays for realized convexity. `attribution_error` is the
        Taylor-expansion residual (higher-order terms + discrete-vs-
        continuous rebalancing effects); it should be small relative to the
        other terms for daily steps on typical single-name vol.

        This is the per-day, non-asymptotic version of Carr & Madan's
        (2002, "Towards a Theory of Volatility Trading") canonical result:
        substituting the Black-Scholes PDE for theta and dS^2 ~= sigma_R^2
        S^2 dt collapses the formula above to
        gamma_pnl + theta_pnl ~= 0.5*Gamma*S^2*(hedge_vol^2 - sigma_R^2)*dt,
        the standard "P&L from delta-hedging at the wrong vol" identity.

        transaction_cost_pnl is added as a fifth, EXACT (not
        Taylor-approximated) term when transaction_cost_bps > 0: the spread
        cost on each rebalance flows straight through the same cash
        account the rest of the derivation above already accounts for, so
        it enters `predicted_pnl` unchanged rather than as part of the
        residual.

        Two further EXPLICIT terms (both exactly zero in the classic
        constant-vol, rebalance-to-delta configuration, so the original
        decomposition is recovered unchanged there):

        - **vega_pnl** = vega_prev * (hedge_vol_t - hedge_vol_{t-1}): the
          re-marking P&L when `run` was given a time-varying hedge_vol
          series. On a real book this is often the DOMINANT daily option
          P&L term; without it, a time-varying-vol attribution would
          silently mislabel real vega P&L as higher-order residual.
        - **delta_gap_pnl** = (shares_prev - delta_prev) * dS: the first-
          order P&L of deliberately holding away from the model delta,
          which is exactly what band policies (BandPolicy,
          WhalleyWilmottPolicy) do inside their no-trade region. The
          derivation above assumed shares_prev == delta_prev; this term is
          the correction when a policy chooses otherwise.
        """
        df = result.reset_index(drop=True).copy()
        n = len(df)
        realized = np.zeros(n)
        financing = np.zeros(n)
        gamma_pnl = np.zeros(n)
        theta_pnl = np.zeros(n)
        vega_pnl = np.zeros(n)
        delta_gap_pnl = np.zeros(n)
        transaction_cost_pnl = np.zeros(n)

        has_vega = "vega" in df and "hedge_vol" in df
        has_shares = "shares" in df and "delta" in df

        for i in range(1, n):
            elapsed_days = (df["date"].iat[i] - df["date"].iat[i - 1]).days or 1
            dt_years = elapsed_days / 365.0
            realized[i] = df["portfolio_value"].iat[i] - df["portfolio_value"].iat[i - 1]
            # Same rate resolution as run(): the option's own remaining
            # maturity as of the END of this step (falling back to the
            # step's starting maturity at expiry, where remaining T=0).
            t_i = df["T"].iat[i]
            step_rate = resolve_rate(self.rate, t_i if t_i > 0.0 else df["T"].iat[i - 1])
            financing[i] = df["cash"].iat[i - 1] * (math.exp(step_rate * dt_years) - 1.0)
            d_spot = df["spot"].iat[i] - df["spot"].iat[i - 1]
            gamma_pnl[i] = -0.5 * df["gamma"].iat[i - 1] * d_spot ** 2
            theta_pnl[i] = -df["theta"].iat[i - 1] * dt_years
            if has_vega:
                d_vol = df["hedge_vol"].iat[i] - df["hedge_vol"].iat[i - 1]
                vega_pnl[i] = -df["vega"].iat[i - 1] * d_vol  # short the option: -vega exposure
            if has_shares:
                delta_gap_pnl[i] = (df["shares"].iat[i - 1] - df["delta"].iat[i - 1]) * d_spot
            transaction_cost_pnl[i] = -df["transaction_cost"].iat[i] if "transaction_cost" in df else 0.0

        df["realized_pnl"] = realized
        df["financing_pnl"] = financing
        df["gamma_pnl"] = gamma_pnl
        df["theta_pnl"] = theta_pnl
        df["vega_pnl"] = vega_pnl
        df["delta_gap_pnl"] = delta_gap_pnl
        df["transaction_cost_pnl"] = transaction_cost_pnl
        df["predicted_pnl"] = (df["financing_pnl"] + df["gamma_pnl"] + df["theta_pnl"] +
                                df["vega_pnl"] + df["delta_gap_pnl"] +
                                df["transaction_cost_pnl"])
        df["attribution_error"] = df["realized_pnl"] - df["predicted_pnl"]
        return df


def realized_vs_implied_experiment(
    hedge_vol: float,
    realized_vols: list[float],
    rate,
    spot: float = 100.0,
    strike: float = 100.0,
    t_days: int = 60,
    option_type: str = "call",
    n_paths_per_vol: int = 200,
    seed: int = 123,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Sweep realized vol against a fixed hedge_vol and report mean hedging P&L.

    Expect mean_hedging_pnl > 0 when realized_vol < hedge_vol (you sold rich
    and the underlying stayed calmer than you hedged for) and < 0 when
    realized_vol > hedge_vol (the underlying moved more than you were paid
    for). This is a Monte Carlo demonstration, not a closed-form check --
    the sign and rough magnitude are what matter.

    transaction_cost_bps shifts every mean_hedging_pnl down -- more so at
    higher realized_vol, since bigger daily spot moves mean bigger delta
    changes and therefore more rebalance notional traded (empirically,
    roughly 30-45% more drag at realized_vol=0.45 than at 0.15 for a fixed
    hedge_vol=0.30 in this sweep's own test case) -- a real, non-negligible
    effect a frictionless version of this sweep would miss entirely.
    """
    rng = np.random.default_rng(seed)
    backtester = HedgingBacktester(rate=rate, transaction_cost_bps=transaction_cost_bps)
    anchor = dt.date.today()
    dates = pd.date_range(anchor, periods=t_days + 1, freq="D")
    expiration = dates[-1].date()
    dt_frac = 1.0 / 365.0
    sim_rate = resolve_rate(rate, t_days / 365.0)  # scalar drift for the simulated GBM path

    rows = []
    for rv in realized_vols:
        pnls = []
        for _ in range(n_paths_per_vol):
            z = rng.normal(size=t_days)
            log_ret = (sim_rate - 0.5 * rv**2) * dt_frac + rv * math.sqrt(dt_frac) * z
            path = spot * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
            series = pd.Series(path, index=dates)

            result = backtester.run(series, strike, expiration, hedge_vol, option_type)
            pnls.append(result["portfolio_value"].iloc[-1])

        pnls = np.array(pnls)
        rows.append({
            "realized_vol": rv,
            "hedge_vol": hedge_vol,
            "mean_hedging_pnl": pnls.mean(),
            "std_hedging_pnl": pnls.std(ddof=1),
            "n_paths": n_paths_per_vol,
        })

    return pd.DataFrame(rows)
