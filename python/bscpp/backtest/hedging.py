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


class HedgingBacktester:
    """Simulates writing one option and delta-hedging it against a price path."""

    def __init__(self, rate: float, dividend_yield: float = 0.0):
        self.rate = rate
        self.dividend_yield = dividend_yield

    def run(
        self,
        price_path: pd.Series,
        strike: float,
        expiration: dt.date,
        hedge_vol: float,
        option_type: str = "call",
    ) -> pd.DataFrame:
        """Replicate the standard "sell option, delta-hedge daily" accounting.

        Convention: we sell 1 option for its Black-Scholes premium at
        hedge_vol, and hold delta shares of the underlying financed out of
        a cash account that accrues at `rate`. `portfolio_value` is the
        hedged position's mark-to-market P&L since inception (cash + stock
        - option liability); it is exactly 0 on day 0 by construction, so
        it doubles as the cumulative hedging P&L series.
        """
        dates = list(price_path.index)
        if len(dates) < 2:
            raise ValueError("price_path needs at least 2 observations")

        spot0 = float(price_path.iloc[0])
        t0 = max((expiration - dates[0].date()).days, 1) / 365.0
        inputs0 = bscpp.make_inputs(spot0, strike, self.rate, hedge_vol, t0, option_type,
                                     self.dividend_yield)
        result0 = bscpp.bs_price_with_greeks(inputs0)

        cash = result0.price - result0.greeks.delta * spot0
        shares = result0.greeks.delta
        spot_prev = spot0

        rows = [{
            "date": dates[0], "spot": spot0, "T": t0, "delta": shares,
            "gamma": result0.greeks.gamma, "theta": result0.greeks.theta,
            "option_value": result0.price, "cash": cash, "shares": shares,
            "portfolio_value": cash + shares * spot0 - result0.price,
        }]

        for i in range(1, len(dates)):
            date = dates[i]
            spot = float(price_path.iloc[i])
            elapsed_days = (dates[i] - dates[i - 1]).days or 1
            dt_years = elapsed_days / 365.0
            cash *= math.exp(self.rate * dt_years)
            # Dividend income on the stock leg carried into this interval
            # (held at `shares` since the last rebalance). Without this,
            # the hedge is inconsistent with using dividend_yield-adjusted
            # deltas from Black-Scholes, which price in the assumption that
            # the stock holder collects this yield.
            cash += shares * spot_prev * (math.exp(self.dividend_yield * dt_years) - 1.0)
            spot_prev = spot

            t = max((expiration - date.date()).days, 0) / 365.0
            if t <= 0.0:
                payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
                cash += shares * spot  # liquidate the hedge; the unified formula
                shares, option_value = 0.0, payoff  # below nets out -option_value (=payoff) once
                gamma, theta = 0.0, 0.0  # degenerate at expiry; unused past this row anyway
            else:
                inputs = bscpp.make_inputs(spot, strike, self.rate, hedge_vol, t, option_type,
                                            self.dividend_yield)
                result = bscpp.bs_price_with_greeks(inputs)
                option_value = result.price
                new_delta = result.greeks.delta
                gamma, theta = result.greeks.gamma, result.greeks.theta
                cash -= (new_delta - shares) * spot  # buy/sell the delta change
                shares = new_delta

            portfolio_value = cash + shares * spot - option_value
            rows.append({
                "date": date, "spot": spot, "T": t, "delta": shares,
                "gamma": gamma, "theta": theta,
                "option_value": option_value, "cash": cash, "shares": shares,
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

        Known gap: `hedge_vol` is constant for the life of a run, so vega
        P&L is exactly zero here by construction, not by finding it small.
        On a real book, day-to-day vol re-marking is often the DOMINANT
        source of daily option P&L, not gamma/theta -- if this backtester
        is ever extended to a time-varying hedge_vol, attribution_error
        would silently absorb the real vega P&L and mislabel it as
        higher-order residual unless this decomposition is extended to
        include an explicit vega term.
        """
        df = result.reset_index(drop=True).copy()
        n = len(df)
        realized = np.zeros(n)
        financing = np.zeros(n)
        gamma_pnl = np.zeros(n)
        theta_pnl = np.zeros(n)

        for i in range(1, n):
            elapsed_days = (df["date"].iat[i] - df["date"].iat[i - 1]).days or 1
            dt_years = elapsed_days / 365.0
            realized[i] = df["portfolio_value"].iat[i] - df["portfolio_value"].iat[i - 1]
            financing[i] = df["cash"].iat[i - 1] * (math.exp(self.rate * dt_years) - 1.0)
            d_spot = df["spot"].iat[i] - df["spot"].iat[i - 1]
            gamma_pnl[i] = -0.5 * df["gamma"].iat[i - 1] * d_spot ** 2
            theta_pnl[i] = -df["theta"].iat[i - 1] * dt_years

        df["realized_pnl"] = realized
        df["financing_pnl"] = financing
        df["gamma_pnl"] = gamma_pnl
        df["theta_pnl"] = theta_pnl
        df["predicted_pnl"] = df["financing_pnl"] + df["gamma_pnl"] + df["theta_pnl"]
        df["attribution_error"] = df["realized_pnl"] - df["predicted_pnl"]
        return df


def realized_vs_implied_experiment(
    hedge_vol: float,
    realized_vols: list[float],
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    t_days: int = 60,
    option_type: str = "call",
    n_paths_per_vol: int = 200,
    seed: int = 123,
) -> pd.DataFrame:
    """Sweep realized vol against a fixed hedge_vol and report mean hedging P&L.

    Expect mean_hedging_pnl > 0 when realized_vol < hedge_vol (you sold rich
    and the underlying stayed calmer than you hedged for) and < 0 when
    realized_vol > hedge_vol (the underlying moved more than you were paid
    for). This is a Monte Carlo demonstration, not a closed-form check --
    the sign and rough magnitude are what matter.
    """
    rng = np.random.default_rng(seed)
    backtester = HedgingBacktester(rate=rate)
    anchor = dt.date.today()
    dates = pd.date_range(anchor, periods=t_days + 1, freq="D")
    expiration = dates[-1].date()
    dt_frac = 1.0 / 365.0

    rows = []
    for rv in realized_vols:
        pnls = []
        for _ in range(n_paths_per_vol):
            z = rng.normal(size=t_days)
            log_ret = (rate - 0.5 * rv**2) * dt_frac + rv * math.sqrt(dt_frac) * z
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
