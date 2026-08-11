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

        rows = [{
            "date": dates[0], "spot": spot0, "T": t0, "delta": shares,
            "option_value": result0.price, "cash": cash, "shares": shares,
            "portfolio_value": cash + shares * spot0 - result0.price,
        }]

        for i in range(1, len(dates)):
            date = dates[i]
            spot = float(price_path.iloc[i])
            elapsed_days = (dates[i] - dates[i - 1]).days or 1
            cash *= math.exp(self.rate * elapsed_days / 365.0)

            t = max((expiration - date.date()).days, 0) / 365.0
            if t <= 0.0:
                payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
                cash += shares * spot  # liquidate the hedge; the unified formula
                shares, option_value = 0.0, payoff  # below nets out -option_value (=payoff) once
            else:
                inputs = bscpp.make_inputs(spot, strike, self.rate, hedge_vol, t, option_type,
                                            self.dividend_yield)
                result = bscpp.bs_price_with_greeks(inputs)
                option_value = result.price
                new_delta = result.greeks.delta
                cash -= (new_delta - shares) * spot  # buy/sell the delta change
                shares = new_delta

            portfolio_value = cash + shares * spot - option_value
            rows.append({
                "date": date, "spot": spot, "T": t, "delta": shares,
                "option_value": option_value, "cash": cash, "shares": shares,
                "portfolio_value": portfolio_value,
            })

        return pd.DataFrame(rows)


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
