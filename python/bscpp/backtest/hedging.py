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
from bscpp.clock import Clock
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

    def __init__(self, rate, dividend_yield: float = 0.0, transaction_cost_bps: float = 0.0,
                 clock: Clock = Clock()):
        """rate: a bare float (flat rate) or a bscpp.ZeroCurve. Resolved to
        the scalar rate at the option's OWN remaining maturity at each
        step (via bscpp.curve.resolve_rate) -- both for pricing the
        option and for the cash leg's financing accrual over that step.
        Using the option-maturity rate for financing too is a deliberate
        simplification (a real desk separates repo/overnight financing
        from the option's own discount rate); a genuine short-end curve
        is out of scope here, see the README's scope statement.

        clock: the day-count convention (see bscpp.clock.Clock) for every
        time-to-expiry and every step's elapsed time. Defaults to ACT/365
        (calendar days) -- theta and financing accrue in calendar time
        (an option loses value over a weekend even though nothing
        trades), so this is the convention every other pricer/hedger in
        this project uses too, not an independent choice per class.
        """
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.transaction_cost_bps = transaction_cost_bps
        self.clock = clock

    def price_path(
        self,
        price_path: pd.Series,
        strike: float,
        expiration: dt.date,
        hedge_vol,
        option_type: str = "call",
    ) -> pd.DataFrame:
        """Prices AND computes Greeks for every date in `price_path` in
        ONE batched call (`bs_price_with_greeks_batch_arrays`) instead of
        one Python->C++ crossing per day -- the option's price/Greeks at
        each date depend only on (spot, vol, T, strike, rate), never on a
        hedging POLICY's decisions, so this table can be computed ONCE
        and reused across every policy simulated against the same window.
        `run` below calls this internally (so there's a single source of
        truth for the pricing logic); `bscpp.backtest.frontier.
        run_policy_grid` calls it directly, ONCE per window, then runs
        every (risk_aversion, band_multiplier) policy cell against the
        same table instead of re-pricing per cell -- the dominant cost in
        that study before this existed.

        Returns a DataFrame indexed like `price_path`, with columns
        [date, spot, T, rate, hedge_vol, price, delta, gamma, theta,
        vega] -- one row per date, the exact per-day option state `run`'s
        simulation loop consumes. Rows at or past expiry (T<=0) are
        priced as intrinsic value with zero Greeks and the PRIOR day's
        `hedge_vol` (no re-mark at expiry: a payoff has no vol) --
        matching `run`'s existing expiry handling exactly, not a new
        convention. `T`/`rate` for day 0 use `floor_at_one_day=True`
        (never exactly the "instant" of pricing), matching `run`'s own
        day-0 special case.
        """
        dates = list(price_path.index)
        if len(dates) < 2:
            raise ValueError("price_path needs at least 2 observations")
        n = len(dates)

        if isinstance(hedge_vol, pd.Series):
            vols = hedge_vol.reindex(price_path.index)
            if vols.isna().any():
                raise ValueError("hedge_vol series does not cover every price_path date")
            vols = vols.astype(float).to_numpy()
        else:
            vols = np.full(n, float(hedge_vol))

        t_arr = np.empty(n)
        t_arr[0] = self.clock.time_to_expiry(dates[0].date(), expiration, floor_at_one_day=True)
        for i in range(1, n):
            t_arr[i] = self.clock.time_to_expiry(dates[i].date(), expiration)

        rate_arr = np.empty(n)
        rate_arr[0] = resolve_rate(self.rate, t_arr[0])
        for i in range(1, n):
            # Same fallback as run(): the option's remaining maturity as
            # of the END of this step, falling back to day 0's maturity
            # at/past expiry (T<=0), not the previous step's T.
            rate_arr[i] = resolve_rate(self.rate, t_arr[i] if t_arr[i] > 0.0 else t_arr[0])

        spot_arr = price_path.to_numpy(dtype=float)
        is_expiry = t_arr <= 0.0

        price_arr = np.empty(n)
        delta_arr = np.zeros(n)
        gamma_arr = np.zeros(n)
        theta_arr = np.zeros(n)
        vega_arr = np.zeros(n)

        live = ~is_expiry
        if live.any():
            otype_val = 0 if option_type == "call" else 1
            otype_arr = np.full(int(live.sum()), otype_val, dtype=np.int32)
            # bs_price_with_greeks_batch_arrays returns (price, delta,
            # gamma, VEGA, THETA, rho) -- vega before theta, matching the
            # C++ binding's own tuple order (cpp/src/bindings.cpp).
            p, d, g, v, th, _rho = bscpp.bs_price_with_greeks_batch_arrays(
                spot_arr[live], np.full(int(live.sum()), strike), rate_arr[live],
                np.full(int(live.sum()), self.dividend_yield), vols[live], t_arr[live], otype_arr)
            price_arr[live] = p
            delta_arr[live] = d
            gamma_arr[live] = g
            theta_arr[live] = th
            vega_arr[live] = v

        if is_expiry.any():
            if option_type == "call":
                price_arr[is_expiry] = np.maximum(spot_arr[is_expiry] - strike, 0.0)
            else:
                price_arr[is_expiry] = np.maximum(strike - spot_arr[is_expiry], 0.0)
            # delta/gamma/theta/vega already 0 for these rows.

        reported_vol = vols.copy()
        for i in range(1, n):
            if is_expiry[i]:
                reported_vol[i] = vols[i - 1]  # no re-mark at expiry: payoff has no vol

        return pd.DataFrame({
            "date": dates, "spot": spot_arr, "T": t_arr, "rate": rate_arr,
            "hedge_vol": reported_vol, "price": price_arr,
            "delta": delta_arr, "gamma": gamma_arr, "theta": theta_arr, "vega": vega_arr,
        })

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
        pricing = self.price_path(price_path, strike, expiration, hedge_vol, option_type)
        return self._run_from_pricing(dates, pricing, policy)

    def _run_from_pricing(self, dates: list, pricing: pd.DataFrame, policy) -> pd.DataFrame:
        """Policy-simulation half of `run()`: consumes an already-priced
        table (from `price_path()`) and drives the cash/shares/
        transaction-cost recursion. Factored out so a caller that needs
        the SAME price path under many different policies -- e.g.
        frontier.run_policy_grid, which reruns dozens of (risk_aversion,
        band_multiplier) policies per window -- can price the window ONCE
        and reuse the table, instead of repricing (a C++ crossing per day)
        once per policy for pricing that never depended on the policy in
        the first place.
        """
        cost_frac = self.transaction_cost_bps / 10_000.0
        spot_arr = pricing["spot"].to_numpy()
        t_arr = pricing["T"].to_numpy()
        rate_arr = pricing["rate"].to_numpy()
        vol_arr = pricing["hedge_vol"].to_numpy()
        price_arr = pricing["price"].to_numpy()
        delta_arr = pricing["delta"].to_numpy()
        gamma_arr = pricing["gamma"].to_numpy()
        theta_arr = pricing["theta"].to_numpy()
        vega_arr = pricing["vega"].to_numpy()

        spot0 = float(spot_arr[0])
        state0 = HedgeState(t=float(t_arr[0]), spot=spot0, delta=float(delta_arr[0]),
                            gamma=float(gamma_arr[0]), vega=float(vega_arr[0]),
                            rate=float(rate_arr[0]), cost_frac=cost_frac)
        shares = policy.target_shares(0.0, state0)
        cost0 = abs(shares) * spot0 * cost_frac  # crossing the spread to establish the hedge
        cash = float(price_arr[0]) - shares * spot0 - cost0
        spot_prev = spot0

        rows = [{
            "date": dates[0], "spot": spot0, "T": float(t_arr[0]), "delta": float(delta_arr[0]),
            "gamma": float(gamma_arr[0]), "theta": float(theta_arr[0]),
            "vega": float(vega_arr[0]), "hedge_vol": float(vol_arr[0]),
            "option_value": float(price_arr[0]), "cash": cash, "shares": shares,
            "transaction_cost": cost0,
            "portfolio_value": cash + shares * spot0 - float(price_arr[0]),
        }]

        for i in range(1, len(dates)):
            date = dates[i]
            spot = float(spot_arr[i])
            vol = float(vol_arr[i])
            t = float(t_arr[i])
            rate = float(rate_arr[i])
            dt_years = self.clock.elapsed(dates[i - 1], dates[i], floor_at_one_day=True)
            # Financing accrues over this step at the rate for the OPTION'S
            # remaining maturity as of this step (not a separate overnight/
            # repo rate -- see __init__'s docstring for why that's a
            # deliberate simplification, not an oversight).
            cash *= math.exp(rate * dt_years)
            # Dividend income on the stock leg carried into this interval
            # (held at `shares` since the last rebalance). Without this,
            # the hedge is inconsistent with using dividend_yield-adjusted
            # deltas from Black-Scholes, which price in the assumption that
            # the stock holder collects this yield.
            cash += shares * spot_prev * (math.exp(self.dividend_yield * dt_years) - 1.0)
            spot_prev = spot

            if t <= 0.0:
                payoff = float(price_arr[i])
                cost = abs(shares) * spot * cost_frac  # crossing the spread to unwind the hedge
                cash += shares * spot - cost  # liquidate the hedge; the unified formula
                shares, option_value = 0.0, payoff  # below nets out -option_value (=payoff) once
                delta, gamma, theta, vega = 0.0, 0.0, 0.0, 0.0  # degenerate at expiry
            else:
                option_value = float(price_arr[i])
                delta, gamma = float(delta_arr[i]), float(gamma_arr[i])
                theta, vega = float(theta_arr[i]), float(vega_arr[i])
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
            dt_years = self.clock.elapsed(df["date"].iat[i - 1], df["date"].iat[i],
                                           floor_at_one_day=True)
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
