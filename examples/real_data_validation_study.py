"""Multi-ticker, multi-window validation of the realized-vs-implied vol
hedging P&L relationship against REAL historical prices.

The single-window, single-ticker AAPL demo (real_data_hedging_demo.py) is a
useful worked example but is an anecdote, not evidence -- one path, one
strike, one 60-day window proves nothing statistically. This script fixes
that: it pulls real daily closes for several liquid names, rolls a
trailing-realized-vol-as-hedge-vol strategy forward across many
overlapping windows per ticker (a realistic, honest use of ONLY real data
-- no synthetic implied vol is invented, since the current API tier has no
options entitlement), and reports whether the sign relationship between
(hedge_vol - forward realized vol) and hedging P&L holds up in aggregate,
not just in one cherry-pickable window.

Requires POLYGON_API_KEY (base equities tier -- no options entitlement
needed, since only get_price_history/aggs is used).

    python examples/real_data_validation_study.py
"""

import datetime as dt

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from bscpp.backtest import HedgingBacktester, PolygonProvider

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
LOOKBACK_DAYS = 420  # calendar days of history to pull per ticker
WINDOW_DAYS = 45  # length of each hedging window (and of the trailing vol lookback)
STRIDE_DAYS = 20  # step between window starts (overlapping windows, more data points)


def annualized_realized_vol(closes: pd.Series) -> float:
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std(ddof=1) * np.sqrt(365))


def run_ticker(provider: PolygonProvider, ticker: str, rate: float) -> list[dict]:
    end = dt.date.today()
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    history = provider.get_price_history(ticker, start, end)
    if len(history) < WINDOW_DAYS * 2 + 5:
        return []

    backtester = HedgingBacktester(rate=rate)
    rows = []
    # start after one full WINDOW_DAYS so there's a trailing window to
    # estimate hedge_vol from before the hedged window even begins
    start_idx = WINDOW_DAYS
    while start_idx + WINDOW_DAYS < len(history):
        trailing = history.iloc[start_idx - WINDOW_DAYS: start_idx]
        window = history.iloc[start_idx: start_idx + WINDOW_DAYS]

        hedge_vol = annualized_realized_vol(trailing)
        forward_realized_vol = annualized_realized_vol(window)
        if not (0.02 < hedge_vol < 3.0):  # skip degenerate trailing-vol estimates
            start_idx += STRIDE_DAYS
            continue

        spot0 = float(window.iloc[0])
        strike = round(spot0 / 5) * 5
        expiration = window.index[-1].date()

        try:
            result = backtester.run(window, strike=strike, expiration=expiration,
                                     hedge_vol=hedge_vol, option_type="call")
        except Exception:
            start_idx += STRIDE_DAYS
            continue

        rows.append({
            "ticker": ticker,
            "window_start": window.index[0].date(),
            "hedge_vol": hedge_vol,
            "forward_realized_vol": forward_realized_vol,
            "vol_gap": hedge_vol - forward_realized_vol,
            "hedging_pnl": result["portfolio_value"].iloc[-1],
        })
        start_idx += STRIDE_DAYS

    return rows


def main():
    load_dotenv()
    provider = PolygonProvider()
    rate = 0.05

    all_rows = []
    for ticker in TICKERS:
        rows = run_ticker(provider, ticker, rate)
        print(f"{ticker}: {len(rows)} windows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if len(df) < 10:
        raise SystemExit(f"Only {len(df)} windows collected -- not enough for a meaningful study.")

    print(f"\n{len(df)} total ticker-windows across {len(TICKERS)} tickers\n")
    print(df.to_string(index=False))

    hit_rate = float((np.sign(df["vol_gap"]) == np.sign(df["hedging_pnl"])).mean())
    corr, pvalue = stats.pearsonr(df["vol_gap"], df["hedging_pnl"])

    print(f"\nHit rate (sign(hedge_vol - forward_realized_vol) == sign(hedging_pnl)): "
          f"{hit_rate:.1%} across {len(df)} windows")
    print(f"Correlation(vol_gap, hedging_pnl): r={corr:.3f}, p={pvalue:.4f}")
    print("\nTheory predicts a POSITIVE correlation and hit rate meaningfully above 50%: "
          "hedging at a vol richer than what subsequently realizes should be profitable for "
          "the seller. This is now checked out-of-sample across many real windows and several "
          "tickers, not asserted from one hand-picked path.")


if __name__ == "__main__":
    main()
