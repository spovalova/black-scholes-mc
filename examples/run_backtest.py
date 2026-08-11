"""Example: price a live option chain slice and (optionally) backtest it.

Usage:
    # No API key needed -- exercises the full pipeline against synthetic data:
    python examples/run_backtest.py --mock --ticker SPY

    # Real data via Polygon.io (requires POLYGON_API_KEY, "Options Starter"+ plan):
    python examples/run_backtest.py --ticker SPY --expiration 2026-09-19
"""

from __future__ import annotations

import argparse
import datetime as dt

from dotenv import load_dotenv

from bscpp.backtest import Backtester, MockProvider, PolygonProvider, StripPricer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--expiration", type=str, default=None,
                    help="YYYY-MM-DD; defaults to the nearest available expiration")
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--mc-paths", type=int, default=50_000)
    p.add_argument("--strike-low", type=float, default=0.85)
    p.add_argument("--strike-high", type=float, default=1.15)
    p.add_argument("--mock", action="store_true", help="use synthetic data, no API key required")
    p.add_argument("--backtest-days", type=int, default=0,
                    help="if > 0, also run a naive multi-day loop (see Backtester docstring caveat)")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.mock:
        provider = MockProvider(spot=450.0, base_vol=0.18)
    else:
        provider = PolygonProvider()  # reads POLYGON_API_KEY from env

    pricer = StripPricer(provider, rate=args.rate, mc_paths=args.mc_paths)

    if args.expiration:
        expiration = dt.date.fromisoformat(args.expiration)
    else:
        expirations = provider.get_expirations(args.ticker)
        if not expirations:
            raise SystemExit(f"No expirations found for {args.ticker}")
        expiration = expirations[0]

    print(f"Pricing {args.ticker} chain, expiration={expiration}")
    result = pricer.price_strip(
        args.ticker, expiration, strike_range=(args.strike_low, args.strike_high), use_mc=True
    )
    if result.empty:
        raise SystemExit("No contracts found in the requested strike range.")

    cols = ["strike", "type", "mid", "model_iv", "bs_price", "mc_price",
            "bs_error_vs_market", "delta", "gamma", "vega", "theta"]
    print(result[cols].to_string(index=False))

    if args.backtest_days > 0:
        dates = [dt.date.today() - dt.timedelta(days=d) for d in range(args.backtest_days)]
        backtester = Backtester(pricer)
        history = backtester.run(args.ticker, expiration, dates,
                                  strike_range=(args.strike_low, args.strike_high), use_mc=False)
        print("\nBacktest summary (see Backtester docstring re: historical-data caveats):")
        print(backtester.summary(history).to_string(index=False))


if __name__ == "__main__":
    main()
