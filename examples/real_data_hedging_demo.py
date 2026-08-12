"""Demo: delta-hedging P&L against REAL historical prices via Polygon/Massive.

Requires POLYGON_API_KEY (see .env.example) -- daily aggregates work on
Polygon's base equities tier, no options entitlement needed. (Chain-based
demos like run_backtest.py without --mock DO require an options-capable
plan; this one doesn't.)

    python examples/real_data_hedging_demo.py --ticker AAPL --hedge-vol 0.30
"""

import argparse
import datetime as dt

from dotenv import load_dotenv

from bscpp.backtest import HedgingBacktester, PolygonProvider


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--hedge-vol", type=float, default=0.30,
                    help="constant vol used to compute BS deltas each day")
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--option-type", choices=["call", "put"], default="call")
    p.add_argument("--transaction-cost-bps", type=float, default=5.0,
                    help="cost of crossing the spread on each rebalance, in bps of trade notional")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    provider = PolygonProvider()
    end = dt.date.today()
    start = end - dt.timedelta(days=args.lookback_days)
    history = provider.get_price_history(args.ticker, start, end)
    if history.empty:
        raise SystemExit(f"No price history returned for {args.ticker} in [{start}, {end}]")

    spot0 = float(history.iloc[0])
    strike = round(spot0 / 5) * 5  # nearest $5 strike, roughly ATM at inception
    expiration = history.index[-1].date()

    print(f"{args.ticker}: {len(history)} closes, {history.index[0].date()} -> {expiration}")
    print(f"Spot at inception: {spot0:.2f}, strike: {strike}, hedge_vol: {args.hedge_vol}, "
          f"transaction_cost_bps: {args.transaction_cost_bps}\n")

    backtester = HedgingBacktester(rate=args.rate, transaction_cost_bps=args.transaction_cost_bps)
    result = backtester.run(history, strike=strike, expiration=expiration,
                             hedge_vol=args.hedge_vol, option_type=args.option_type)

    print(result[["date", "spot", "delta", "option_value", "transaction_cost",
                   "portfolio_value"]].to_string(index=False))
    print(f"\nFinal hedging P&L (short {strike} {args.option_type}, "
          f"hedged daily at {args.hedge_vol:.0%} vol): {result['portfolio_value'].iloc[-1]:.2f}")
    print(f"Total transaction costs paid: {result['transaction_cost'].sum():.2f}")

    attributed = backtester.attribute_pnl(result)
    print("\nP&L attribution (financing + gamma + theta + transaction cost; "
          "see HedgingBacktester.attribute_pnl):")
    totals = attributed[["financing_pnl", "gamma_pnl", "theta_pnl", "transaction_cost_pnl",
                          "predicted_pnl", "realized_pnl", "attribution_error"]].sum()
    print(totals.to_string())


if __name__ == "__main__":
    main()
