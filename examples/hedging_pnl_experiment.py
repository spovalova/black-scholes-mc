"""Demo: realized-vs-implied vol delta-hedging P&L.

Simulates selling and delta-hedging an option at a fixed hedge_vol across
many GBM paths at different realized vols, and shows the resulting mean
hedging P&L flips sign around realized_vol == hedge_vol -- the basic
mechanic behind selling implied vol / buying realized vol.

No market data or API key needed.

    python examples/hedging_pnl_experiment.py
"""

from bscpp.backtest import realized_vs_implied_experiment


def main():
    hedge_vol = 0.30
    common = dict(hedge_vol=hedge_vol, realized_vols=[0.10, 0.20, 0.30, 0.40, 0.50], spot=100,
                  strike=100, rate=0.05, t_days=60, n_paths_per_vol=400, seed=42)

    frictionless = realized_vs_implied_experiment(**common, transaction_cost_bps=0)
    with_cost = realized_vs_implied_experiment(**common, transaction_cost_bps=10)

    print(f"Hedging a short ATM call at hedge_vol={hedge_vol}, 60 days to expiry, 400 paths/level:\n")
    print("Frictionless (transaction_cost_bps=0):")
    print(frictionless.to_string(index=False))
    print(
        "\nNote the sign flip around realized_vol == hedge_vol: hedging at a vol richer "
        "than what actually realizes is where the P&L comes from."
    )

    print("\nWith a realistic 10bps transaction cost on every rebalance:")
    print(with_cost.to_string(index=False))
    shift = frictionless["mean_hedging_pnl"] - with_cost["mean_hedging_pnl"]
    print(f"\nP&L reduction from costs alone: {shift.round(3).tolist()}")
    print("Costs bite hardest at high realized vol -- more rebalancing trades happen "
          "when the underlying moves more, so the drag scales with realized vol, not "
          "just with hedge_vol.")


if __name__ == "__main__":
    main()
