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
    df = realized_vs_implied_experiment(
        hedge_vol=hedge_vol,
        realized_vols=[0.10, 0.20, 0.30, 0.40, 0.50],
        spot=100,
        strike=100,
        rate=0.05,
        t_days=60,
        n_paths_per_vol=400,
        seed=42,
    )
    print(f"Hedging a short ATM call at hedge_vol={hedge_vol}, 60 days to expiry, 400 paths/level:\n")
    print(df.to_string(index=False))
    print(
        "\nNote the sign flip around realized_vol == hedge_vol: hedging at a vol richer "
        "than what actually realizes is where the P&L comes from."
    )


if __name__ == "__main__":
    main()
