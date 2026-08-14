"""Demo: fit an SVI implied-vol slice to a (synthetic) option chain.

No market data or API key needed -- uses MockProvider's built-in smile.
To fit against a real chain instead, swap MockProvider for PolygonProvider
and pass its `implied_volatility` column straight through (skip the
per-contract IV solve StripPricer does when the feed already supplies IVs).

    python examples/vol_surface_fit_demo.py
"""

import datetime as dt

import numpy as np

from bscpp.backtest import (
    MockProvider,
    StripPricer,
    fit_svi_slice,
    svi_butterfly_arbitrage_check,
    svi_fit_rmse,
    svi_min_total_variance,
)


def main():
    provider = MockProvider(rate=0.05, spot=450.0, base_vol=0.18, smile_strength=0.40)
    pricer = StripPricer(provider, rate=0.05, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=45)

    chain = pricer.price_strip("SPY", expiration, strike_range=(0.75, 1.25), use_mc=False)
    # OTM-only smile: StripPricer solves calls above the implied forward
    # and puts below (see engine.extract_forward_and_carry), so the ITM
    # leg at each strike is a deliberate NaN fallback -- dropping it (not
    # filtering to "calls" alone, which would silently discard the entire
    # ITM-call half of the range) keeps every strike, just from whichever
    # side is actually OTM there.
    smile = chain.dropna(subset=["model_iv"]).sort_values("strike")
    t_years = float(chain["T"].iloc[0])
    forward = float(chain["implied_forward"].iloc[0])

    svi = fit_svi_slice(smile["strike"], smile["model_iv"], spot=450.0, t_years=t_years, rate=0.05)
    rmse = svi_fit_rmse(svi, smile["strike"], smile["model_iv"], spot=450.0, rate=0.05)

    print(f"SVI fit ({len(smile)} strikes, OTM-only, T={t_years:.3f}y, "
          f"implied forward={forward:.2f}):")
    print(f"  a={svi.a:.5f}  b={svi.b:.5f}  rho={svi.rho:.4f}  m={svi.m:.4f}  sigma={svi.sigma:.4f}")
    print(f"  fit RMSE: {rmse * 100:.2f} vol points")

    print(f"  min total variance: {svi_min_total_variance(svi):.5f} (must be >= 0)")
    arb = svi_butterfly_arbitrage_check(svi, spot=450.0, rate=0.05)
    print(f"  Breeden-Litzenberger butterfly check: min density = {arb['min_density']:.6f} "
          f"-> {'arbitrage-free' if arb['arbitrage_free'] else 'ARBITRAGE VIOLATION'}\n")

    print(f"{'strike':>8} {'type':>5} {'market_iv':>10} {'svi_iv':>10}")

    for _, row in smile.iterrows():
        k = np.log(row["strike"] / forward)
        svi_iv = float(svi.implied_vol(k))
        print(f"{row['strike']:>8.1f} {row['type']:>5} {row['model_iv']:>10.4f} {svi_iv:>10.4f}")


if __name__ == "__main__":
    main()
