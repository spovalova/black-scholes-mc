"""Demo: calibrate Heston stochastic volatility to an option chain and
compare its smile fit against SVI on the same data.

No market data or API key needed -- uses MockProvider's built-in smile.

    python examples/heston_calibration_demo.py
"""

import datetime as dt

import bscpp
from bscpp.backtest import (
    MockProvider,
    StripPricer,
    calibrate_heston,
    calibrate_heston_with_stability,
    fit_svi_slice,
    heston_fit_rmse,
    svi_fit_rmse,
)


def main():
    spot, rate = 450.0, 0.05
    provider = MockProvider(spot=spot, base_vol=0.18, smile_strength=0.40)
    pricer = StripPricer(provider, rate=rate, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=45)

    # Narrower range, calls-only, on the SAME contracts for both fits -- an
    # apples-to-apples comparison, and it avoids the deep-tail strikes where
    # MockProvider's synthetic noise occasionally breaks the IV solver (see
    # StripPricer's fallback-to-0.20 behavior, documented in engine.py).
    chain = pricer.price_strip("SPY", expiration, strike_range=(0.85, 1.15), use_mc=False)
    calls = chain[chain["type"] == "call"]
    t_years = float(chain["T"].iloc[0])
    strikes = calls["strike"].to_numpy()
    option_types = calls["type"].tolist()
    market_ivs = calls["model_iv"].to_numpy()

    print(f"Calibrating to {len(calls)} call strikes, T={t_years:.3f}y ({45}d)\n")

    heston = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate)
    heston_rmse = heston_fit_rmse(heston, strikes, option_types, market_ivs, spot, t_years, rate)
    feller_ok = bscpp.heston_satisfies_feller_condition(heston)

    svi = fit_svi_slice(strikes, market_ivs, spot=spot, t_years=t_years, rate=rate)
    svi_rmse = svi_fit_rmse(svi, strikes, market_ivs, spot=spot, rate=rate)

    print(f"Heston (regularized): {heston}")
    print(f"  fit RMSE: {heston_rmse * 100:.3f} vol points")
    print(f"  Feller condition (2*kappa*theta >= xi^2): {'satisfied' if feller_ok else 'violated'}\n")

    print(f"SVI:    {svi}")
    print(f"  fit RMSE: {svi_rmse * 100:.3f} vol points\n")

    # This exact short-dated, mildly-curved smile is a known case where the
    # UNREGULARIZED fit drives v0 to its lower bound -- a real
    # identifiability failure mode, not a solver bug. Show both, and the
    # multi-start stability diagnostic that actually quantifies it, rather
    # than just asserting the regularized fit is fine.
    unregularized = calibrate_heston(strikes, option_types, market_ivs, spot, t_years, rate,
                                      regularization_weight=0.0)
    print(f"Unregularized fit for comparison: v0={unregularized.v0:.6f} "
          f"(vs. regularized v0={heston.v0:.6f}, ATM variance ~{market_ivs[len(market_ivs)//2]**2:.6f})")
    print("-> regularization pulls v0 out of the degenerate near-zero corner without materially "
          "changing fit quality; nothing about the RAW data pins v0 down on its own here.\n")

    print("Multi-start stability diagnostic (6 random initial guesses):")
    stability = calibrate_heston_with_stability(strikes, option_types, market_ivs, spot, t_years,
                                                 rate, n_starts=6)
    print(f"  all_rmse: {[round(r, 5) for r in stability['all_rmse']]}")
    print(f"  param_std: {{{', '.join(f'{k}: {v:.5f}' for k, v in stability['param_std'].items())}}}")
    print(f"  fit_quality_stable: {stability['fit_quality_stable']}   "
          f"params_stable: {stability['params_stable']}\n")

    print("Heston's edge over SVI: it's a full DYNAMIC model (the smile evolves consistently "
          "as spot/time move, since it comes from an actual SDE) rather than a snapshot curve "
          "fit -- useful for exotic/path-dependent pricing and for forward-starting options, "
          "where SVI has nothing to say. SVI's edge: cheaper to fit, no identifiability issues, "
          "and it can match an arbitrary smile shape more exactly since it isn't constrained "
          "to what a 5-parameter SDE can produce.")


if __name__ == "__main__":
    main()
