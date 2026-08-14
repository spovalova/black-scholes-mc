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
    provider = MockProvider(rate=0.05, spot=spot, base_vol=0.18, smile_strength=0.40)
    pricer = StripPricer(provider, rate=rate, mc_paths=1)
    days_to_expiry = 30  # not 45: with the corrected OTM-only smile, 45 days
    # no longer reproduces the v0-degeneracy failure mode this demo exists to
    # show (see test_heston.py's _short_dated_mock_chain for the same fix
    # and a sweep of maturities that do/don't reproduce it).
    expiration = dt.date.today() + dt.timedelta(days=days_to_expiry)

    # Narrower range, OTM-only smile, on the SAME contracts for both fits --
    # an apples-to-apples comparison. StripPricer solves calls above the
    # implied forward and puts below (see engine.extract_forward_and_carry);
    # dropping the ITM-fallback NaN rows (not filtering to "calls" alone,
    # which would silently discard the whole ITM-call half of the range)
    # keeps every strike, each from whichever side is actually OTM there.
    chain = pricer.price_strip("SPY", expiration, strike_range=(0.85, 1.15), use_mc=False)
    smile = chain.dropna(subset=["model_iv"]).sort_values("strike")
    t_years = float(chain["T"].iloc[0])
    strikes = smile["strike"].to_numpy()
    option_types = smile["type"].tolist()
    market_ivs = smile["model_iv"].to_numpy()

    print(f"Calibrating to {len(smile)} strikes (OTM-only), T={t_years:.3f}y ({days_to_expiry}d)\n")

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
