"""Demo: European vs. American pricing, CRR tree (production) cross-checked
against Longstaff-Schwartz LSM (Monte Carlo).

No market data or API key needed -- pure pricing demo.

    python examples/american_pricing_demo.py
"""

import bscpp


def main():
    print("American call, no dividends -- should ~= European (never optimal to exercise early):")
    euro_call = bscpp.price(100, 100, 0.05, 0.2, 1.0, "call")
    crr_call = bscpp.price_american_crr(100, 100, 0.05, 0.2, 1.0, "call", num_steps=500)
    lsm_call = bscpp.price_american(100, 100, 0.05, 0.2, 1.0, "call", num_paths=50_000, num_steps=50)
    print(f"  European:        {euro_call:.4f}")
    print(f"  American (CRR):  {crr_call:.4f}  (deterministic, ~15us at num_steps=200)")
    print(f"  American (LSM):  {lsm_call.price:.4f} +/- {lsm_call.std_error:.4f}\n")

    print("American put -- Longstaff & Schwartz (2001) headline example (S=36,K=40,r=6%,vol=20%,T=1y):")
    euro_put = bscpp.price(36, 40, 0.06, 0.2, 1.0, "put")
    crr_put = bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=1000)
    lsm_put = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=100_000, num_steps=50)
    print(f"  European:        {euro_put:.4f}")
    print(f"  American (CRR):  {crr_put:.4f}  "
          f"(independent implementations of this case converge to ~4.47-4.48)")
    print(f"  American (LSM):  {lsm_put.price:.4f} +/- {lsm_put.std_error:.4f}")
    print(f"  CRR vs LSM agreement: {abs(crr_put - lsm_put.price):.4f} "
          f"({abs(crr_put - lsm_put.price) / lsm_put.std_error:.1f} std errors)")
    print(f"  Early exercise premium (CRR): {crr_put - euro_put:.4f}\n")

    print("CRR is this project's production American pricer (deterministic, microseconds per")
    print("price -- see StripPricer(american=True)); LSM remains as an independent cross-check")
    print("and for path-dependent/multi-factor cases a binomial tree can't handle. See README's")
    print("'C++ pricing core' section.")


if __name__ == "__main__":
    main()
