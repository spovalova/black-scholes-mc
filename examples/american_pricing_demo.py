"""Demo: European vs. American pricing via Longstaff-Schwartz LSM.

No market data or API key needed -- pure pricing demo.

    python examples/american_pricing_demo.py
"""

import bscpp


def main():
    print("American call, no dividends -- should ~= European (never optimal to exercise early):")
    euro_call = bscpp.price(100, 100, 0.05, 0.2, 1.0, "call")
    amer_call = bscpp.price_american(100, 100, 0.05, 0.2, 1.0, "call", num_paths=50_000, num_steps=50)
    print(f"  European: {euro_call:.4f}")
    print(f"  American: {amer_call.price:.4f} +/- {amer_call.std_error:.4f}\n")

    print("American put -- Longstaff & Schwartz (2001) headline example (S=36,K=40,r=6%,vol=20%,T=1y):")
    euro_put = bscpp.price(36, 40, 0.06, 0.2, 1.0, "put")
    amer_put = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=100_000, num_steps=50)
    print(f"  European: {euro_put:.4f}")
    print(f"  American: {amer_put.price:.4f} +/- {amer_put.std_error:.4f}  (paper reports ~4.478)")
    print(f"  Early exercise premium: {amer_put.price - euro_put:.4f}")


if __name__ == "__main__":
    main()
