"""Demo: net Greeks aggregation and limit breach checking across a
multi-underlying portfolio.

No market data or API key needed. See bscpp/risk.py for an explicit
statement of what this is (net-Greeks aggregation + limits) and is NOT
(a production risk system -- no live feed, no margin model, no kill switch).

    python examples/portfolio_risk_demo.py
"""

import bscpp


def main():
    positions = [
        bscpp.Position("AAPL call", "call", quantity=10, underlying="AAPL", spot=200, rate=0.05,
                        strike=210, vol=0.25, maturity=0.5),
        bscpp.Position("AAPL delta hedge", "stock", quantity=-500, underlying="AAPL", spot=200,
                        rate=0.05),
        bscpp.Position("MSFT put", "put", quantity=-20, underlying="MSFT", spot=400, rate=0.05,
                        strike=380, vol=0.22, maturity=0.25),
    ]

    mgr = bscpp.PortfolioRiskManager()

    print("Per-position Greeks:")
    print(mgr.position_greeks(positions).to_string(index=False))

    print("\nNet portfolio Greeks:")
    for k, v in mgr.net_greeks(positions).items():
        print(f"  {k}: {v:,.2f}")

    print("\nBy underlying:")
    print(mgr.greeks_by_underlying(positions).to_string(index=False))

    limits = bscpp.RiskLimits(max_abs_vega=50, per_underlying_max_abs_delta=100)
    mgr_with_limits = bscpp.PortfolioRiskManager(limits)
    breaches = mgr_with_limits.check_limits(positions)

    print(f"\nLimits: max_abs_vega={limits.max_abs_vega}, "
          f"per_underlying_max_abs_delta={limits.per_underlying_max_abs_delta}")
    if breaches:
        print("Breaches:")
        for b in breaches:
            print(f"  {b}")
    else:
        print("No breaches.")


if __name__ == "__main__":
    main()
