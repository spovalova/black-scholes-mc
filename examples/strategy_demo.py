"""Demo: multi-leg options strategies -- straddle, vertical spread, strip, strap.

Prices each with the C++ Black-Scholes core, reports net Greeks, and
computes exact breakevens from the piecewise-linear expiration payoff.

No market data or API key needed.

    python examples/strategy_demo.py
"""

import bscpp


def report(name, legs, spot, vol, maturity, pricer):
    result = pricer.price(legs, spot, vol, maturity)
    _, breakevens = pricer.payoff_diagram(legs, spot, vol, maturity)
    be_str = ", ".join(f"{b:.2f}" for b in breakevens) or "none in range"
    print(f"{name}:")
    print(f"  net_price={result.net_price:8.4f}  delta={result.net_delta:7.4f}  "
          f"gamma={result.net_gamma:7.4f}  vega={result.net_vega:8.4f}  theta={result.net_theta:8.4f}")
    print(f"  breakeven(s): {be_str}\n")


def main():
    spot, vol, maturity = 100.0, 0.25, 0.5
    pricer = bscpp.StrategyPricer(rate=0.05)

    report("Straddle @100", bscpp.straddle(100), spot, vol, maturity, pricer)
    report("Strangle 95/105 (put/call)", bscpp.strangle(call_strike=105, put_strike=95),
           spot, vol, maturity, pricer)
    report("Call vertical (long 95, short 105)",
           bscpp.vertical_spread("call", long_strike=95, short_strike=105), spot, vol, maturity, pricer)
    report("Strip @100 (1 call + 2 puts, bearish-biased)", bscpp.strip(100), spot, vol, maturity, pricer)
    report("Strap @100 (2 calls + 1 put, bullish-biased)", bscpp.strap(100), spot, vol, maturity, pricer)
    report("Butterfly 90/100/110 (call)", bscpp.butterfly("call", 90, 100, 110), spot, vol, maturity, pricer)


if __name__ == "__main__":
    main()
