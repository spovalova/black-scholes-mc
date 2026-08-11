# bscpp

Black-Scholes options pricing with a Monte Carlo engine, implemented in C++
and exposed to Python via pybind11, plus a Python backtesting layer that
prices live option chain slices against real market data.

## Layout

```
cpp/                  C++ core (analytic BS pricer, MC pricer, pybind11 bindings)
  include/bscpp/
  src/
python/bscpp/          Python package (imports the compiled extension)
  backtest/             data providers + pricing/backtest engine
tests/                 pytest suite (pricer correctness + backtest plumbing)
examples/run_backtest.py
```

## What's implemented

- **Analytic Black-Scholes-Merton** pricing (calls/puts, continuous dividend
  yield) with closed-form Greeks (delta, gamma, vega, theta, rho) and a
  Newton-Raphson (+ bisection fallback) implied-vol solver.
- **Monte Carlo European pricer** under GBM, with antithetic variates for
  variance reduction, and Greeks via bump-and-reprice using common random
  numbers (same draws reused across bumped scenarios).
- **Backtester**: a `DataProvider` interface (currently `PolygonProvider` for
  real data, `MockProvider` for synthetic/offline testing) feeding a
  `StripPricer` that prices a slice of an option chain with both BS and MC
  and reports theoretical price vs. observed market mid, plus Greeks.

Not implemented (out of scope for this pass): American-style / early
exercise pricing (would need LSM Monte Carlo), and a real historical
backtest loop with P&L simulation of a specific strategy — the `Backtester`
class here is a pricing-error-over-time scaffold you can extend.

## Setup

Requires a C++17 compiler (Xcode command line tools on macOS) and Python
3.9+. No CMake needed — the extension builds via `pybind11.setup_helpers`.

```bash
cd black-scholes-mc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/
```

## Real market data

The backtester targets [Polygon.io](https://polygon.io). Options chain data
requires at least their "Options Starter" plan (the free tier is
equities-only); historical chain snapshots for arbitrary past dates require
a higher tier still — see the `Backtester` docstring in
`python/bscpp/backtest/engine.py`.

Set your key via environment variable or a local `.env` (gitignored):

```bash
cp .env.example .env
# edit .env and set POLYGON_API_KEY=...
```

## Try it without an API key

```bash
python examples/run_backtest.py --mock --ticker SPY
```

This runs the full pipeline — fetch chain, solve implied vol, price with BS
+ MC, compare to "market" mid — against `MockProvider`'s synthetic chain
(which has a built-in volatility smile, so pricing error isn't trivially
zero).

## With a real Polygon key

```bash
python examples/run_backtest.py --ticker SPY --expiration 2026-09-19
```

## Python API quick reference

```python
import bscpp

# Analytic price / Greeks
price = bscpp.price(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, option_type="call")
inputs = bscpp.make_inputs(100, 100, 0.05, 0.2, 1.0, "call")
greeks = bscpp.bs_greeks(inputs)

# Monte Carlo
mc = bscpp.price_mc(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, num_paths=200_000)
print(mc.price, mc.std_error)

# Implied vol
iv = bscpp.bs_implied_vol(inputs, market_price=10.45)
```

```python
from bscpp.backtest import MockProvider, StripPricer

provider = MockProvider(spot=450.0, base_vol=0.18)
pricer = StripPricer(provider, rate=0.05)
chain = pricer.price_strip("SPY", expiration, strike_range=(0.9, 1.1))
```
