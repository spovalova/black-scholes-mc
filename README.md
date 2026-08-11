# bscpp

Options pricing and quantitative research toolkit: a C++ pricing core
(analytic Black-Scholes, Monte Carlo, and Longstaff-Schwartz American
pricing) exposed to Python via pybind11, plus a Python layer that prices
live option chains against real market data, fits implied vol surfaces,
and simulates delta-hedging P&L.

## Layout

```
cpp/                          C++ core
  include/bscpp/
    types.hpp                  shared structs (MarketInputs, Greeks, MCResult)
    black_scholes.hpp           analytic BS pricer, Greeks, implied-vol solver
    monte_carlo.hpp              European MC pricer (antithetic + CRN Greeks)
    longstaff_schwartz.hpp        American MC pricer (LSM regression)
  src/                          implementations + pybind11 bindings

python/bscpp/                 Python package (imports the compiled extension)
  __init__.py                  Pythonic wrappers around the C++ API
  backtest/
    data_provider.py            DataProvider interface: PolygonProvider, MockProvider
    engine.py                    StripPricer / Backtester -- chain pricing vs. market
    hedging.py                   HedgingBacktester -- delta-hedging P&L simulation
    vol_surface.py                SVI implied-vol smile fitting

tests/                        pytest suite (17 tests: pricer correctness,
                               benchmark comparisons, backtest plumbing)
examples/                     runnable demos, none require an API key except
                               run_backtest.py without --mock
```

## What's implemented, and why it's not a toy

**C++ pricing core**
- Closed-form Black-Scholes-Merton (calls/puts, continuous dividend yield),
  full Greeks (delta, gamma, vega, theta, rho), and a Newton-Raphson +
  bisection-fallback implied-vol solver.
- Monte Carlo European pricer under GBM: antithetic variates for variance
  reduction, Greeks via bump-and-reprice with **common random numbers**
  (the same underlying draws reused across bumped scenarios, which is what
  keeps finite-difference Greeks from being unusably noisy).
- **American-style pricing via Longstaff-Schwartz (2001) least-squares
  Monte Carlo**: simulates full paths, then walks backward from maturity
  regressing realized continuation value onto a polynomial basis to decide
  optimal early exercise at each step. Validated against the paper's own
  benchmark (S=36, K=40, r=6%, vol=20%, T=1y -> published ~4.478) and
  against the textbook no-arbitrage fact that an American call with no
  dividends should price identically to its European counterpart.

**Python quant/backtesting layer**
- `StripPricer` / `Backtester`: pull a chain slice from a `DataProvider`,
  solve implied vol per contract, price with both BS and MC, and report
  theoretical price vs. observed market mid plus full Greeks.
- `HedgingBacktester` + `realized_vs_implied_experiment`: simulates
  selling an option and delta-hedging it daily at a given `hedge_vol`
  against a real or simulated price path. This is the actual "does the
  model make money" question a market maker asks, and it recovers the
  correct, non-obvious result: hedging at a vol *above* what actually
  realizes is profitable for the option seller, hedging below it is not,
  and P&L crosses zero right around `realized_vol == hedge_vol` (see the
  worked example below).
- `fit_svi_slice`: fits Gatheral's SVI parameterization
  (`w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))`, total variance vs.
  log-moneyness) to a chain's implied vols per expiry -- the same smile
  parameterization used across the industry, chosen over a raw polynomial
  fit because it stays well-behaved across a much wider strike range.

**Deliberately out of scope** (documented, not silently missing): stochastic
vol / local vol models (Heston, SABR), a full historical options-tick
database integration, and P&L simulation of *specific* multi-leg strategies
(straddles, verticals, etc.) beyond the generic chain-slice and
single-option-hedging tools above -- those are natural next extensions on
top of the `DataProvider` / pricer primitives here.

## Setup

Requires a C++17 compiler (Xcode command line tools on macOS) and Python
3.9+. No CMake needed -- the extension builds via `pybind11.setup_helpers`.

```bash
cd black-scholes-mc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/
```

## Run the demos (no API key needed)

```bash
python examples/american_pricing_demo.py     # European vs. American (LSM) pricing
python examples/hedging_pnl_experiment.py     # realized-vs-implied vol P&L sweep
python examples/vol_surface_fit_demo.py       # SVI fit to a synthetic smile
python examples/run_backtest.py --mock --ticker SPY   # full chain-pricing pipeline
```

Sample output from `hedging_pnl_experiment.py` (selling an ATM call, hedging
daily at 30% vol, 60 days to expiry, 400 simulated paths per realized-vol
level):

```
 realized_vol  hedge_vol  mean_hedging_pnl  std_hedging_pnl
         0.10       0.30              3.19             0.66
         0.20       0.30              1.61             0.61
         0.30       0.30             -0.01             0.59
         0.40       0.30             -1.64             1.06
         0.50       0.30             -3.24             1.75
```

## Real market data

The backtester targets [Polygon.io](https://polygon.io). Options chain
data requires at least their "Options Starter" plan (the free tier is
equities-only); daily underlying price history (`get_price_history`, used
by the hedging backtest) works on the base equities tier. Historical chain
*snapshots* for arbitrary past dates require a higher tier still -- see the
`Backtester` docstring in `python/bscpp/backtest/engine.py`.

Set your key via environment variable or a local `.env` (gitignored):

```bash
cp .env.example .env
# edit .env and set POLYGON_API_KEY=...
```

Then:

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

# Monte Carlo (European)
mc = bscpp.price_mc(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, num_paths=200_000)
print(mc.price, mc.std_error)

# American, via Longstaff-Schwartz LSM
am = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=100_000, num_steps=50)

# Implied vol
iv = bscpp.bs_implied_vol(inputs, market_price=10.45)
```

```python
from bscpp.backtest import MockProvider, StripPricer, HedgingBacktester, fit_svi_slice

provider = MockProvider(spot=450.0, base_vol=0.18)
pricer = StripPricer(provider, rate=0.05)
chain = pricer.price_strip("SPY", expiration, strike_range=(0.9, 1.1))

svi = fit_svi_slice(chain["strike"], chain["model_iv"], spot=450.0, t_years=chain["T"].iloc[0])

hedger = HedgingBacktester(rate=0.05)
pnl = hedger.run(price_history, strike=450, expiration=expiration, hedge_vol=0.18)
```
