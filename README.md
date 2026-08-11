# bscpp

A derivatives pricing and quantitative research toolkit: a C++ pricing
core (analytic Black-Scholes, Monte Carlo, Longstaff-Schwartz American
pricing) exposed to Python via pybind11, plus a Python layer that prices
live option chains against real market data, constructs and analyzes
multi-leg strategies, fits arbitrage-checked implied vol surfaces, and
simulates delta-hedging P&L with a full Greeks attribution.

## Layout

```
cpp/                          C++ core
  include/bscpp/
    types.hpp                  shared structs (MarketInputs, Greeks, MCResult)
    black_scholes.hpp           analytic BS pricer, Greeks, IV solver, batch variants
    monte_carlo.hpp              European MC pricer (antithetic + CRN Greeks)
    longstaff_schwartz.hpp        American MC pricer (LSM regression)
  src/                          implementations + pybind11 bindings

python/bscpp/                 Python package (imports the compiled extension)
  __init__.py                  Pythonic wrappers, trading_greeks() unit conversion
  strategies.py                 multi-leg strategies: straddle/strangle/vertical/
                                 strip/strap/butterfly, net Greeks, exact breakevens
  backtest/
    data_provider.py            DataProvider interface: PolygonProvider, MockProvider
    engine.py                    StripPricer / Backtester -- chain pricing vs. market
    hedging.py                   HedgingBacktester -- delta-hedging P&L + attribution
    vol_surface.py                SVI fitting + Breeden-Litzenberger arbitrage checks

tests/                        pytest suite (28 tests): closed-form benchmarks,
                               convergence behavior, arbitrage-free checks,
                               accounting-identity checks, backtest plumbing
examples/                     runnable demos -- all but run_backtest.py (non-mock)
                               and real_data_hedging_demo.py need no API key
```

## What's implemented, and why it isn't a toy

**C++ pricing core**
- Closed-form Black-Scholes-Merton (calls/puts, continuous dividend yield),
  full Greeks (delta, gamma, vega, theta, rho), and a Newton-Raphson +
  bisection-fallback implied-vol solver.
- **Batch pricing/IV variants** (`bs_price_with_greeks_batch`,
  `bs_implied_vol_batch`) that loop in C++ rather than crossing the
  Python/C++ boundary once per contract -- `StripPricer` prices a whole
  chain slice in one call instead of one per row.
- Monte Carlo European pricer under GBM: antithetic variates for variance
  reduction, Greeks via bump-and-reprice with **common random numbers**
  (same underlying draws reused across bumped scenarios -- what keeps
  finite-difference Greeks from being unusably noisy). Verified: MC
  standard error shrinks ~4x when path count goes up 16x, matching the
  theoretical O(1/sqrt(N)) Monte Carlo convergence rate.
- **American-style pricing via Longstaff-Schwartz (2001) least-squares
  Monte Carlo**: simulates full paths, walks backward from maturity
  regressing realized continuation value onto a polynomial basis to decide
  optimal early exercise. Validated against the paper's own benchmark
  (S=36, K=40, r=6%, vol=20%, T=1y -> published ~4.478) and against the
  no-arbitrage identity that an American call with no dividends prices
  identically to its European counterpart.

**Multi-leg strategies** (`bscpp.strategies`)
- `straddle`, `strangle`, `vertical_spread`, `butterfly`, and the two
  strategies that give "options strip" its literal meaning: `strip` (long
  1 call + 2 puts, bearish-biased) and `strap` (long 2 calls + 1 put,
  bullish-biased).
- `StrategyPricer.price()` returns net Greeks across all legs (supports a
  per-strike vol dict for skew-aware pricing, e.g. sourced from a fitted
  SVI slice). `payoff_diagram()` computes the expiration P&L curve and
  **exact breakevens** -- since a vanilla-option portfolio's payoff is
  piecewise-*linear* in spot with kinks only at each leg's strike, the
  breakevens are found by linear interpolation between kink points, not a
  numerical root-finder. Verified: straddle breakevens land exactly at
  `K +/- premium`; vertical spread max gain/loss land exactly at
  `width - debit` / `-debit`.

**Implied vol surface** (`bscpp.backtest.vol_surface`)
- `fit_svi_slice`: least-squares fit of Gatheral's SVI parameterization
  (`w(k) = a + b(rho(k-m) + sqrt((k-m)^2+sigma^2))`, total variance vs.
  log-moneyness) to a chain's implied vols. Verified to recover a known
  synthetic smile to <1e-4 RMSE.
- `svi_butterfly_arbitrage_check`: rather than trusting a memorized
  closed-form arbitrage condition on the SVI parameters, this prices calls
  off the fitted smile with the (already-tested) BS pricer across a strike
  grid and takes a finite-difference second derivative -- directly
  applying **Breeden-Litzenberger (1978)**: the risk-neutral density
  `q(K) = e^{rT} d^2C/dK^2` must be non-negative everywhere, or the smile
  implies a butterfly arbitrage. Verified against both a well-behaved
  slice (passes) and a deliberately pathological one (extreme rho, tiny
  sigma -- correctly flagged as arbitrage-violating).

**Delta-hedging P&L + Greeks attribution** (`bscpp.backtest.hedging`)
- `HedgingBacktester`: simulates selling an option and delta-hedging it
  daily at a given `hedge_vol` against a real or simulated price path.
- `realized_vs_implied_experiment`: sweeps realized vol against a fixed
  hedge_vol and shows the textbook result -- hedging at a vol *above* what
  actually realizes is profitable for the seller, *below* it is not, P&L
  crosses zero right at `realized_vol == hedge_vol` (see worked example
  below).
- `attribute_pnl`: decomposes each day's hedging P&L into financing +
  gamma + theta terms, derived from a second-order Taylor expansion of the
  option's own pricing function (full derivation in the method's
  docstring). This recovers the classic "theta pays you, gamma costs you"
  identity for a short, hedged option position, and is verified two ways:
  the decomposition's daily sum exactly equals the simulation's own
  cumulative P&L (an accounting identity, checked to float precision), and
  the financing+gamma+theta prediction explains the large majority of
  realized P&L with a bounded higher-order residual.

**Chain pricing vs. real market data** (`bscpp.backtest.engine`)
- `StripPricer`/`Backtester`: pull a chain slice from a `DataProvider`,
  solve implied vol per contract (batched), price with both BS and MC, and
  report theoretical price vs. observed market mid plus full Greeks.

**Trading-desk Greeks** (`bscpp.trading_greeks`)
- The raw Greeks are calculus derivatives: vega/rho per 1.00 (100 points)
  of vol/rate, theta per *year*. No desk quotes them that way. This is a
  one-line unit conversion (vega, rho /100; theta /365) that exists because
  mixing the two conventions is a classic, costly mistake for anyone new
  to reading options risk.

**Deliberately out of scope** (documented, not silently missing):
stochastic vol / local vol models (Heston, SABR), a full historical
options-tick database integration, and a full backtest P&L simulation of
*multi-leg* strategies over time (the hedging backtest covers single-leg
positions; extending `HedgingBacktester` to net multi-leg Greeks is a
natural next step on top of `strategies.py`).

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
python examples/strategy_demo.py               # straddle/strangle/vertical/strip/strap/butterfly
python examples/hedging_pnl_experiment.py       # realized-vs-implied vol P&L sweep
python examples/vol_surface_fit_demo.py         # SVI fit + arbitrage check on a synthetic smile
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

## Run against real data (needs an API key, no options entitlement required)

```bash
python examples/real_data_hedging_demo.py --ticker AAPL --hedge-vol 0.30
```

This pulls real historical daily closes, runs the delta-hedging backtest
against them, and prints the P&L attribution -- confirmed working on
Polygon/Massive's **base** equities tier (their daily aggregates endpoint
doesn't require an options plan). Sample attribution output from a real
run against AAPL:

```
financing_pnl       -1.67
gamma_pnl           -7.75
theta_pnl            7.77
predicted_pnl       -1.66
realized_pnl        -1.94
attribution_error   -0.28
```

`run_backtest.py`/`StripPricer` (chain snapshots) is the one demo that does
need an options-capable plan -- see below.

## Real market data

The backtester targets Polygon.io's REST API (the company has since
rebranded to "Massive", but the API itself is unchanged -- `PolygonProvider`
still hits `api.polygon.io` and that's confirmed working). Options chain
data requires at least an "Options Starter" plan (a base-tier key gets a
`403 NOT_AUTHORIZED` on `/v3/snapshot/options/...`, confirmed against a live
key); daily underlying price history (`get_price_history`, used by the
hedging backtest) works fine on the base equities tier, confirmed against
real AAPL data. Historical chain *snapshots* for arbitrary past dates
require a higher tier still -- see the `Backtester` docstring in
`python/bscpp/backtest/engine.py`.

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
desk_greeks = bscpp.trading_greeks(greeks)  # vega/rho per 1%, theta per day

# Monte Carlo (European) and American (Longstaff-Schwartz LSM)
mc = bscpp.price_mc(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0, num_paths=200_000)
am = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=100_000, num_steps=50)

# Multi-leg strategies
strat_pricer = bscpp.StrategyPricer(rate=0.05)
result = strat_pricer.price(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)
plot_df, breakevens = strat_pricer.payoff_diagram(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)
```

```python
from bscpp.backtest import (
    MockProvider, StripPricer, HedgingBacktester,
    fit_svi_slice, svi_butterfly_arbitrage_check,
)

provider = MockProvider(spot=450.0, base_vol=0.18)
pricer = StripPricer(provider, rate=0.05)
chain = pricer.price_strip("SPY", expiration, strike_range=(0.9, 1.1))

svi = fit_svi_slice(chain["strike"], chain["model_iv"], spot=450.0, t_years=chain["T"].iloc[0])
arb = svi_butterfly_arbitrage_check(svi, spot=450.0, rate=0.05)

hedger = HedgingBacktester(rate=0.05)
result = hedger.run(price_history, strike=450, expiration=expiration, hedge_vol=0.18)
attributed = hedger.attribute_pnl(result)  # financing / gamma / theta P&L breakdown
```
