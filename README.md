# bscpp

[![CI](https://github.com/spovalova/black-scholes-mc/actions/workflows/ci.yml/badge.svg)](https://github.com/spovalova/black-scholes-mc/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

A derivatives pricing and quantitative research toolkit: a C++ pricing core
(analytic Black-Scholes, Monte Carlo, Longstaff-Schwartz American, Heston
stochastic vol) exposed to Python via pybind11, plus a Python layer that
prices live option chains against real market data, constructs and analyzes
multi-leg strategies, fits arbitrage-checked implied vol surfaces, simulates
delta-hedging P&L with pluggable rebalancing policies and full attribution,
and aggregates portfolio-level risk. See [CHANGELOG.md](CHANGELOG.md) for
version history.

## Research finding

Most of this project replicates known results correctly (Black-Scholes,
Heston, SVI, LSM) -- useful for learning, not itself a contribution. One
piece goes further: **is Whalley & Wilmott's (1997) asymptotically-optimal
hedging band actually cost-risk-minimizing on real market data, or only
under the continuous-monitoring assumption it's derived under?**

`examples/hedging_policy_frontier_study.py` sweeps the WW band width
(via the exact identity `band(risk_aversion = lam0/c^3) = c * band(lam0)`,
regression-tested in `test_policies.py`) across real daily closes for 5
liquid names, 50 rolling out-of-sample windows, and 3 risk-aversion
regimes, scoring each width by the same mean-variance objective (cost +
lam*variance) the theory itself optimizes.

Two of the three regimes turned out to be cost-dominated -- the objective
kept improving all the way to the edge of the tested grid, which is a
methodological trap, not a finding (a wider grid would just move the
"optimum" again; extending it 4x confirmed exactly that). Only the regime
where cost and variance are genuinely balanced produces a well-posed
interior optimum, and that's the one worth trusting: **the empirically
cost-risk-minimizing band is ~2x wider than the asymptotic theory predicts
-- a modest (+7%) but statistically real gap** (stationary block-bootstrap
CI on the objective difference excludes zero: `[0.008, 0.051]`,
n=50, n_effective≈22). Plausible mechanism, not yet independently
confirmed: WW's derivation assumes continuous monitoring, and this
backtest -- like most practical implementations -- only rebalances once a
day, which can't realize the fine-grained control the continuous-time
band assumes.

Run it yourself: `python examples/hedging_policy_frontier_study.py`
(needs `POLYGON_API_KEY`, base equities tier only).

**Follow-up control experiment** (`examples/gbm_control_experiment.py`, no
API key needed): the "continuous vs. daily monitoring" mechanism above was
plausible but untested -- it could equally have been fat tails or
volatility clustering, real-market features GBM doesn't have. Rerunning
the identical sweep (same objective, same bootstrap methodology) on
simulated GBM paths at the same daily-rebalancing cadence, with hedge_vol
set to the *true* simulation vol (removing vol-estimation noise as a
further confound), isolates discretization specifically. Result: pure GBM
with only daily monitoring **does** produce a real, statistically
significant band-widening (`c*=4x` theory, `+20%` objective gap, bootstrap
CI excludes zero) -- confirming discretization is a genuine, non-negligible
mechanism, not a negligible technicality. But it **overshoots** the
real-data result (`4x`/`+20%` vs. the real study's `2x`/`+7%`): pure
discretization alone predicts *more* widening than real data actually
shows, which rules out "discretization explains none of it" but also means
real-market structure isn't simply adding extra widening on top of a GBM
baseline -- something about real dynamics pulls the empirical optimum back
*toward* theory relative to what discretization alone predicts. Open
question, stated as such: genuine dynamical effect, or a scale mismatch
between the control's vol/cost grid and the real study's actual realized
vols.

## Layout

```
cpp/                          C++ core
  include/bscpp/
    types.hpp                  shared structs (MarketInputs, Greeks, MCResult)
    black_scholes.hpp           analytic BS pricer, Greeks, IV solver, batch variants
    monte_carlo.hpp              European MC pricer (antithetic + CRN Greeks)
    longstaff_schwartz.hpp        American MC pricer (LSM regression)
    heston.hpp                     Heston stochastic-vol semi-analytic + MC pricer
  src/                          implementations + pybind11 bindings

python/bscpp/                 Python package (imports the compiled extension)
  __init__.py                  Pythonic wrappers, trading_greeks() unit conversion
  strategies.py                 multi-leg strategies: straddle/strangle/vertical/
                                 strip/strap/butterfly, net Greeks, exact breakevens
  risk.py                        portfolio Greeks aggregation + configurable limit checks
  stats.py                       inference for dependent data: effective sample size,
                                  Newey-West HAC t-stats, stationary block bootstrap,
                                  dependent-correlation CIs
  backtest/
    data_provider.py            DataProvider interface: PolygonProvider, MockProvider
    engine.py                    StripPricer / Backtester -- chain pricing vs. market
    hedging.py                   HedgingBacktester -- delta-hedging P&L, transaction
                                  costs, vega-aware attribution, pluggable policies
    policies.py                  rebalancing policies: exact-delta, fixed band, and
                                  Whalley-Wilmott (1997) asymptotically-optimal band
    vol_surface.py                SVI fitting + two independent arbitrage checks
    heston_calibration.py          fits Heston params to a chain's implied vols,
                                    with regularization + a stability diagnostic

tests/                        pytest suite (76 tests). `pytest -m "not slow"`
                               runs the fast subset (~15s); full suite includes
                               heavy MC convergence and multi-start calibration checks
examples/                     runnable demos -- all but run_backtest.py (non-mock),
                               real_data_hedging_demo.py, and
                               real_data_validation_study.py need no API key
```

## What's implemented

**C++ pricing core**
- Closed-form Black-Scholes-Merton (calls/puts, continuous dividend yield)
  and full Greeks (delta, gamma, vega, theta, rho).
- Implied-vol solver: Newton-Raphson first, falling back to **Brent's
  method** (bracket-guaranteed convergent, never divides by vega) for
  deep-ITM/OTM and near-expiry cases where Newton's derivative-based step
  is unreliable. Not a port of Jaeckel's "Let's Be Rational" (2015, the
  solver `py_vollib` uses) -- slower per call, comparable robustness in the
  well-posed regime; see `black_scholes.hpp`.
- Batch pricing/IV variants (`bs_price_with_greeks_batch`,
  `bs_implied_vol_batch`) that loop in C++ rather than crossing the
  Python/C++ boundary once per contract.
- Monte Carlo European pricer under GBM: antithetic variates for variance
  reduction, Greeks via bump-and-reprice with common random numbers. The
  reported standard error is computed over antithetic **pair means** (the
  correct i.i.d. unit for a negatively-correlated antithetic sample; see
  CHANGELOG for the fix history), calibration-tested against realized
  estimator dispersion across independent seeds.
- American-style pricing via Longstaff-Schwartz (2001) least-squares Monte
  Carlo, using two independently-seeded path sets (calibration and
  pricing) -- matches QuantLib's `MCLongstaffSchwartzEngine`, avoiding the
  look-ahead bias of regressing and pricing on the same paths. Validated
  against the paper's own benchmark (S=36, K=40, r=6%, vol=20%, T=1y ->
  ~4.47-4.48) and the no-early-exercise identity for dividend-free calls.

**Multi-leg strategies** (`bscpp.strategies`)
- `straddle`, `strangle`, `vertical_spread`, `butterfly`, `strip` (long 1
  call + 2 puts, bearish-biased), `strap` (long 2 calls + 1 put,
  bullish-biased).
- `StrategyPricer.price()` returns net Greeks across all legs (accepts a
  per-strike vol dict for skew-aware pricing). `payoff_diagram()` computes
  the expiration P&L curve and exact breakevens via linear interpolation
  between each leg's strike (the payoff of a vanilla-option portfolio is
  piecewise-linear in spot, so this is exact, not a numerical root-find).

**Implied vol surface** (`bscpp.backtest.vol_surface`)
- `fit_svi_slice`: least-squares fit of Gatheral's SVI parameterization
  (`w(k) = a + b(rho(k-m) + sqrt((k-m)^2+sigma^2))`) to a chain's implied
  vols.
- Two independent no-arbitrage checks: `svi_butterfly_arbitrage_check`
  applies **Breeden-Litzenberger (1978)** numerically (prices calls off
  the fitted smile, takes a finite-difference second derivative in
  strike); `svi_gatheral_jacquier_check` implements the closed-form
  `g(k) >= 0` condition from Gatheral & Jacquier (2013) directly on the
  SVI parameters. Both enforce the `w(k) > 0` precondition and are
  confirmed to agree on both a well-behaved and a pathological test slice.
- Known gap: fitting slices independently gives no guarantee of
  consistency *across* expiries (no calendar-spread arbitrage guarantee);
  that's what Gatheral & Jacquier's SSVI extension provides and this
  project doesn't implement.

**Delta-hedging P&L + attribution** (`bscpp.backtest.hedging`)
- `HedgingBacktester`: sells an option and delta-hedges it against a real
  or simulated price path, correctly crediting dividend income on the
  stock leg when `dividend_yield > 0`.
- `transaction_cost_bps`: cost of crossing the bid-ask spread on every
  trade. Flows through `attribute_pnl` as an exact term (not a Taylor
  approximation), leaving `attribution_error` unchanged at any cost level.
- `attribute_pnl`: decomposes daily P&L into financing + gamma + theta
  terms from a second-order Taylor expansion of the option's own pricing
  function (derivation in the method's docstring). The decomposition's
  daily sum is an exact accounting identity against the simulation's own
  cumulative P&L, and on a deterministic path collapses to Carr & Madan's
  (2002) closed-form limit `0.5*Gamma*S^2*(hedge_vol^2 - realized_vol^2)*dt`
  to within ~0.5%.
- `hedge_vol` may be a time series (re-marked at each date); `attribute_pnl`
  then reports an explicit **vega** term, identically zero in the
  constant-vol configuration.
- **Rebalancing policies** (`bscpp.backtest.policies`): `DeltaPolicy`
  (baseline), `BandPolicy` (fixed no-trade band), and
  `WhalleyWilmottPolicy` (1997 asymptotically-optimal band; published
  Gamma^(2/3)/cost^(1/3)/risk-aversion^(-1/3) scalings verified numerically
  to <1e-9). Band policies cut total spread cost >30% vs. daily
  rebalancing in this suite's own simulated-path tests. `CallablePolicy`
  is a plug-in point for custom or learned (e.g. RL) policies.
- **`bscpp.stats`**: effective sample size, Newey-West HAC t-stats,
  Politis-Romano stationary block bootstrap, dependent-correlation CIs.
  No function returns a bare p-value -- every result carries its
  assumptions (HAC lags, block length, effective N) in its repr.

**Chain pricing vs. real market data** (`bscpp.backtest.engine`)
- `StripPricer`/`Backtester`: pull a chain slice from a `DataProvider`,
  solve implied vol per contract (batched), price with both BS and MC,
  and report theoretical price vs. observed market mid. Every row carries
  an `iv_source` column (`quoted`/`solved`/`fallback`); fallback rows are
  NaN rather than a silently invented vol.

**Portfolio risk aggregation** (`bscpp.risk`)
- `PortfolioRiskManager`: net Greeks across multi-underlying `Position`s,
  grouped by underlying, with configurable `RiskLimits` and breach
  flagging. This is net-Greeks aggregation with limits, not a production
  risk system -- no live feed, margin model, or kill switch.

**Heston stochastic volatility** (`bscpp.heston_price`, `bscpp.backtest.heston_calibration`)
- Semi-analytic pricer via the Heston (1993) characteristic function,
  using the Albrecher, Mayer, Schoutens & Tistaert (2007) **"Little
  Trap"** reformulation (avoids the branch-cut discontinuity in the
  original formula's complex logarithm). Confirmed term-by-term against
  QuantLib's `AnalyticHestonEngine`.
- An independent Monte Carlo pricer (full-truncation Euler for the CIR
  variance process) cross-checks the characteristic-function formula:
  collapses to Black-Scholes as vol-of-vol -> 0, agrees with MC across
  strikes/types/a Feller-violating stress case, exact put-call parity.
- The P1/P2 integrals use adaptive Simpson quadrature with adaptive
  upper-bound extension rather than a fixed-node table, self-terminating
  on measured error.
- `calibrate_heston` fits (kappa, theta, xi, rho, v0) in **IV space**
  (price-space residuals underweight the OTM wings, where the vol
  information actually lives), with Tikhonov-style regularization toward
  sane priors -- Heston has a known identifiability issue where `v0` can
  otherwise land on a degenerate corner of the loss surface.
  `calibrate_heston_with_stability` runs several perturbed initial
  guesses and reports whether fit quality *and* the parameters themselves
  are stable across starts.
- `heston_price_batch` prices a whole strike grid in one call by sharing
  characteristic-function evaluations across strikes (the CF doesn't
  depend on strike, only its phase factor does) over a fixed, not
  adaptive, quadrature grid -- profiling showed `heston_price` accounting
  for 96.8% of a calibration call's runtime, and `calibrate_heston` now
  uses this path internally. Fixed quadrature can't inherit the adaptive
  pricer's self-terminating accuracy for free, so resolution is chosen by
  measured error rather than assumed: a maturity **and** vol-of-vol sweep
  against the adaptive price found both short maturity and high xi
  independently degrade a fast, cheap grid, so `calibrate_heston` falls
  back to a validated-accurate higher-resolution grid outside that
  regime. Net effect on a 9-strike calibration: 254ms -> 82ms (3.1x),
  fitted parameters and fit RMSE unchanged. See
  `_batch_resolution_for_maturity` in `heston_calibration.py` for the
  measured thresholds and a disclosed residual limitation at
  near-degenerate correlation.
- `heston_satisfies_feller_condition`: checks `2*kappa*theta >= xi^2` as a
  diagnostic (many market-calibrated fits violate it in practice).

**Trading-desk Greeks** (`bscpp.trading_greeks`)
- Converts calculus Greeks (vega/rho per 1.00 of vol/rate, theta per year)
  to desk convention (vega/rho per 1%, theta per day).

**Out of scope**: local vol (Dupire), SSVI/other cross-expiry-consistent
surface models, stochastic vol beyond Heston (SABR, rough vol), a full
historical options-tick database, multi-leg strategy backtesting over
time, a real-time market data feed, order/execution management, margin
and capital modeling, and a kill switch. The last four are the difference
between "correct pricing math" and "safe to connect to a real trading
account" -- a different, much larger undertaking than this project.

## Setup

Requires a C++17 compiler (Xcode command line tools on macOS) and Python
3.9+. No CMake needed -- the extension builds via `pybind11.setup_helpers`.

```bash
cd black-scholes-mc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q -m "not slow"   # fast suite, ~15s
```

## Run the demos (no API key needed)

```bash
python examples/american_pricing_demo.py     # European vs. American (LSM) pricing
python examples/strategy_demo.py               # straddle/strangle/vertical/strip/strap/butterfly
python examples/portfolio_risk_demo.py          # net Greeks + limit breaches across a portfolio
python examples/hedging_pnl_experiment.py       # realized-vs-implied vol P&L sweep + transaction costs
python examples/vol_surface_fit_demo.py         # SVI fit + arbitrage check on a synthetic smile
python examples/heston_calibration_demo.py       # Heston calibration + stability diagnostic vs. SVI
python examples/run_backtest.py --mock --ticker SPY   # full chain-pricing pipeline
```

Sample output from `hedging_pnl_experiment.py` (selling an ATM call, hedging
daily at 30% vol, 60 days to expiry, 400 simulated paths per realized-vol
level, frictionless):

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
python examples/real_data_validation_study.py
```

`real_data_hedging_demo.py` pulls real historical daily closes, runs the
delta-hedging backtest against them, and prints the P&L attribution
(including transaction costs) -- works on Polygon/Massive's **base**
equities tier (the daily aggregates endpoint doesn't require an options
plan). Sample attribution output from a real run against AAPL:

```
financing_pnl          -1.82
gamma_pnl               -6.80
theta_pnl                6.99
transaction_cost_pnl    -0.39
predicted_pnl           -2.01
realized_pnl            -2.26
attribution_error       -0.25
```

`real_data_validation_study.py` is the multi-ticker, multi-window version:
real daily closes for 5 liquid names (AAPL, MSFT, GOOGL, AMZN, SPY) over
~9 months, rolled forward across 10 overlapping 45-day windows per ticker
using trailing realized vol as the `hedge_vol` forecast. Across 50
ticker-windows:

```
Hit rate (sign(hedge_vol - forward_realized_vol) == sign(hedging_pnl)): 80.0%
Correlation(vol_gap, hedging_pnl): r=0.877
```

Two caveats reported alongside the number, not left implicit: the windows
overlap within each ticker and the tickers share a market vol regime, so
the effective sample size is well below 50 (no i.i.d. p-value is
reported, for that reason); and the relationship being reproduced is close
to a mechanical identity (Carr & Madan 2002), so this is a
software-correctness check on real data, not evidence of a tradable edge.

`run_backtest.py`/`StripPricer` (chain snapshots) is the one demo that
needs an options-capable data plan -- see below.

## Real market data

The backtester targets Polygon.io's REST API (rebranded "Massive";
`PolygonProvider` still hits `api.polygon.io`). Options chain data
requires at least an "Options Starter" plan (`403 NOT_AUTHORIZED`
otherwise); daily underlying price history (`get_price_history`, used by
the hedging backtest) works on the base equities tier. Historical chain
*snapshots* for arbitrary past dates require a higher tier still -- see
the `Backtester` docstring in `python/bscpp/backtest/engine.py`. Paginated
endpoints follow `next_url` to completion with retry/backoff on
rate-limit/5xx responses.

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

# Heston stochastic vol (semi-analytic; HestonMCPricer for the independent MC cross-check)
hp = bscpp.HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.05)
heston_price = bscpp.heston_price(100, 100, 0.05, 0.0, 1.0, bscpp.OptionType.Call, hp)

# Multi-leg strategies
strat_pricer = bscpp.StrategyPricer(rate=0.05)
result = strat_pricer.price(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)
plot_df, breakevens = strat_pricer.payoff_diagram(bscpp.straddle(strike=100), spot=100, vol=0.2, maturity=1.0)

# Portfolio risk aggregation
positions = [
    bscpp.Position("AAPL call", "call", quantity=10, underlying="AAPL", spot=200, rate=0.05,
                    strike=210, vol=0.25, maturity=0.5),
    bscpp.Position("AAPL hedge", "stock", quantity=-500, underlying="AAPL", spot=200, rate=0.05),
]
risk_mgr = bscpp.PortfolioRiskManager(bscpp.RiskLimits(max_abs_vega=50))
net = risk_mgr.net_greeks(positions)
breaches = risk_mgr.check_limits(positions)
```

```python
from bscpp.backtest import (
    MockProvider, StripPricer, HedgingBacktester,
    fit_svi_slice, svi_butterfly_arbitrage_check, svi_gatheral_jacquier_check,
    calibrate_heston, calibrate_heston_with_stability,
)

provider = MockProvider(spot=450.0, base_vol=0.18)
pricer = StripPricer(provider, rate=0.05)
chain = pricer.price_strip("SPY", expiration, strike_range=(0.9, 1.1))

svi = fit_svi_slice(chain["strike"], chain["model_iv"], spot=450.0, t_years=chain["T"].iloc[0])
arb = svi_butterfly_arbitrage_check(svi, spot=450.0, rate=0.05)       # numerical (Breeden-Litzenberger)
arb2 = svi_gatheral_jacquier_check(svi)                                # closed-form g(k), cross-check

heston = calibrate_heston(chain["strike"], chain["type"], chain["model_iv"],
                           spot=450.0, t_years=chain["T"].iloc[0], rate=0.05)
stability = calibrate_heston_with_stability(chain["strike"], chain["type"], chain["model_iv"],
                                             spot=450.0, t_years=chain["T"].iloc[0], rate=0.05)

hedger = HedgingBacktester(rate=0.05, transaction_cost_bps=5.0)
result = hedger.run(price_history, strike=450, expiration=expiration, hedge_vol=0.18)
attributed = hedger.attribute_pnl(result)  # financing / gamma / theta / transaction cost breakdown
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 -- see `LICENSE`.
