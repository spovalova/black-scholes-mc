# bscpp

A derivatives pricing and quantitative research toolkit: a C++ pricing
core (analytic Black-Scholes, Monte Carlo, Longstaff-Schwartz American,
Heston stochastic vol) exposed to Python via pybind11, plus a Python layer
that prices live option chains against real market data, constructs and
analyzes multi-leg strategies, fits arbitrage-checked implied vol surfaces,
simulates delta-hedging P&L (with transaction costs and a full Greeks
attribution) against both simulated and real historical data, and
aggregates portfolio-level risk with configurable limits. Every non-trivial
formula and algorithm was checked against the published literature and/or
production implementations (mostly QuantLib's source) before being trusted
-- see "External verification" below for what that surfaced, fixed, and
still leaves open.

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
  stats.py                       honest inference for dependent data: effective sample
                                  size, Newey-West HAC t-stats, stationary block
                                  bootstrap, dependent-correlation CIs
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

tests/                        pytest suite (58 tests): closed-form benchmarks,
                               convergence behavior, arbitrage-free checks,
                               error-bar calibration, regression tests for
                               every externally-caught bug, stress tests,
                               backtest plumbing ("-m 'not slow'" runs in ~15s)
examples/                     runnable demos -- all but run_backtest.py (non-mock),
                               real_data_hedging_demo.py, and
                               real_data_validation_study.py need no API key
```

## What's implemented, and why it isn't a toy

**C++ pricing core**
- Closed-form Black-Scholes-Merton (calls/puts, continuous dividend yield)
  and full Greeks (delta, gamma, vega, theta, rho).
- **Implied-vol solver**: Newton-Raphson first (fast when vega isn't
  tiny), falling back to **Brent's method** (bracket-guaranteed
  convergent, never divides by vega) rather than the crude bisection this
  used to fall back to. Stress-tested across 20,000 random (strike, rate,
  maturity, vol) combinations spanning extreme moneyness and maturities
  from 1 day to 3 years: **zero NaN failures**, and machine-precision
  recovery (<1e-5 error) in the ~89% of cases where the inverse problem is
  well-posed (vega meaningfully above zero). The remaining ~11% (deep ITM
  + near-expiry) aren't solver failures -- price is numerically flat in
  vol there for *any* solver, including Jaeckel's "Let's Be Rational"
  (2015, the industry-standard solver used by `py_vollib`, which this
  still isn't a byte-for-byte port of -- see `black_scholes.hpp` for the
  full writeup and remaining speed gap).
- **Batch pricing/IV variants** (`bs_price_with_greeks_batch`,
  `bs_implied_vol_batch`) that loop in C++ rather than crossing the
  Python/C++ boundary once per contract -- `StripPricer` prices a whole
  chain slice in one call instead of one per row.
- Monte Carlo European pricer under GBM: antithetic variates for variance
  reduction, Greeks via bump-and-reprice with **common random numbers**
  (same underlying draws reused across bumped scenarios -- what keeps
  finite-difference Greeks from being unusably noisy). Verified: MC
  standard error shrinks ~4x when path count goes up 16x, matching the
  theoretical O(1/sqrt(N)) Monte Carlo convergence rate. The reported
  standard error is computed over **antithetic pair means** (an earlier
  version pooled the negatively-correlated pair members into the i.i.d.
  formula, overstating std_error by a measured ~32% -- caught in external
  review; the estimator is now regression-tested against the realized
  dispersion of the estimator across independent seeds).
- **American-style pricing via Longstaff-Schwartz (2001) least-squares
  Monte Carlo**: simulates full paths, walks backward from maturity
  regressing realized continuation value onto a polynomial basis to decide
  optimal early exercise, using **two independently-seeded path sets**
  (calibration and pricing) -- matching QuantLib's `MCLongstaffSchwartzEngine`
  and specifically avoiding the small upward look-ahead bias of regressing
  and pricing on the same paths. Validated against the paper's own
  benchmark (S=36, K=40, r=6%, vol=20%, T=1y -> independent implementations
  converge to ~4.47-4.48) and against the no-arbitrage identity that an
  American call with no dividends prices identically to its European
  counterpart. The ITM-only regression filter and monomial-basis-with-
  S/K-normalization choice were both confirmed to match QuantLib's own
  production defaults.

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
- **Two independent no-arbitrage checks, cross-verified against each
  other**: `svi_butterfly_arbitrage_check` prices calls off the fitted
  smile with the (already-tested) BS pricer across a strike grid and takes
  a finite-difference second derivative -- directly applying
  **Breeden-Litzenberger (1978)**: the risk-neutral density
  `q(K) = e^{rT} d^2C/dK^2` must be non-negative everywhere. Separately,
  `svi_gatheral_jacquier_check` implements the closed-form `g(k) >= 0`
  condition from Gatheral & Jacquier (2013) directly on the SVI
  parameters -- the same underlying condition, reparametrized into
  (log-moneyness, total-variance) coordinates so it's ~2 orders of
  magnitude cheaper and has no finite-difference noise floor. Both were
  confirmed to agree on a well-behaved slice (passes) and a deliberately
  pathological one (extreme rho, tiny sigma -- both correctly flag
  arbitrage). Both checks also enforce the **w(k) > 0 precondition**: the
  g(k) criterion is only meaningful where total variance is positive, and
  an earlier version passed a negative-total-variance slice (an outright
  arbitrage) as "arbitrage-free" -- caught in external review, now
  rejected explicitly with `reason="negative_total_variance"` and
  regression-tested. The closed-form check exists specifically because the
  numerical one's fixed density tolerance is scale-dependent: on a 1-day
  synthetic maturity it registered a technically-negative density of
  ~1e-13 (finite-difference noise, saved only by the tolerance), while
  `svi_gatheral_jacquier_check` gives an unambiguous answer on the same
  slice.
- Known gap: fitting slices independently (one `SVISlice` per expiry) has
  no guarantee they're jointly consistent across expiries -- no
  calendar-spread arbitrage guarantee, which is exactly what Gatheral &
  Jacquier's SSVI surface extension of this same paper exists to provide
  and this project does not implement.

**Delta-hedging P&L + Greeks attribution** (`bscpp.backtest.hedging`)
- `HedgingBacktester`: simulates selling an option and delta-hedging it
  daily at a given `hedge_vol` against a real or simulated price path.
  Correctly credits dividend income on the stock hedge leg when
  `dividend_yield > 0` -- an earlier version didn't, which was silently
  inconsistent with using dividend-adjusted Black-Scholes deltas (caught
  during external verification, since every existing test/demo happened
  to use `dividend_yield=0` and never exercised the bug).
- **`transaction_cost_bps`**: charges the cost of crossing the bid-ask
  spread on every trade (the initial hedge, each daily rebalance, and the
  final liquidation) -- the single biggest realism gap in calling this a
  "backtester" versus a frictionless pricing demo. Confirmed to strictly
  reduce P&L and scale (near-)linearly with bps; confirmed to bite harder
  at higher realized vol (bigger daily moves -> bigger delta changes ->
  more rebalance notional traded), roughly 30-45% more drag at
  realized_vol=0.45 than at 0.15 in `hedging_pnl_experiment.py`'s own
  sweep. Unlike gamma/theta (Taylor approximations), the cost flows
  through the same cash account **exactly**, so it's added to
  `attribute_pnl`'s `predicted_pnl` as an exact term -- confirmed to leave
  `attribution_error` completely unchanged (to float precision) at every
  cost level tested, which is what an exact (non-approximated) term should do.
- `realized_vs_implied_experiment`: sweeps realized vol against a fixed
  hedge_vol and shows the textbook result -- hedging at a vol *above* what
  actually realizes is profitable for the seller, *below* it is not, P&L
  crosses zero right at `realized_vol == hedge_vol` (see worked example
  below).
- `attribute_pnl`: decomposes each day's hedging P&L into financing +
  gamma + theta terms, derived from a second-order Taylor expansion of the
  option's own pricing function (full derivation in the method's
  docstring). This recovers the classic "theta pays you, gamma costs you"
  identity for a short, hedged option position, and is verified three ways:
  the decomposition's daily sum exactly equals the simulation's own
  cumulative P&L (an accounting identity, checked to float precision); the
  financing+gamma+theta prediction explains the large majority of realized
  P&L with a bounded higher-order residual; and on a deterministic path
  with an exact known realized vol, `financing+gamma+theta` matches
  **Carr & Madan's (2002)** canonical closed-form limit
  `0.5*Gamma*S^2*(hedge_vol^2 - realized_vol^2)*dt` to within ~0.5%.
- **Re-marked vols and vega P&L** (closes what used to be a documented
  gap): `hedge_vol` can be a time series, in which case the option is
  re-marked at each date's vol and `attribute_pnl` reports the re-marking
  P&L as an explicit **vega term** -- verified to leave the accounting
  identity intact, to explain the realized P&L where excluding it blows
  the residual up >2x, and to be identically zero in the constant-vol
  configuration (exact backward compatibility).
- **Rebalancing policies** (`bscpp.backtest.policies`): the hedge *timing*
  ladder the transaction-cost literature settled and almost nothing open
  implements -- rebalance-to-delta baseline, fixed no-trade band (trade to
  the nearest edge on breach), and the **Whalley-Wilmott (1997)**
  asymptotically-optimal band with its published Gamma^(2/3) and
  cost^(1/3) scalings (verified numerically to <1e-9 relative error).
  Band policies cut total spread cost >30% vs. daily rebalancing on
  simulated paths in this suite's own tests; a `delta_gap_pnl` attribution
  term accounts exactly for the first-order P&L of sitting away from the
  model delta inside the band. `CallablePolicy` lets custom or learned
  (e.g. RL) policies plug into the same interface for apples-to-apples
  comparison against the classical ladder.
- **Honest statistics** (`bscpp.stats`): effective sample size, Newey-West
  HAC t-stats, the Politis-Romano stationary block bootstrap, and
  dependent-correlation CIs. The real-data validation study now reports a
  block-bootstrap CI and its effective sample size instead of an i.i.d.
  p-value (which is meaningless for overlapping windows on co-moving
  tickers, and is therefore no longer printed anywhere). Design rule: no
  function in `bscpp.stats` returns a bare p-value -- every result object
  carries its assumptions (HAC lags, block length, effective N) in its
  repr, so the number can't be quoted without its caveats.

**Chain pricing vs. real market data** (`bscpp.backtest.engine`)
- `StripPricer`/`Backtester`: pull a chain slice from a `DataProvider`,
  solve implied vol per contract (batched), price with both BS and MC, and
  report theoretical price vs. observed market mid plus full Greeks.

**Portfolio risk aggregation** (`bscpp.risk`)
- `PortfolioRiskManager`: net Greeks across a list of `Position`s (options
  and/or stock, spanning multiple underlyings), grouped by underlying, with
  configurable `RiskLimits` and breach flagging. Stated plainly in the
  module docstring and here: this is net-Greeks aggregation with limits, not
  a production risk system -- no live feed, no margin/capital model, no
  kill switch, no scenario engine. That gap is a category difference (a
  personal research project vs. a trading desk's infrastructure), not a
  code-quality problem this project can or should pretend to close.

**Heston stochastic volatility** (`bscpp.heston_price`, `bscpp.backtest.heston_calibration`)
- Semi-analytic pricer via the Heston (1993) characteristic function, using
  the Albrecher, Mayer, Schoutens & Tistaert (2007) **"Little Trap"**
  reformulation -- the original formula has a branch-cut discontinuity in
  its complex logarithm that silently produces wrong prices for some
  parameter/maturity combinations, which is exactly the kind of bug that's
  invisible unless you specifically check for it.
- An independent **Monte Carlo pricer** (full-truncation Euler scheme for
  the CIR variance process) exists specifically to cross-check the
  characteristic-function formula against a from-scratch simulation, not
  just to be a second pricing option. This project does not trust a
  memorized complex-analysis formula on its own: verified three ways --
  (1) collapses cleanly to Black-Scholes as vol-of-vol -> 0 (monotonic
  convergence, the sharpest test of a sign/branch-cut error), (2) agrees
  with the independent MC engine within a few standard errors across
  strikes, option types, and a Feller-condition-violating stress case,
  (3) exact put-call parity. Also confirmed accurate at 1-day maturity and
  at vol-of-vol so extreme it badly violates the Feller condition (xi=3.0
  vs. 2*kappa*theta=0.16) -- a naive MC comparison there is *misleading*
  (the MC scheme's own known discretization bias is ~40 std errors off at
  300 steps, converging to the analytic price only by ~3000 steps; see
  `heston.hpp` for the full writeup).
- The P1/P2 integrals are evaluated via **adaptive Simpson quadrature**
  with adaptive upper-bound extension, not a fixed-node table -- swapped
  in specifically to avoid the risk of a transcribed Gauss-Laguerre
  magic-number error, and because it self-terminates on measured error
  rather than an unverified fixed truncation (the previous fixed
  phi_max=200/4000-point rule was explicitly untested at extreme
  parameters; the adaptive version is verified there instead).
- `calibrate_heston`: fits (kappa, theta, xi, rho, v0) to a chain's implied
  vols via `scipy.optimize.least_squares`, calibrating in **IV space**
  rather than raw price space (price-space residuals are dominated by deep
  ITM contracts whose price is nearly pure intrinsic value and barely
  moves with the vol parameters -- the OTM wings, where the actual vol
  information lives, would be underweighted). Validated by recovering a
  known synthetic Heston smile to <1e-3 RMSE.
- Heston has a well-known parameter identifiability issue -- different
  parameter vectors can produce near-identical smiles, especially for
  short-dated, mildly-curved data, where `v0` is poorly pinned down
  independent of `theta`/`kappa`. Rather than just documenting this,
  `calibrate_heston` applies **Tikhonov-style regularization** (pulling
  v0/theta toward the ATM variance prior, kappa toward a market-typical
  scale) that measurably fixes it on the exact case that used to break: v0
  moves from a degenerate 0.0006 to a sane 0.032 (right at the ATM
  variance level) at essentially unchanged fit quality. `calibrate_heston_with_stability`
  runs several randomly-perturbed initial guesses and reports whether fit
  quality *and* the parameters themselves are stable across starts --
  confirmed to correctly distinguish the regularized case (perfectly
  stable, param std=0 across 6 starts) from the unregularized one
  (kappa std=5.8!, despite nearly identical RMSE) -- the diagnostic a desk
  re-hedging off tomorrow's recalibration actually needs, not just a
  single point estimate.
- `heston_satisfies_feller_condition`: checks `2*kappa*theta >= xi^2`
  (keeps the variance process a.s. positive) as a diagnostic -- many
  market-calibrated fits violate it in practice, which is itself a useful
  thing to know about a calibration, not a hard failure.

**Trading-desk Greeks** (`bscpp.trading_greeks`)
- The raw Greeks are calculus derivatives: vega/rho per 1.00 (100 points)
  of vol/rate, theta per *year*. No desk quotes them that way. This is a
  one-line unit conversion (vega, rho /100; theta /365) that exists because
  mixing the two conventions is a classic, costly mistake for anyone new
  to reading options risk.

**Deliberately out of scope** (documented, not silently missing): local vol
(Dupire), SSVI/other cross-expiry-consistent surface models, other
stochastic vol models beyond Heston (SABR, rough vol), a full historical
options-tick database integration, and a full backtest P&L simulation of
*multi-leg* strategies over time (the hedging backtest covers single-leg
positions; extending `HedgingBacktester` to net multi-leg Greeks is a
natural next step on top of `strategies.py`). Also out of scope, and worth
being direct about rather than implying otherwise: a real-time market data
feed, an order/execution management system, margin and capital modeling,
and a kill switch -- the things that separate "correct pricing math" from
"safe to connect to a real trading account." Nothing in this project should
be read as claiming to have closed that gap; it's a different, much larger
undertaking than a personal pricing/research project.

## External verification

Every formula and algorithm above was checked against the published
literature and/or a production implementation (mainly QuantLib's C++
source) rather than trusted on the strength of passing this project's own
tests alone -- self-consistent tests can pass even when a formula encodes
the same subtle error twice. Highlights:

- The Heston characteristic function was confirmed **term-by-term**
  against QuantLib's `AnalyticHestonEngine` -- every intermediate quantity
  (`d`, the little-trap substitution, the C/D coefficients) matches
  exactly. Separately reassuring: QuantLib, Attari (2004), and
  Andersen-Piterbarg (2010) each independently discovered and fixed the
  *same* branch-cut instability in the original 1993 formula -- a known,
  well-studied hard problem, not something this project got lucky avoiding.
- LSM's in-the-money-only regression filter and monomial-basis-with-
  normalization choice both match QuantLib's actual `MCAmericanEngine`
  defaults.
- The SVI numerical (Breeden-Litzenberger) and closed-form (Gatheral-
  Jacquier g(k)) arbitrage checks were confirmed to agree with each other
  on both a well-behaved and a pathological test slice.
- The hedging P&L attribution's derivation was independently re-derived
  from the Black-Scholes PDE and shown to algebraically collapse to
  Carr & Madan's (2002) canonical realized-vs-implied-variance result --
  matched numerically to within ~0.5% on a deterministic test path.
- One real bug was found this way and fixed: `HedgingBacktester` wasn't
  crediting dividend income on the stock hedge leg, silently inconsistent
  with using dividend-adjusted deltas. Every existing test/demo used
  `dividend_yield=0` and never exercised it.
- A second, adversarial external review pass (v0.2.0) found **four more
  real bugs** -- notably all in the *statistics and preconditions around*
  otherwise-correct formulas, not in the formulas themselves: the
  antithetic MC standard error pooled negatively-correlated pair members
  as i.i.d. (~32% overstated, measured); the Gatheral-Jacquier check
  lacked its w(k)>0 precondition (passed an outright-arbitrage slice);
  the Heston pricer could return slightly negative deep-OTM prices; and
  the Polygon provider silently truncated paginated chains. All four are
  fixed with regression tests (see CHANGELOG.md). The lesson is now a
  house rule: verify the estimators and preconditions, not just the
  formulas.
- Concrete gaps this surfaced (IV solver, LSM's shared path set, Heston's
  quadrature scheme, and hedging's lack of transaction costs and a risk
  aggregation layer) were then **fixed**, not just documented -- see the
  bullets above for each; every fix was independently verified before being
  called done (a 20,000-case IV solver stress test, Heston re-verified at
  Feller-violating and 1-day-maturity extremes, transaction costs confirmed
  to leave `attribution_error` byte-for-byte unchanged, and the multi-ticker
  real-data study below in place of a single-anecdote demo). What's still
  explicitly out of scope (SSVI, vega P&L in the hedging attribution, and
  everything under "production risk system") remains genuinely out of
  scope for the reasons stated -- not a fix that was skipped, but ground
  this project isn't claiming to cover.

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
(including transaction costs) -- confirmed working on Polygon/Massive's
**base** equities tier (their daily aggregates endpoint doesn't require an
options plan). Sample attribution output from a real run against AAPL:

```
financing_pnl          -1.82
gamma_pnl               -6.80
theta_pnl                6.99
transaction_cost_pnl    -0.39
predicted_pnl           -2.01
realized_pnl            -2.26
attribution_error       -0.25
```

`real_data_validation_study.py` is the multi-ticker, multi-window
statistical version of the same idea: real daily closes for 5 liquid names
(AAPL, MSFT, GOOGL, AMZN, SPY) over ~9 months, rolled forward across 10
overlapping 45-day windows per ticker using trailing realized vol as the
hedge_vol forecast (no invented implied vol -- this only needs the base
equities tier). Across 50 real out-of-sample ticker-windows:

```
Hit rate (sign(hedge_vol - forward_realized_vol) == sign(hedging_pnl)): 80.0%
Correlation(vol_gap, hedging_pnl): r=0.877
```

That's the realized-vs-implied vol P&L relationship reproducing on real
data across multiple names and windows -- not a single cherry-pickable
window. **Two honesty caveats, stated rather than implied away:** (1) the
50 ticker-windows are NOT independent observations -- windows overlap
within each ticker, and the five tickers share the same market vol regime
(SPY literally contains the other four), so the effective sample size is
far smaller than 50 and an i.i.d. p-value would be meaningless (none is
reported for exactly that reason); (2) the relationship being reproduced
is close to a mechanical identity (Carr & Madan 2002), so this is a
software-correctness check on real data, not evidence of a tradable edge.

`run_backtest.py`/`StripPricer` (chain snapshots) is the one demo that does
need an options-capable plan -- see below.

## Real market data

The backtester targets Polygon.io's REST API (the company has since
rebranded to "Massive", but the API itself is unchanged -- `PolygonProvider`
still hits `api.polygon.io` and that's confirmed working). The provider
follows `next_url` pagination to completion (an earlier version fetched a
single 250-row page and silently truncated large chains -- caught in
external review, regression-tested against a mocked multi-page session)
and retries with exponential backoff on rate-limit/5xx responses. Options chain
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

## License

Apache-2.0 -- see `LICENSE`.
