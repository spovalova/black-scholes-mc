#pragma once

#include "bscpp/types.hpp"

namespace bscpp {

// Cox-Ross-Rubinstein (1979) binomial tree, dividend-aware (continuous
// yield q) and early-exercise-aware -- the standard, deterministic
// American-option pricer, not a Monte Carlo one. Exists specifically for
// solving American implied vol in the chain pipeline (see
// bscpp.backtest.engine.StripPricer): a real equity-option chain is
// American-style, and solving a EUROPEAN Black-Scholes IV against an
// American market price silently mixes in an early-exercise premium the
// European formula has no way to represent -- worst for ITM puts (or ITM
// calls on a dividend payer), where that premium is largest. CRR removes
// that mismatch by pricing the same exercise style the market actually
// trades.
//
// Deliberately NOT the project's American Monte Carlo path (see
// longstaff_schwartz.hpp): LSM exists for path-dependent/multi-factor
// generalizations Monte Carlo is suited for and for cross-checking this
// tree independently; it is not itself the production American pricer
// here, and using it as one would be needless -- CRR is deterministic (no
// simulation noise to average away) and runs in microseconds at the node
// counts below, where LSM needs tens of thousands of paths for comparable
// precision. Two independent methods, two different jobs.
class CRRPricer {
public:
    // American-style price via a CRR binomial tree. num_steps=200 is a
    // reasonable default (see test_crr_tree.py for the convergence sweep
    // this was picked from); very large vol*sqrt(dt) can push the
    // risk-neutral probability p outside [0,1] at low step counts for a
    // given (rate, dividend_yield, vol, maturity) combination -- p is
    // clamped into [0,1] rather than left to silently misprice, and
    // that's a signal num_steps is too low for the requested precision at
    // that (vol, maturity) combination, not that the price is untrustworthy
    // by construction.
    static double price(double spot, double strike, double rate, double dividend_yield,
                         double maturity, OptionType type, double vol, int num_steps = 200);

    // American implied vol via Brent's method (see brent.hpp): CRR prices
    // monotonically in vol (like the European formula), so the same
    // bracket-guaranteed solver applies directly -- no analytic vega
    // needed, and no risk of Newton dividing by a near-zero one. Returns
    // NaN if [1e-6, 5.0] doesn't bracket market_price (a genuinely
    // arbitrage-violating or otherwise bad quote), matching
    // BlackScholes::implied_vol's contract exactly so StripPricer can
    // swap one for the other without changing its NaN-handling.
    static double implied_vol(double spot, double strike, double rate, double dividend_yield,
                               double maturity, OptionType type, double market_price,
                               int num_steps = 200, double tol = 1e-6, int max_iter = 100);
};

}  // namespace bscpp
