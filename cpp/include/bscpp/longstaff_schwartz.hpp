#pragma once

#include <cstdint>
#include <random>

#include "bscpp/types.hpp"

namespace bscpp {

// American-style option pricing via the Longstaff-Schwartz (2001) least-
// squares Monte Carlo algorithm: simulate full GBM paths, then walk
// backwards from maturity, at each exercise date regressing the discounted
// realized continuation value onto a polynomial basis in the (in-the-money)
// spot price to estimate E[continuation | S_t], and exercising whenever
// immediate exercise beats that estimate.
//
// Known limitation, confirmed against QuantLib's MCLongstaffSchwartzEngine:
// this regresses and prices along the SAME path set. QuantLib deliberately
// uses a separately-seeded calibration path set specifically to avoid this;
// the shared-path-set approach has a known small upward (look-ahead) bias,
// asymptotically proportional to the regressors-to-paths ratio. With the
// path counts this project defaults to (tens of thousands of paths against
// 3-4 regressors) the bias should be small -- consistent with the
// benchmark test passing -- but it is not eliminated the way QuantLib's is.
class AmericanPricer {
public:
    explicit AmericanPricer(std::uint64_t seed = 42);

    // num_steps is the number of exercise opportunities between 0 and T
    // (monitoring dates), poly_degree sets the regression basis
    // {1, x, x^2, ..., x^poly_degree} where x = S / strike. Monomial basis
    // (rather than the Laguerre polynomials the original paper used) is
    // fine at this low degree with normalized input -- QuantLib's own
    // default LSM engine uses the same combination (Monomial basis, degree
    // 2, S/strike normalization) for the same numerical-conditioning
    // reason. It only becomes a real problem at higher polynomial degree.
    MCResult price(const MarketInputs& in, long num_paths, int num_steps, int poly_degree = 2);

private:
    static double payoff(double s, double strike, OptionType type);

    std::mt19937_64 rng_;
};

}  // namespace bscpp
