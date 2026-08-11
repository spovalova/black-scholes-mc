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
class AmericanPricer {
public:
    explicit AmericanPricer(std::uint64_t seed = 42);

    // num_steps is the number of exercise opportunities between 0 and T
    // (monitoring dates), poly_degree sets the regression basis
    // {1, x, x^2, ..., x^poly_degree} where x = S / strike.
    MCResult price(const MarketInputs& in, long num_paths, int num_steps, int poly_degree = 2);

private:
    static double payoff(double s, double strike, OptionType type);

    std::mt19937_64 rng_;
};

}  // namespace bscpp
