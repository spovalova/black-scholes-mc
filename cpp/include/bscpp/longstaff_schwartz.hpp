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
// Uses TWO independently-seeded path sets, matching QuantLib's
// MCLongstaffSchwartzEngine: a calibration set (fits the regression
// coefficients per exercise date, backward pass) and a separate pricing
// set (applies those fixed coefficients forward, deciding exercise as it
// goes). Regressing and pricing on the *same* path set is a known source
// of small upward look-ahead bias; QuantLib avoids it with a
// separately-seeded calibration set (seed + 1768237423, literally copied
// from QuantLib's own offset choice, of no significance beyond "a large
// arbitrary constant") and this class does the same.
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
    // num_calibration_paths defaults to num_paths if left at 0.
    MCResult price(const MarketInputs& in, long num_paths, int num_steps, int poly_degree = 2,
                    long num_calibration_paths = 0);

    // Public: shared by the free-function calibration helper in the .cpp
    // (which operates on a plain path matrix, not an AmericanPricer
    // instance) as well as by price() itself.
    static double payoff(double s, double strike, OptionType type);

private:
    std::mt19937_64 rng_;              // pricing path set
    std::mt19937_64 rng_calibration_;  // independently-seeded calibration path set
};

}  // namespace bscpp
