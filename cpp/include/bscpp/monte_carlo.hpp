#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "bscpp/types.hpp"

namespace bscpp {

// Monte Carlo pricer for European options under geometric Brownian motion.
// Supports antithetic variates for variance reduction, and computes Greeks
// via bump-and-reprice with common random numbers (same underlying draws
// reused across bumped scenarios to cut variance in the finite differences).
class MonteCarloPricer {
public:
    explicit MonteCarloPricer(std::uint64_t seed = 42);

    MCResult price_european(const MarketInputs& in, long num_paths, bool antithetic = true);

    Greeks greeks_european(const MarketInputs& in, long num_paths, bool antithetic = true,
                            double bump_frac = 0.01);

private:
    std::vector<double> generate_normals(long n);
    static double payoff(double s_t, double strike, OptionType type);
    MCResult price_with_z(const MarketInputs& in, const std::vector<double>& z, bool antithetic);

    std::mt19937_64 rng_;
};

}  // namespace bscpp
