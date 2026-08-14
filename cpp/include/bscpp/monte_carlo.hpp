#pragma once

#include <cstdint>
#include <vector>

#include "bscpp/philox.hpp"
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

    // generate_normals draws are index-addressed (see the .cpp), not
    // sequentially consumed from one shared, stateful generator -- that's
    // what makes the loop safe to parallelize (#pragma omp parallel for)
    // with output that's bit-identical regardless of thread count or
    // scheduling: block_cursor_ hands out a disjoint counter range per
    // call, and output index i within that range depends only on i.
    std::uint64_t seed_;
    std::uint64_t block_cursor_ = 0;
};

}  // namespace bscpp
