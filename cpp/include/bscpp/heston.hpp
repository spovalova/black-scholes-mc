#pragma once

#include <complex>
#include <cstdint>
#include <random>

#include "bscpp/types.hpp"

namespace bscpp {

// Heston (1993) stochastic volatility model:
//   dS_t/S_t = (r-q) dt + sqrt(v_t) dW1_t
//   dv_t     = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW2_t
//   Corr(dW1_t, dW2_t) = rho dt
struct HestonParams {
    double kappa;  // mean-reversion speed of variance
    double theta;  // long-run variance
    double xi;     // vol-of-vol
    double rho;    // correlation between spot and variance Brownian motions
    double v0;     // initial (instantaneous) variance
};

// Semi-analytic pricer via the Heston characteristic function.
class HestonPricer {
public:
    // Uses the Albrecher, Mayer, Schoutens & Tistaert (2007) "Little Trap"
    // reformulation of the original Heston (1993) characteristic function,
    // which avoids the branch-cut discontinuities the naive formula hits
    // for long maturities / certain parameter regions.
    static double price(double spot, double strike, double rate, double dividend_yield,
                         double maturity, OptionType type, const HestonParams& hp);

    // Necessary (not sufficient in the fitted-to-market sense) condition for
    // the variance process to almost-surely stay strictly positive. Reported
    // as a diagnostic -- many market-calibrated parameter sets violate it in
    // practice, and the pricer still works (variance just clamped at 0) but
    // the model interpretation as "always-positive variance" no longer holds.
    static bool satisfies_feller_condition(const HestonParams& hp);

private:
    static std::complex<double> char_function(std::complex<double> phi, double spot, double rate,
                                                double dividend_yield, double maturity,
                                                const HestonParams& hp, int j);
    static double probability(double spot, double strike, double rate, double dividend_yield,
                               double maturity, const HestonParams& hp, int j);
};

// Independent cross-check on HestonPricer: simulates the SDE directly
// instead of relying on the (bug-prone) closed-form characteristic function.
class HestonMCPricer {
public:
    explicit HestonMCPricer(std::uint64_t seed = 42);

    // Full-truncation Euler scheme (Lord, Koekkoek & van Dijk, 2010) for the
    // CIR variance process: v is floored at 0 wherever it appears in the
    // drift/diffusion, which keeps S well-defined without needing a more
    // elaborate (e.g. QE) scheme.
    MCResult price(double spot, double strike, double rate, double dividend_yield,
                   double maturity, OptionType type, const HestonParams& hp, long num_paths,
                   int num_steps);

private:
    std::mt19937_64 rng_;
};

}  // namespace bscpp
