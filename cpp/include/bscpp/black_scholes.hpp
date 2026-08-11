#pragma once

#include "bscpp/types.hpp"

namespace bscpp {

// Closed-form Black-Scholes-Merton pricing (supports a continuous dividend yield).
class BlackScholes {
public:
    static double price(const MarketInputs& in);
    static Greeks greeks(const MarketInputs& in);
    static PricingResult price_with_greeks(const MarketInputs& in);

    // Solve for implied volatility given an observed market price via
    // Newton-Raphson with a bisection fallback for robustness.
    // Returns NaN if it fails to converge within max_iter.
    static double implied_vol(const MarketInputs& in, double market_price,
                               double initial_guess = 0.2, int max_iter = 100,
                               double tol = 1e-8);

private:
    static double norm_cdf(double x);
    static double norm_pdf(double x);
    static double d1(const MarketInputs& in);
    static double d2(double d1_val, const MarketInputs& in);
};

}  // namespace bscpp
