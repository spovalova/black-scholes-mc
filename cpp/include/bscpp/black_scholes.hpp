#pragma once

#include <vector>

#include "bscpp/types.hpp"

namespace bscpp {

// Closed-form Black-Scholes-Merton pricing (supports a continuous dividend yield).
class BlackScholes {
public:
    static double price(const MarketInputs& in);
    static Greeks greeks(const MarketInputs& in);
    static PricingResult price_with_greeks(const MarketInputs& in);

    // Batch variant of price_with_greeks: prices an arbitrary list of
    // contracts (independent spot/strike/vol/etc per entry) in a single
    // call. This exists because pricing a whole option chain one contract
    // at a time from Python pays a Python<->C++ call-boundary cost on
    // every contract; looping in C++ instead amortizes that to one crossing
    // for the whole chain.
    static std::vector<PricingResult> price_with_greeks_batch(const std::vector<MarketInputs>& inputs);

    // Solve for implied volatility given an observed market price via
    // Newton-Raphson with a bisection fallback for robustness.
    // Returns NaN if it fails to converge within max_iter.
    //
    // Known limitation: this is materially weaker than the industry-
    // standard solver, Peter Jaeckel's "Let's Be Rational" (2015, used by
    // py_vollib) -- four rational-function initial guesses selected by
    // moneyness region plus 4th-order Householder iteration, engineered to
    // converge to full float64 precision in ~2 iterations for EVERY input,
    // including deep ITM/OTM. This is not a hypothetical gap: it is the
    // exact reason StripPricer's IV solve can fail (return NaN, triggering
    // the 0.20 fallback documented in engine.py) on deep-ITM/OTM synthetic
    // contracts whose quoted price sits close to the intrinsic-value floor
    // implied by vol->0 -- precisely the boundary regime Jaeckel's method
    // exists to handle robustly and this Newton+bisection solver does not.
    static double implied_vol(const MarketInputs& in, double market_price,
                               double initial_guess = 0.2, int max_iter = 100,
                               double tol = 1e-8);

    // Batch implied-vol solve, one market_price per input.
    static std::vector<double> implied_vol_batch(const std::vector<MarketInputs>& inputs,
                                                  const std::vector<double>& market_prices,
                                                  double initial_guess = 0.2, int max_iter = 100,
                                                  double tol = 1e-8);

private:
    static double norm_cdf(double x);
    static double norm_pdf(double x);
    static double d1(const MarketInputs& in);
    static double d2(double d1_val, const MarketInputs& in);
};

}  // namespace bscpp
