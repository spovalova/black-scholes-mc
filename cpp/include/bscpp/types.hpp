#pragma once

namespace bscpp {

enum class OptionType { Call, Put };

struct Greeks {
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;   // per 1.00 (100%) change in vol
    double theta = 0.0;  // per year
    double rho = 0.0;    // per 1.00 (100%) change in rate
};

struct PricingResult {
    double price = 0.0;
    Greeks greeks;
};

struct MCResult {
    double price = 0.0;
    double std_error = 0.0;
};

// Inputs shared by both pricers.
struct MarketInputs {
    double spot;    // S: underlying price
    double strike;  // K
    double rate;    // r: risk-free rate (annualized, continuous compounding)
    double dividend_yield;  // q: continuous dividend yield
    double vol;     // sigma: annualized volatility
    double maturity;  // T: time to expiry in years
    OptionType type;
};

}  // namespace bscpp
