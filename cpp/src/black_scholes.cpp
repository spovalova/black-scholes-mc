#include "bscpp/black_scholes.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace bscpp {

namespace {
constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1 / sqrt(2*pi)
}

double BlackScholes::norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double BlackScholes::norm_pdf(double x) {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

double BlackScholes::d1(const MarketInputs& in) {
    if (in.maturity <= 0.0 || in.vol <= 0.0) {
        throw std::invalid_argument("maturity and vol must be positive");
    }
    const double num = std::log(in.spot / in.strike) +
                        (in.rate - in.dividend_yield + 0.5 * in.vol * in.vol) * in.maturity;
    const double den = in.vol * std::sqrt(in.maturity);
    return num / den;
}

double BlackScholes::d2(double d1_val, const MarketInputs& in) {
    return d1_val - in.vol * std::sqrt(in.maturity);
}

double BlackScholes::price(const MarketInputs& in) {
    const double D1 = d1(in);
    const double D2 = d2(D1, in);
    const double disc_r = std::exp(-in.rate * in.maturity);
    const double disc_q = std::exp(-in.dividend_yield * in.maturity);

    if (in.type == OptionType::Call) {
        return in.spot * disc_q * norm_cdf(D1) - in.strike * disc_r * norm_cdf(D2);
    }
    return in.strike * disc_r * norm_cdf(-D2) - in.spot * disc_q * norm_cdf(-D1);
}

Greeks BlackScholes::greeks(const MarketInputs& in) {
    const double D1 = d1(in);
    const double D2 = d2(D1, in);
    const double disc_r = std::exp(-in.rate * in.maturity);
    const double disc_q = std::exp(-in.dividend_yield * in.maturity);
    const double sqrtT = std::sqrt(in.maturity);
    const double pdf_d1 = norm_pdf(D1);

    Greeks g;
    g.gamma = disc_q * pdf_d1 / (in.spot * in.vol * sqrtT);
    g.vega = in.spot * disc_q * pdf_d1 * sqrtT;  // per 1.00 (100%) vol change

    if (in.type == OptionType::Call) {
        g.delta = disc_q * norm_cdf(D1);
        g.theta = -disc_q * in.spot * pdf_d1 * in.vol / (2.0 * sqrtT) -
                   in.rate * in.strike * disc_r * norm_cdf(D2) +
                   in.dividend_yield * in.spot * disc_q * norm_cdf(D1);
        g.rho = in.strike * in.maturity * disc_r * norm_cdf(D2);  // per 1.00 (100%) rate change
    } else {
        g.delta = -disc_q * norm_cdf(-D1);
        g.theta = -disc_q * in.spot * pdf_d1 * in.vol / (2.0 * sqrtT) +
                   in.rate * in.strike * disc_r * norm_cdf(-D2) -
                   in.dividend_yield * in.spot * disc_q * norm_cdf(-D1);
        g.rho = -in.strike * in.maturity * disc_r * norm_cdf(-D2);
    }
    return g;
}

PricingResult BlackScholes::price_with_greeks(const MarketInputs& in) {
    PricingResult r;
    r.price = price(in);
    r.greeks = greeks(in);
    return r;
}

double BlackScholes::implied_vol(const MarketInputs& in, double market_price,
                                  double initial_guess, int max_iter, double tol) {
    MarketInputs work = in;
    work.vol = initial_guess;

    // Newton-Raphson using vega as the derivative.
    for (int i = 0; i < max_iter; ++i) {
        double model_price;
        double vega;
        try {
            model_price = price(work);
            vega = greeks(work).vega;
        } catch (const std::invalid_argument&) {
            break;  // vol collapsed to <= 0; fall through to bisection
        }
        const double diff = model_price - market_price;
        if (std::abs(diff) < tol) {
            return work.vol;
        }
        if (vega < 1e-10) {
            break;  // vega too flat, Newton step is unreliable
        }
        double next_vol = work.vol - diff / vega;
        if (next_vol <= 0.0 || !std::isfinite(next_vol)) {
            break;
        }
        work.vol = next_vol;
    }

    // Bisection fallback on [lo, hi].
    double lo = 1e-6, hi = 5.0;
    MarketInputs lo_in = in, hi_in = in;
    lo_in.vol = lo;
    hi_in.vol = hi;
    double f_lo = price(lo_in) - market_price;
    double f_hi = price(hi_in) - market_price;
    if (f_lo * f_hi > 0.0) {
        return std::numeric_limits<double>::quiet_NaN();  // no sign change; can't bracket a root
    }
    for (int i = 0; i < max_iter; ++i) {
        double mid = 0.5 * (lo + hi);
        MarketInputs mid_in = in;
        mid_in.vol = mid;
        double f_mid = price(mid_in) - market_price;
        if (std::abs(f_mid) < tol) {
            return mid;
        }
        if (f_lo * f_mid < 0.0) {
            hi = mid;
            f_hi = f_mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // namespace bscpp
