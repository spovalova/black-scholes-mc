#include "bscpp/heston.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>

namespace bscpp {

namespace {

using cdouble = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

// Composite Simpson's rule, n subintervals (must be even).
double simpson_integrate(const std::function<double(double)>& f, double a, double b, int n) {
    if (n % 2 != 0) ++n;
    const double h = (b - a) / n;
    double sum = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        sum += f(a + i * h) * (i % 2 == 0 ? 2.0 : 4.0);
    }
    return sum * h / 3.0;
}

}  // namespace

std::complex<double> HestonPricer::char_function(std::complex<double> phi, double spot,
                                                    double rate, double dividend_yield,
                                                    double maturity, const HestonParams& hp,
                                                    int j) {
    const cdouble i(0.0, 1.0);
    const double u = (j == 1) ? 0.5 : -0.5;
    const double b = (j == 1) ? (hp.kappa - hp.rho * hp.xi) : hp.kappa;
    const double a = hp.kappa * hp.theta;
    const double xi2 = hp.xi * hp.xi;

    const cdouble rho_xi_i_phi = hp.rho * hp.xi * i * phi;
    const cdouble d = std::sqrt((rho_xi_i_phi - b) * (rho_xi_i_phi - b) -
                                 xi2 * (2.0 * u * i * phi - phi * phi));

    // "Little Trap" form: c = 1/g, using exp(-d*tau) (bounded, since
    // std::sqrt's principal branch keeps Re(d) >= 0) instead of exp(+d*tau)
    // -- this is what keeps the log() below from winding around its branch
    // cut as phi or tau grows, unlike the original Heston (1993) formula.
    const cdouble bmr = b - rho_xi_i_phi;  // "b minus rho*xi*i*phi"
    const cdouble c = (bmr - d) / (bmr + d);

    const cdouble exp_neg_d_tau = std::exp(-d * maturity);
    const cdouble log_term = std::log((1.0 - c * exp_neg_d_tau) / (1.0 - c));

    const cdouble C = (rate - dividend_yield) * i * phi * maturity +
                       (a / xi2) * ((bmr - d) * maturity - 2.0 * log_term);
    const cdouble D = ((bmr - d) / xi2) * ((1.0 - exp_neg_d_tau) / (1.0 - c * exp_neg_d_tau));

    return std::exp(C + D * hp.v0 + i * phi * std::log(spot));
}

double HestonPricer::probability(double spot, double strike, double rate, double dividend_yield,
                                  double maturity, const HestonParams& hp, int j) {
    const cdouble i(0.0, 1.0);
    const double log_strike = std::log(strike);

    auto integrand = [&](double phi_real) -> double {
        const cdouble phi(phi_real, 0.0);
        const cdouble cf = char_function(phi, spot, rate, dividend_yield, maturity, hp, j);
        const cdouble numerator = std::exp(-i * phi * log_strike) * cf;
        return (numerator / (i * phi)).real();
    };

    // The integrand has a removable singularity at phi=0 (finite limit),
    // so we start just past it rather than evaluating exactly there (an
    // approximation, not a proven-safe closed-form limit the way QuantLib's
    // AnalyticHestonEngine handles it via L'Hopital). The fixed truncation
    // at phi_max=200 with 4000 Simpson points has only been validated on
    // the moderate maturity/vol-of-vol ranges exercised by this project's
    // tests -- it has NOT been stress-tested against very short maturities
    // or very high vol-of-vol, where the characteristic function decays
    // more slowly and could silently lose accuracy at this fixed cutoff.
    // Production engines (QuantLib) use ~144-point Gauss-Laguerre or
    // adaptive Gauss-Lobatto quadrature here -- both far cheaper per
    // evaluation and error-controlled, unlike this fixed-step rule. Fine
    // for a single price; wasteful (and unverified at the edges) if used
    // inside a tight calibration loop at scale.
    const double integral = simpson_integrate(integrand, 1e-8, 200.0, 4000);
    return 0.5 + integral / kPi;
}

double HestonPricer::price(double spot, double strike, double rate, double dividend_yield,
                            double maturity, OptionType type, const HestonParams& hp) {
    const double p1 = probability(spot, strike, rate, dividend_yield, maturity, hp, 1);
    const double p2 = probability(spot, strike, rate, dividend_yield, maturity, hp, 2);

    const double call = spot * std::exp(-dividend_yield * maturity) * p1 -
                         strike * std::exp(-rate * maturity) * p2;
    if (type == OptionType::Call) {
        return call;
    }
    // put-call parity, rather than re-deriving a separate put formula
    return call - spot * std::exp(-dividend_yield * maturity) + strike * std::exp(-rate * maturity);
}

bool HestonPricer::satisfies_feller_condition(const HestonParams& hp) {
    return 2.0 * hp.kappa * hp.theta >= hp.xi * hp.xi;
}

HestonMCPricer::HestonMCPricer(std::uint64_t seed) : rng_(seed) {}

MCResult HestonMCPricer::price(double spot, double strike, double rate, double dividend_yield,
                                double maturity, OptionType type, const HestonParams& hp,
                                long num_paths, int num_steps) {
    const double dt = maturity / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_one_minus_rho2 = std::sqrt(std::max(1.0 - hp.rho * hp.rho, 0.0));
    const double discount = std::exp(-rate * maturity);

    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0.0;
    double sum_sq = 0.0;
    for (long p = 0; p < num_paths; ++p) {
        double s = spot;
        double v = hp.v0;
        for (int step = 0; step < num_steps; ++step) {
            const double z_v = normal(rng_);
            const double z_indep = normal(rng_);
            const double z_s = hp.rho * z_v + sqrt_one_minus_rho2 * z_indep;

            const double v_pos = std::max(v, 0.0);  // full truncation
            const double sqrt_v_pos = std::sqrt(v_pos);

            const double v_next = v + hp.kappa * (hp.theta - v_pos) * dt +
                                   hp.xi * sqrt_v_pos * sqrt_dt * z_v;
            const double s_next =
                s * std::exp((rate - dividend_yield - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z_s);

            v = v_next;
            s = s_next;
        }

        const double payoff = (type == OptionType::Call) ? std::max(s - strike, 0.0)
                                                           : std::max(strike - s, 0.0);
        const double discounted = discount * payoff;
        sum += discounted;
        sum_sq += discounted * discounted;
    }

    const double mean = sum / static_cast<double>(num_paths);
    double variance = sum_sq / static_cast<double>(num_paths) - mean * mean;
    variance = std::max(variance, 0.0);
    const double std_error = std::sqrt(variance / static_cast<double>(num_paths));

    return {mean, std_error};
}

}  // namespace bscpp
