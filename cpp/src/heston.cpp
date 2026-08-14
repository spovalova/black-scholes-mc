#include "bscpp/heston.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

#include "bscpp/portable_normal.hpp"

namespace bscpp {

namespace {

using cdouble = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

// Adaptive Simpson's rule (recursive, Richardson-extrapolated error
// control): standard textbook algorithm, not a hardcoded quadrature table.
// Deliberately NOT a fixed-node Gauss-Laguerre rule here, despite that
// being what production Heston engines (QuantLib) typically use for
// speed -- transcribing a table of Gauss-Laguerre nodes/weights from
// memory risks a silent, hard-to-detect transcription error in exactly
// the kind of magic-number table this project has otherwise avoided
// trusting without independent verification. Adaptive Simpson has no such
// risk (every step is a locally-checkable Simpson's rule) and, as a
// bonus, concentrates evaluations only where the integrand actually
// varies -- typically far fewer evaluations than the old fixed 4000-point
// rule for smooth cases, while also being more accurate (not less) on the
// oscillatory/short-maturity cases the fixed rule was never verified
// against.
double adaptive_simpson_recursive(const std::function<double(double)>& f, double a, double b,
                                   double fa, double fm, double fb, double whole, double tol,
                                   int depth) {
    constexpr int kMaxDepth = 40;
    const double m = 0.5 * (a + b);
    const double lm = 0.5 * (a + m);
    const double rm = 0.5 * (m + b);
    const double flm = f(lm);
    const double frm = f(rm);
    const double left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    const double right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    const double combined = left + right;

    if (depth >= kMaxDepth || std::abs(combined - whole) <= 15.0 * tol) {
        return combined + (combined - whole) / 15.0;  // Richardson extrapolation
    }
    return adaptive_simpson_recursive(f, a, m, fa, flm, fm, left, tol / 2.0, depth + 1) +
           adaptive_simpson_recursive(f, m, b, fm, frm, fb, right, tol / 2.0, depth + 1);
}

double adaptive_simpson(const std::function<double(double)>& f, double a, double b, double tol) {
    const double fa = f(a);
    const double fb = f(b);
    const double fm = f(0.5 * (a + b));
    const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return adaptive_simpson_recursive(f, a, b, fa, fm, fb, whole, tol, 0);
}

// Integrates f from near-0 to infinity by adaptive-Simpson-ing an initial
// panel, then doubling the upper bound and adaptively integrating each new
// panel until its own contribution is negligible. Replaces the old fixed
// phi_max=200 truncation (whose adequacy at extreme parameters was
// explicitly UNVERIFIED) with a self-terminating, measured stopping
// condition -- the integral only stops growing the domain once the tail
// itself proves negligible, whatever the parameters.
double integrate_to_infinity(const std::function<double(double)>& f, double tol) {
    double lo = 1e-8;
    double hi = 50.0;
    double total = adaptive_simpson(f, lo, hi, tol);
    constexpr int kMaxExtensions = 12;  // hi grows to 50*2^12 ~ 2e5 in the worst case
    for (int i = 0; i < kMaxExtensions; ++i) {
        const double next_hi = hi * 2.0;
        const double segment = adaptive_simpson(f, hi, next_hi, tol);
        total += segment;
        hi = next_hi;
        if (std::abs(segment) < tol) {
            break;
        }
    }
    return total;
}

// Fixed composite-Simpson quadrature nodes and weights on [a, b] with n
// (even) subintervals -- used only by price_batch, where nodes must be
// IDENTICAL across strikes so the (expensive) characteristic-function
// evaluation at each node can be shared. Deliberately generous, fixed
// resolution rather than a memorized table of magic numbers; validated
// against the adaptive single-price path across the same stress cases
// price() itself was verified against (see test_heston.py) rather than
// assumed to inherit that accuracy for free.
void fixed_simpson_grid(double a, double b, int n, std::vector<double>& nodes,
                         std::vector<double>& weights) {
    if (n % 2 != 0) ++n;
    const double h = (b - a) / n;
    nodes.resize(static_cast<size_t>(n) + 1);
    weights.resize(static_cast<size_t>(n) + 1);
    for (int k = 0; k <= n; ++k) {
        nodes[static_cast<size_t>(k)] = a + k * h;
        const double w = (k == 0 || k == n) ? 1.0 : (k % 2 == 0 ? 2.0 : 4.0);
        weights[static_cast<size_t>(k)] = w * h / 3.0;
    }
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
    // AnalyticHestonEngine handles it via L'Hopital -- fine numerically
    // since dividing a ~1e-8-scale imaginary part by ~1e-8 doesn't
    // destabilize in double precision, but still worth naming as a
    // simplification rather than a proven-exact treatment).
    const double integral = integrate_to_infinity(integrand, 1e-10);
    return 0.5 + integral / kPi;
}

double HestonPricer::price(double spot, double strike, double rate, double dividend_yield,
                            double maturity, OptionType type, const HestonParams& hp) {
    const double p1 = probability(spot, strike, rate, dividend_yield, maturity, hp, 1);
    const double p2 = probability(spot, strike, rate, dividend_yield, maturity, hp, 2);

    const double call = spot * std::exp(-dividend_yield * maturity) * p1 -
                         strike * std::exp(-rate * maturity) * p2;
    // Deep OTM at short maturity, quadrature noise can exceed the (tiny)
    // true price and produce a slightly NEGATIVE value (observed: -2.3e-7
    // for a 5-day 200-strike call on spot 100). A negative option price is
    // itself an arbitrage and poisons downstream IV solves, so clamp both
    // legs at the no-arbitrage floor of zero.
    if (type == OptionType::Call) {
        return std::max(call, 0.0);
    }
    // put-call parity, rather than re-deriving a separate put formula
    const double put =
        call - spot * std::exp(-dividend_yield * maturity) + strike * std::exp(-rate * maturity);
    return std::max(put, 0.0);
}

std::vector<double> HestonPricer::price_batch(double spot, const std::vector<double>& strikes,
                                               const std::vector<OptionType>& types, double rate,
                                               double dividend_yield, double maturity,
                                               const HestonParams& hp, int num_nodes,
                                               double phi_max) {
    if (strikes.size() != types.size()) {
        throw std::invalid_argument("strikes and types must be the same length");
    }

    constexpr double kEps = 1e-8;
    std::vector<double> nodes, weights;
    fixed_simpson_grid(kEps, phi_max, num_nodes, nodes, weights);

    // The expensive part -- complex sqrt/log/exp inside char_function --
    // evaluated once per node, shared across every strike below, since
    // char_function does not depend on strike at all.
    std::vector<cdouble> cf1(nodes.size()), cf2(nodes.size());
    for (size_t k = 0; k < nodes.size(); ++k) {
        const cdouble phi(nodes[k], 0.0);
        cf1[k] = char_function(phi, spot, rate, dividend_yield, maturity, hp, 1);
        cf2[k] = char_function(phi, spot, rate, dividend_yield, maturity, hp, 2);
    }

    const cdouble i(0.0, 1.0);
    const double disc_q = std::exp(-dividend_yield * maturity);
    const double disc_r = std::exp(-rate * maturity);

    std::vector<double> out(strikes.size());
    for (size_t s = 0; s < strikes.size(); ++s) {
        const double log_strike = std::log(strikes[s]);
        double integral1 = 0.0, integral2 = 0.0;
        for (size_t k = 0; k < nodes.size(); ++k) {
            const cdouble phi(nodes[k], 0.0);
            const cdouble phase = std::exp(-i * phi * log_strike);
            integral1 += weights[k] * (phase * cf1[k] / (i * phi)).real();
            integral2 += weights[k] * (phase * cf2[k] / (i * phi)).real();
        }
        const double p1 = 0.5 + integral1 / kPi;
        const double p2 = 0.5 + integral2 / kPi;

        const double call = spot * disc_q * p1 - strikes[s] * disc_r * p2;
        if (types[s] == OptionType::Call) {
            out[s] = std::max(call, 0.0);
        } else {
            const double put = call - spot * disc_q + strikes[s] * disc_r;
            out[s] = std::max(put, 0.0);
        }
    }
    return out;
}

bool HestonPricer::satisfies_feller_condition(const HestonParams& hp) {
    return 2.0 * hp.kappa * hp.theta >= hp.xi * hp.xi;
}

HestonMCPricer::HestonMCPricer(std::uint64_t seed) : seed_(seed) {}

MCResult HestonMCPricer::price(double spot, double strike, double rate, double dividend_yield,
                                double maturity, OptionType type, const HestonParams& hp,
                                long num_paths, int num_steps) {
    const double dt = maturity / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_one_minus_rho2 = std::sqrt(std::max(1.0 - hp.rho * hp.rho, 0.0));
    const double discount = std::exp(-rate * maturity);

    // 2 normals/step (z_v, z_indep) => 1 Philox block/step (2 normals per
    // block) => num_steps blocks per path; see monte_carlo.cpp/
    // longstaff_schwartz.cpp for the same disjoint-counter-range pattern.
    const std::uint64_t per_path = static_cast<std::uint64_t>(num_steps);
    const std::uint64_t base = cursor_;
    cursor_ += static_cast<std::uint64_t>(num_paths) * per_path;

    double sum = 0.0;
    double sum_sq = 0.0;
#pragma omp parallel for reduction(+ : sum, sum_sq)
    for (long p = 0; p < num_paths; ++p) {
        Philox4x64 local(seed_);
        local.seek(base + static_cast<std::uint64_t>(p) * per_path);
        double s = spot;
        double v = hp.v0;
        for (int step = 0; step < num_steps; ++step) {
            const double z_v = standard_normal(local);
            const double z_indep = standard_normal(local);
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
