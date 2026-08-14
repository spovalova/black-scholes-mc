#include "bscpp/heston.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

#include "bscpp/dual.hpp"
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
// Templated on the integrand's return type T (double for the original
// value-only quadrature; RealDual5 for the Jacobian, see char_function_impl
// below) so both share ONE copy of the adaptive-refinement logic instead
// of a hand-duplicated second implementation. The Richardson-extrapolation
// arithmetic and error control both flow through T (via RealDual5's own
// +,-,*,/ and abs_value(), see dual.hpp), with abs_value() specifically
// so refinement is always driven by how well the integral's VALUE is
// resolved -- not its derivatives, which have no natural single scalar to
// compare against tol and simply come along for the ride at whatever
// nodes the value-driven refinement already chose.
template <typename T>
T adaptive_simpson_recursive(const std::function<T(double)>& f, double a, double b, T fa, T fm,
                              T fb, T whole, double tol, int depth) {
    constexpr int kMaxDepth = 40;
    const double m = 0.5 * (a + b);
    const double lm = 0.5 * (a + m);
    const double rm = 0.5 * (m + b);
    const T flm = f(lm);
    const T frm = f(rm);
    const T left = (fa + flm * 4.0 + fm) * ((m - a) / 6.0);
    const T right = (fm + frm * 4.0 + fb) * ((b - m) / 6.0);
    const T combined = left + right;

    if (depth >= kMaxDepth || abs_value(combined - whole) <= 15.0 * tol) {
        return combined + (combined - whole) / 15.0;  // Richardson extrapolation
    }
    return adaptive_simpson_recursive<T>(f, a, m, fa, flm, fm, left, tol / 2.0, depth + 1) +
           adaptive_simpson_recursive<T>(f, m, b, fm, frm, fb, right, tol / 2.0, depth + 1);
}

template <typename T>
T adaptive_simpson(const std::function<T(double)>& f, double a, double b, double tol) {
    const T fa = f(a);
    const T fb = f(b);
    const T fm = f(0.5 * (a + b));
    const T whole = (fa + fm * 4.0 + fb) * ((b - a) / 6.0);
    return adaptive_simpson_recursive<T>(f, a, b, fa, fm, fb, whole, tol, 0);
}

// Integrates f from near-0 to infinity by adaptive-Simpson-ing an initial
// panel, then doubling the upper bound and adaptively integrating each new
// panel until its own contribution is negligible. Replaces the old fixed
// phi_max=200 truncation (whose adequacy at extreme parameters was
// explicitly UNVERIFIED) with a self-terminating, measured stopping
// condition -- the integral only stops growing the domain once the tail
// itself proves negligible, whatever the parameters.
template <typename T>
T integrate_to_infinity(const std::function<T(double)>& f, double tol) {
    double lo = 1e-8;
    double hi = 50.0;
    T total = adaptive_simpson<T>(f, lo, hi, tol);
    constexpr int kMaxExtensions = 12;  // hi grows to 50*2^12 ~ 2e5 in the worst case
    for (int i = 0; i < kMaxExtensions; ++i) {
        const double next_hi = hi * 2.0;
        const T segment = adaptive_simpson<T>(f, hi, next_hi, tol);
        total = total + segment;
        hi = next_hi;
        if (abs_value(segment) < tol) {
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

// COS method (Fang & Oosterlee 2008) helpers.
//
// chi_k/psi_k are the standard cosine-series antiderivative coefficients
// for the payoff function's exponential (chi) and constant (psi) pieces
// over an integration range [c,d] within the truncation domain [a,b].
// Textbook closed forms (e.g. Fang & Oosterlee 2008 eqs. 22-23) -- no
// magic-number table, just trig/exp evaluated per k.
double cos_chi(double k, double a, double b, double c, double d) {
    const double u = k * kPi / (b - a);
    const double u2p1 = 1.0 + u * u;
    const double term_d = std::cos(u * (d - a)) * std::exp(d) + u * std::sin(u * (d - a)) * std::exp(d);
    const double term_c = std::cos(u * (c - a)) * std::exp(c) + u * std::sin(u * (c - a)) * std::exp(c);
    return (term_d - term_c) / u2p1;
}

double cos_psi(double k, double a, double b, double c, double d) {
    if (k == 0.0) {
        return d - c;
    }
    const double u = k * kPi / (b - a);
    return (std::sin(u * (d - a)) - std::sin(u * (c - a))) / u;
}

// The Heston (1993) / "Little Trap" characteristic function, templated on
// the scalar type T used for kappa/theta/xi/rho/v0. T=double reproduces
// today's exact arithmetic (mixing plain double hp-fields with cdouble
// phi, exactly as before templating -- verified byte-for-byte unchanged
// in test_heston.py, not just assumed from the refactor being
// "mechanical"). T=ComplexDual5 (dual.hpp) computes the SAME formula
// while additionally propagating derivatives w.r.t. whichever of
// kappa/theta/xi/rho/v0 were seeded via ComplexDual5::variable -- used by
// HestonPricer::price_jacobian below.
//
// Templating (one formula, two instantiations) rather than hand-writing a
// second copy for the Jacobian path is deliberate: this project extracted
// brent.hpp for the same reason (a second copy of the same logic is a
// second place for a transcription error to hide, invisible until it
// disagrees with the first copy on some untested input).
//
// `using std::sqrt;` etc. below + unqualified calls is what lets normal
// C++ lookup pick std::sqrt/exp/log for T=double's cdouble/double
// arguments and bscpp::sqrt/exp/log (dual.hpp, found via ADL since
// ComplexDual5 lives in namespace bscpp) for T=ComplexDual5 -- the same
// customization-point idiom std::swap relies on.
template <typename T>
auto char_function_impl(cdouble phi, double spot, double rate, double dividend_yield,
                         double maturity, T kappa, T theta, T xi, T rho, T v0, int j) {
    using std::exp;
    using std::log;
    using std::sqrt;
    const cdouble i(0.0, 1.0);
    const double u = (j == 1) ? 0.5 : -0.5;
    const auto b = (j == 1) ? (kappa - rho * xi) : kappa;
    const auto a = kappa * theta;
    const auto xi2 = xi * xi;

    const auto rho_xi_i_phi = rho * xi * i * phi;
    const auto d = sqrt((rho_xi_i_phi - b) * (rho_xi_i_phi - b) -
                         xi2 * (2.0 * u * i * phi - phi * phi));

    // "Little Trap" form: c = 1/g, using exp(-d*tau) (bounded, since
    // std::sqrt's principal branch keeps Re(d) >= 0) instead of exp(+d*tau)
    // -- this is what keeps the log() below from winding around its branch
    // cut as phi or tau grows, unlike the original Heston (1993) formula.
    const auto bmr = b - rho_xi_i_phi;  // "b minus rho*xi*i*phi"
    const auto c = (bmr - d) / (bmr + d);

    const auto exp_neg_d_tau = exp(-d * maturity);
    const auto log_term = log((1.0 - c * exp_neg_d_tau) / (1.0 - c));

    const auto C = (rate - dividend_yield) * i * phi * maturity +
                    (a / xi2) * ((bmr - d) * maturity - 2.0 * log_term);
    const auto D = ((bmr - d) / xi2) * ((1.0 - exp_neg_d_tau) / (1.0 - c * exp_neg_d_tau));

    return exp(C + D * v0 + i * phi * std::log(spot));
}

}  // namespace

std::complex<double> HestonPricer::char_function(std::complex<double> phi, double spot,
                                                    double rate, double dividend_yield,
                                                    double maturity, const HestonParams& hp,
                                                    int j) {
    return char_function_impl<double>(phi, spot, rate, dividend_yield, maturity, hp.kappa,
                                       hp.theta, hp.xi, hp.rho, hp.v0, j);
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
    const double integral = integrate_to_infinity<double>(integrand, 1e-10);
    return 0.5 + integral / kPi;
}

namespace {

// Jacobian counterpart of HestonPricer::probability: same formula, same
// quadrature, but with kappa/theta/xi/rho/v0 passed in as ComplexDual5 (at
// most one -- or, for price_jacobian below, all five at once via vector
// forward-mode AD -- actually seeded as a differentiation variable) so the
// integral comes back as a RealDual5 carrying d(P_j)/d(param) alongside
// the value itself. See char_function_impl and dual.hpp for why this
// needs a second (still holomorphic, still Re()-exact) imaginary "unit"
// rather than literal complex-step.
RealDual5 probability_jacobian(double spot, double strike, double rate, double dividend_yield,
                                double maturity, const ComplexDual5& kappa,
                                const ComplexDual5& theta, const ComplexDual5& xi,
                                const ComplexDual5& rho, const ComplexDual5& v0, int j) {
    const cdouble i(0.0, 1.0);
    const double log_strike = std::log(strike);

    auto integrand = [&](double phi_real) -> RealDual5 {
        const cdouble phi(phi_real, 0.0);
        const ComplexDual5 cf = char_function_impl<ComplexDual5>(phi, spot, rate, dividend_yield,
                                                                    maturity, kappa, theta, xi,
                                                                    rho, v0, j);
        const ComplexDual5 numerator = ComplexDual5(std::exp(-i * phi * log_strike)) * cf;
        return real_part(numerator / ComplexDual5(i * phi));
    };

    const RealDual5 integral = integrate_to_infinity<RealDual5>(integrand, 1e-10);
    return RealDual5(0.5) + integral / kPi;
}

}  // namespace

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

HestonJacobian HestonPricer::price_jacobian(double spot, double strike, double rate,
                                             double dividend_yield, double maturity,
                                             OptionType type, const HestonParams& hp) {
    // Seed all five parameters at once (vector forward-mode AD): each
    // gets derivative 1 w.r.t. itself and 0 w.r.t. the other four, so a
    // SINGLE pass through probability_jacobian's quadrature recovers all
    // 5 partials together, not one pass per parameter.
    const ComplexDual5 kappa = ComplexDual5::variable(hp.kappa, 0);
    const ComplexDual5 theta = ComplexDual5::variable(hp.theta, 1);
    const ComplexDual5 xi = ComplexDual5::variable(hp.xi, 2);
    const ComplexDual5 rho = ComplexDual5::variable(hp.rho, 3);
    const ComplexDual5 v0 = ComplexDual5::variable(hp.v0, 4);

    const RealDual5 p1 =
        probability_jacobian(spot, strike, rate, dividend_yield, maturity, kappa, theta, xi, rho,
                              v0, 1);
    const RealDual5 p2 =
        probability_jacobian(spot, strike, rate, dividend_yield, maturity, kappa, theta, xi, rho,
                              v0, 2);

    const double disc_q = std::exp(-dividend_yield * maturity);
    const double disc_r = std::exp(-rate * maturity);

    const RealDual5 call = p1 * (spot * disc_q) - p2 * (strike * disc_r);
    const RealDual5 raw = (type == OptionType::Call)
                               ? call
                               : call - RealDual5(spot * disc_q) + RealDual5(strike * disc_r);

    // Same no-arbitrage floor price() applies (std::max(price, 0.0)) --
    // but max(x,0) isn't differentiable exactly at x=0, so its correct
    // subgradient (0 for every partial) is used explicitly here rather
    // than left ambiguous. Only matters for parameter combinations that
    // would ALREADY be numerically dubious for price() itself (deep OTM,
    // short-dated) -- a calibration converging near that floor has a
    // separate, existing problem the Jacobian can't paper over.
    if (raw.value <= 0.0) {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    return {raw.value,       raw.deriv[0], raw.deriv[1],
             raw.deriv[2],    raw.deriv[3], raw.deriv[4]};
}

std::vector<HestonJacobian> HestonPricer::price_jacobian_batch(
    double spot, const std::vector<double>& strikes, const std::vector<OptionType>& types,
    double rate, double dividend_yield, double maturity, const HestonParams& hp, int num_nodes,
    double phi_max) {
    if (strikes.size() != types.size()) {
        throw std::invalid_argument("strikes and types must be the same length");
    }

    const ComplexDual5 kappa = ComplexDual5::variable(hp.kappa, 0);
    const ComplexDual5 theta = ComplexDual5::variable(hp.theta, 1);
    const ComplexDual5 xi = ComplexDual5::variable(hp.xi, 2);
    const ComplexDual5 rho = ComplexDual5::variable(hp.rho, 3);
    const ComplexDual5 v0 = ComplexDual5::variable(hp.v0, 4);

    constexpr double kEps = 1e-8;
    std::vector<double> nodes, weights;
    fixed_simpson_grid(kEps, phi_max, num_nodes, nodes, weights);

    // Same sharing price_batch relies on: char_function (now
    // char_function_impl<ComplexDual5>) does not depend on strike, so
    // it's evaluated once per node -- INCLUDING its 5 derivatives -- and
    // reused across every strike below, instead of redoing the whole
    // (5x wider) dual-number quadrature from scratch per strike.
    std::vector<ComplexDual5> cf1(nodes.size()), cf2(nodes.size());
    for (size_t k = 0; k < nodes.size(); ++k) {
        const cdouble phi(nodes[k], 0.0);
        cf1[k] = char_function_impl<ComplexDual5>(phi, spot, rate, dividend_yield, maturity,
                                                     kappa, theta, xi, rho, v0, 1);
        cf2[k] = char_function_impl<ComplexDual5>(phi, spot, rate, dividend_yield, maturity,
                                                     kappa, theta, xi, rho, v0, 2);
    }

    const cdouble i(0.0, 1.0);
    const double disc_q = std::exp(-dividend_yield * maturity);
    const double disc_r = std::exp(-rate * maturity);

    std::vector<HestonJacobian> out(strikes.size());
    for (size_t s = 0; s < strikes.size(); ++s) {
        const double log_strike = std::log(strikes[s]);
        RealDual5 integral1(0.0), integral2(0.0);
        for (size_t k = 0; k < nodes.size(); ++k) {
            const cdouble phi(nodes[k], 0.0);
            const ComplexDual5 phase(std::exp(-i * phi * log_strike));
            integral1 = integral1 + real_part(phase * cf1[k] / ComplexDual5(i * phi)) * weights[k];
            integral2 = integral2 + real_part(phase * cf2[k] / ComplexDual5(i * phi)) * weights[k];
        }
        const RealDual5 p1 = RealDual5(0.5) + integral1 / kPi;
        const RealDual5 p2 = RealDual5(0.5) + integral2 / kPi;

        const RealDual5 call = p1 * (spot * disc_q) - p2 * (strikes[s] * disc_r);
        const RealDual5 raw = (types[s] == OptionType::Call)
                                   ? call
                                   : call - RealDual5(spot * disc_q) +
                                         RealDual5(strikes[s] * disc_r);

        out[s] = (raw.value <= 0.0)
                     ? HestonJacobian{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
                     : HestonJacobian{raw.value,    raw.deriv[0], raw.deriv[1],
                                        raw.deriv[2], raw.deriv[3], raw.deriv[4]};
    }
    return out;
}

namespace {
// One COS evaluation at a fixed (kL, num_terms) resolution -- returns the
// UNCLAMPED (pre no-arbitrage-floor) price, since the floor can make two
// genuinely-different-but-both-garbage estimates compare equal (both
// clamped to exactly 0.0) and falsely look "converged" to the caller's
// fixed-point iteration below.
double price_cos_raw(double strike, double rate, double maturity, OptionType type,
                      double c1, double c2, double kL, int num_terms,
                      const std::function<std::complex<double>(double)>& cf_at) {
    const cdouble i(0.0, 1.0);
    const double std_dev = std::sqrt(std::max(c2, 0.0));
    const double a = c1 - kL * std_dev;
    const double b = c1 + kL * std_dev;

    const double log_strike = std::log(strike);
    if (type == OptionType::Call && log_strike >= b) return 0.0;
    if (type == OptionType::Put && log_strike <= a) return 0.0;
    const double c_lo = (type == OptionType::Call) ? std::max(log_strike, a) : a;
    const double c_hi = (type == OptionType::Call) ? b : std::min(log_strike, b);

    // Payoff coefficients V_k = integral of the payoff against the k-th
    // cosine basis function over [c_lo,c_hi], scaled by 2/(b-a) to match
    // the A_k normalization below. NOTE: since x here is the ABSOLUTE
    // log-price ln(S_T) (not Fang & Oosterlee's log-moneyness x=ln(S_T/K)
    // convention), the payoff is (e^x - K) for a call / (K - e^x) for a
    // put -- so K multiplies only the psi (constant-term) integral, not
    // the chi (e^x-term) integral. Using their paper's literal
    // K*(chi-psi) formula here would silently assume the log-moneyness
    // convention and be wrong by construction; this was caught by
    // cross-checking against a from-scratch BS-COS reimplementation
    // (same chi/psi, textbook BS characteristic function) before this
    // method was trusted -- see test_heston.py.
    double sum = 0.0;
    for (int k = 0; k < num_terms; ++k) {
        const double kd = static_cast<double>(k);
        double v_k;
        if (type == OptionType::Call) {
            v_k = (2.0 / (b - a)) *
                  (cos_chi(kd, a, b, c_lo, c_hi) - strike * cos_psi(kd, a, b, c_lo, c_hi));
        } else {
            v_k = (2.0 / (b - a)) *
                  (strike * cos_psi(kd, a, b, c_lo, c_hi) - cos_chi(kd, a, b, c_lo, c_hi));
        }
        const double u = kd * kPi / (b - a);
        const cdouble cf = cf_at(u);
        const cdouble phase = std::exp(-i * u * a);
        const double weight = (k == 0) ? 0.5 : 1.0;  // first term counted at half weight
        sum += weight * (cf * phase).real() * v_k;
    }
    return std::exp(-rate * maturity) * sum;
}
}  // namespace

double HestonPricer::price_cos(double spot, double strike, double rate, double dividend_yield,
                                double maturity, OptionType type, const HestonParams& hp,
                                int num_terms) {
    // Cumulants of x=ln(S_T), estimated NUMERICALLY via finite differences
    // on ln(char_function(..., j=2)) -- char_function's j=2 branch is
    // already the standard risk-neutral CF of ln(S_T) (no separate
    // re-derivation), so this reuses the same, already-verified formula
    // price() itself is built on. ln(phi(u)) = iu*c1 - (u^2/2)*c2 +
    // O(u^3) for small u, so c1 = -i*d/du[ln phi](0), c2 =
    // -d^2/du^2[ln phi](0), both via central differences.
    constexpr double h = 1e-4;
    const auto log_cf = [&](double u) {
        const cdouble phi(u, 0.0);
        return std::log(char_function(phi, spot, rate, dividend_yield, maturity, hp, 2));
    };
    const cdouble l_plus = log_cf(h);
    const cdouble l_minus = log_cf(-h);
    const cdouble l_zero = log_cf(0.0);  // ~0 exactly; kept for the 2nd-difference formula's clarity
    const cdouble i(0.0, 1.0);
    const double c1 = (-i * (l_plus - l_minus) / (2.0 * h)).real();
    const double c2 = (-(l_plus - 2.0 * l_zero + l_minus) / (h * h)).real();

    const auto cf_at = [&](double u) {
        return char_function(cdouble(u, 0.0), spot, rate, dividend_yield, maturity, hp, 2);
    };

    // A single fixed (kL, num_terms) choice is NOT robust across this
    // pricer's full parameter range: a narrow-but-cheap domain undershoots
    // for long maturities / badly Feller-violating vol-of-vol (fat-tailed
    // ln(S_T)), while a domain wide enough for those cases wastes cycles
    // -- and needs far more terms -- on the common (short/moderate
    // maturity, well-behaved) case. So this widens the truncation range
    // and term count together, iteration by iteration, and stops as soon
    // as two successive estimates agree -- the same "self-terminating
    // rather than assumed at a fixed truncation" philosophy price()
    // already uses for its adaptive quadrature.
    //
    // Comparing against price(), not just num_terms, matters: holding kL
    // fixed and only growing num_terms can converge cleanly to a value
    // that's simply wrong, because the DOMAIN (not the resolution within
    // it) was too narrow -- num_terms convergence alone can't detect
    // that. Both must widen together, and the loop must be willing to
    // fall back to the trusted (if slower) adaptive-quadrature price()
    // rather than trust an estimate it never confirmed.
    //
    // Verified (test_heston.py) against price() across a 300-case random
    // sweep spanning maturities from 1 day to 3 years, kappa/theta/xi/rho/v0
    // covering both well-behaved and badly Feller-violating regimes, and
    // strikes from deep ITM to deep OTM -- including a case where, without
    // the price()-fallback below, two successive iterations both landed on
    // the no-arbitrage floor of exactly 0.0 (a spurious "converged" false
    // positive on genuinely divergent, not just imprecise, estimates) that
    // masked a true price of ~0.8; see the raw (unclamped) comparison used
    // below specifically to prevent that.
    double kL = 10.0;
    int terms = std::max(64, num_terms / 4);
    double prev = price_cos_raw(strike, rate, maturity, type, c1, c2, kL, terms, cf_at);
    constexpr int kMaxIters = 6;
    for (int iter = 0; iter < kMaxIters; ++iter) {
        kL += 4.0;
        terms *= 2;
        const double cur = price_cos_raw(strike, rate, maturity, type, c1, c2, kL, terms, cf_at);
        const double tol = 1e-6 + 1e-5 * std::abs(cur);
        if (std::abs(cur - prev) < tol) {
            return std::max(cur, 0.0);  // same no-arbitrage floor as price()
        }
        prev = cur;
    }
    // Never converged to the required tolerance within the iteration
    // budget -- rather than hand back an unconfirmed (possibly wildly
    // wrong, per the case above) estimate, fall back to the trusted
    // adaptive-quadrature price(). This only fires for the pathological
    // tail of the parameter space (see test_heston.py); price_cos still
    // returns its fast COS estimate for everything that converges.
    return price(spot, strike, rate, dividend_yield, maturity, type, hp);
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

namespace {

// Two independent U(0,1] draws, same raw-Philox convention
// standard_normal (portable_normal.hpp) uses -- kept local to this file,
// not folded into that shared/already-verified header, because Andersen
// QE's Exponential branch needs the RAW uniform directly (for its
// inverse-CDF sampling), not just a Box-Muller-transformed normal.
struct UniformPair {
    double u1, u2;
};

UniformPair draw_uniform_pair(Philox4x64& rng) {
    constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^-53
    double u1;
    do {
        u1 = static_cast<double>(rng() >> 11) * kScale;
    } while (u1 <= 0.0);
    const double u2 = static_cast<double>(rng() >> 11) * kScale;
    return {u1, u2};
}

double box_muller_normal(const UniformPair& u) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    return std::sqrt(-2.0 * std::log(u.u1)) * std::cos(kTwoPi * u.u2);
}

}  // namespace

MCResult HestonMCPricer::price_qe(double spot, double strike, double rate, double dividend_yield,
                                   double maturity, OptionType type, const HestonParams& hp,
                                   long num_paths, int num_steps) {
    const double dt = maturity / num_steps;
    const double discount = std::exp(-rate * maturity);
    const double exp_neg_kappa_dt = std::exp(-hp.kappa * dt);

    // Andersen (2008) sec. 3.2: psi_c=1.5 is the paper's own recommended
    // switching threshold between the Quadratic (small local variance-to-
    // mean-squared ratio -- the CIR conditional law is well-approximated
    // by a squared, shifted Gaussian) and Exponential (large ratio, where
    // a squared-Gaussian can no longer match the true law's heavier tail)
    // branches. Not tuned here; the paper's own value, used as published.
    constexpr double kPsiC = 1.5;

    // gamma1=gamma2=0.5: Andersen's "central" discretization for the log-
    // price step (K0..K4 below) -- the standard, not martingale-corrected,
    // choice; see price_qe's header doc for why the correction wasn't
    // needed at the step counts this method is verified at.
    const double K0 = -hp.rho * hp.kappa * hp.theta * dt / hp.xi;
    const double K1 = 0.5 * dt * (hp.kappa * hp.rho / hp.xi - 0.5) - hp.rho / hp.xi;
    const double K2 = 0.5 * dt * (hp.kappa * hp.rho / hp.xi - 0.5) + hp.rho / hp.xi;
    const double K3 = 0.5 * dt * (1.0 - hp.rho * hp.rho);
    const double K4 = 0.5 * dt * (1.0 - hp.rho * hp.rho);
    const double drift = (rate - dividend_yield) * dt + K0;

    // 2 uniform-pairs/step (one for the v-branch draw, one Box-Muller'd
    // into Z_s) => 4 raw draws/step => 1 Philox block/step, same
    // per-path/per-step block accounting as price() above.
    const std::uint64_t per_path = static_cast<std::uint64_t>(num_steps);
    const std::uint64_t base = cursor_;
    cursor_ += static_cast<std::uint64_t>(num_paths) * per_path;

    double sum = 0.0;
    double sum_sq = 0.0;
#pragma omp parallel for reduction(+ : sum, sum_sq)
    for (long p = 0; p < num_paths; ++p) {
        Philox4x64 local(seed_);
        local.seek(base + static_cast<std::uint64_t>(p) * per_path);
        double log_s = std::log(spot);
        double v = hp.v0;
        for (int step = 0; step < num_steps; ++step) {
            const UniformPair uv = draw_uniform_pair(local);

            const double m = hp.theta + (v - hp.theta) * exp_neg_kappa_dt;
            const double s2 = v * hp.xi * hp.xi * exp_neg_kappa_dt / hp.kappa *
                                   (1.0 - exp_neg_kappa_dt) +
                               hp.theta * hp.xi * hp.xi / (2.0 * hp.kappa) *
                                   (1.0 - exp_neg_kappa_dt) * (1.0 - exp_neg_kappa_dt);
            // m>0 always (theta,v0>0, convex combination); guard s2 against
            // roundoff producing a tiny negative value at v==0.
            const double psi = std::max(s2, 0.0) / (m * m);

            double v_next;
            if (psi <= kPsiC) {
                const double inv_psi = 1.0 / psi;
                const double b2 = 2.0 * inv_psi - 1.0 +
                                   std::sqrt(2.0 * inv_psi) * std::sqrt(2.0 * inv_psi - 1.0);
                const double a = m / (1.0 + b2);
                const double zv = box_muller_normal(uv);
                const double root_b2_plus_zv = std::sqrt(b2) + zv;
                v_next = a * root_b2_plus_zv * root_b2_plus_zv;
            } else {
                const double pr = (psi - 1.0) / (psi + 1.0);
                const double beta = (1.0 - pr) / m;
                v_next = (uv.u1 <= pr) ? 0.0 : std::log((1.0 - pr) / (1.0 - uv.u1)) / beta;
            }

            const UniformPair us = draw_uniform_pair(local);
            const double zs = box_muller_normal(us);
            const double diffusion_var = std::max(K3 * v + K4 * v_next, 0.0);
            log_s += drift + K1 * v + K2 * v_next + std::sqrt(diffusion_var) * zs;

            v = v_next;
        }

        const double s_final = std::exp(log_s);
        const double payoff = (type == OptionType::Call) ? std::max(s_final - strike, 0.0)
                                                           : std::max(strike - s_final, 0.0);
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
