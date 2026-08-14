#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "bscpp/philox.hpp"
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

// price() plus its analytic partial derivatives w.r.t. each of
// kappa/theta/xi/rho/v0 -- see HestonPricer::price_jacobian.
struct HestonJacobian {
    double price;
    double d_kappa;
    double d_theta;
    double d_xi;
    double d_rho;
    double d_v0;
};

// Semi-analytic pricer via the Heston characteristic function.
class HestonPricer {
public:
    // Uses the Albrecher, Mayer, Schoutens & Tistaert (2007) "Little Trap"
    // reformulation of the original Heston (1993) characteristic function,
    // which avoids the branch-cut discontinuities the naive formula hits
    // for long maturities / certain parameter regions. The P1/P2 integrals
    // are evaluated via adaptive Simpson quadrature with adaptive upper-
    // bound extension (not a fixed-node table), specifically so there's no
    // risk of a transcribed magic-number error from a hand-copied
    // Gauss-Laguerre table, and so accuracy is measured/self-terminating
    // rather than assumed at a fixed truncation. Verified accurate even at
    // 1-day maturity and at vol-of-vol badly violating the Feller
    // condition (xi=3.0 against 2*kappa*theta=0.16) -- see test_heston.py.
    static double price(double spot, double strike, double rate, double dividend_yield,
                         double maturity, OptionType type, const HestonParams& hp);

    // Prices a whole strike grid at fixed (spot, rate, dividend_yield,
    // maturity, hp) in one call. This exists because the characteristic
    // function does NOT depend on strike -- only the phase factor
    // exp(-i*phi*ln(K)) inside the P1/P2 integral does -- so evaluating it
    // once per quadrature node and reusing that across every strike turns
    // an O(strikes) cost in the expensive part (complex sqrt/log/exp per
    // node) into O(1). Profiling a single Heston calibration call showed
    // heston_price accounting for 96.8% of runtime (819 individual C++
    // calls for one 13-strike calibration); this is the fix.
    //
    // Uses a fixed (not adaptive) quadrature grid, unlike price() above --
    // adaptive node placement depends on the per-strike integrand, so
    // nodes can't be shared across strikes if they're chosen adaptively.
    // The fixed grid is deliberately generous (not a memorized magic-
    // number table -- a plain composite Simpson resolution choice) and is
    // validated in tests/test_heston.py to agree with the adaptive price()
    // to within a tight tolerance across the SAME stress cases price()
    // itself was verified against (1-day maturity, Feller-violating
    // vol-of-vol) -- not assumed to inherit that accuracy for free.
    // num_nodes/phi_max control the fixed quadrature resolution: higher is
    // more accurate but more expensive PER CALL regardless of how many
    // strikes are batched (the whole point is that cost is shared across
    // strikes, so it does NOT scale with strike count -- but it doesn't
    // shrink for a small strike count either, unlike per-strike adaptive
    // quadrature, which spends less effort on well-behaved integrands).
    // Defaults are tuned for typical calibration use (moderate maturity,
    // moderate vol-of-vol); pass higher values for near-expiry or
    // Feller-violating parameter regimes -- see test_heston.py for the
    // accuracy this buys at each resolution.
    static std::vector<double> price_batch(double spot, const std::vector<double>& strikes,
                                            const std::vector<OptionType>& types, double rate,
                                            double dividend_yield, double maturity,
                                            const HestonParams& hp, int num_nodes = 1500,
                                            double phi_max = 150.0);

    // Necessary (not sufficient in the fitted-to-market sense) condition for
    // the variance process to almost-surely stay strictly positive. Reported
    // as a diagnostic -- many market-calibrated parameter sets violate it in
    // practice, and the pricer still works (variance just clamped at 0) but
    // the model interpretation as "always-positive variance" no longer holds.
    static bool satisfies_feller_condition(const HestonParams& hp);

    // price() plus its exact partial derivatives w.r.t. all five Heston
    // parameters, computed in ONE pass via forward-mode automatic
    // differentiation (dual.hpp's ComplexDual5/RealDual5) rather than
    // scipy's default finite-difference Jacobian -- calibrate_heston
    // (heston_calibration.py) uses this to avoid the ~6 full residual
    // evaluations per iteration (1 base + 5 perturbed) a finite-difference
    // Jacobian costs, and the O(sqrt(eps)) truncation error that comes
    // with it.
    //
    // NOT literal complex-step (kappa -> kappa + i*h): char_function
    // already uses the imaginary unit `i` internally for the Fourier
    // phase factor, and probability() extracts Re[...] from the integrand
    // at every quadrature node before integrating -- Re() is not
    // holomorphic, so reusing `i` for the perturbation would corrupt
    // exactly the real/imaginary split that Re[] depends on (worked
    // through by hand, not discovered by a wrong first attempt -- see
    // dual.hpp). Mathematically equivalent to multicomplex-step (Lantoine,
    // Russell & Dargent 2012): a second, independent differentiation unit
    // per parameter that commutes exactly with Re()/Im() because it's
    // real-linear, sidestepping the issue entirely -- same zero-
    // cancellation, no-tuning-parameter guarantee true complex-step gives
    // for ordinary real functions.
    //
    // Shares char_function_impl (templated on scalar type, see the .cpp)
    // with price() itself -- ONE formula, two instantiations (T=double for
    // price(), T=ComplexDual5 here), not a hand-duplicated second copy
    // that could silently drift from the first (the same reasoning behind
    // extracting brent.hpp). Verified against central finite differences
    // on price() across the same stress regimes price_cos was validated
    // against (short maturity, badly Feller-violating vol-of-vol,
    // dividends); see test_heston.py.
    static HestonJacobian price_jacobian(double spot, double strike, double rate,
                                          double dividend_yield, double maturity, OptionType type,
                                          const HestonParams& hp);

    // price_jacobian, batched across a whole strike grid the same way
    // price_batch batches price() -- and for the same reason. A first cut
    // at calibrate_heston's analytic Jacobian called price_jacobian in a
    // per-strike Python loop and measured it ~3.6x SLOWER than letting
    // scipy fall back to finite differences (97ms vs 27ms on a 13-strike
    // calibration): each price_jacobian call redid its own adaptive
    // quadrature from scratch, giving up exactly the cross-strike
    // characteristic-function sharing price_batch exists for, while ALSO
    // paying ComplexDual5's ~5x wider per-node arithmetic -- fighting the
    // very optimization the finite-difference path got to keep (it calls
    // the already-batched, already-fast heston_price_batch 6 times). This
    // batches the Jacobian computation the same way, so it wins for the
    // reason it should: not because forward-mode AD is inherently fast,
    // but because it can share the expensive part across strikes exactly
    // like the value-only path already does. See test_heston.py for the
    // measured speedup this version actually achieves.
    static std::vector<HestonJacobian> price_jacobian_batch(
        double spot, const std::vector<double>& strikes, const std::vector<OptionType>& types,
        double rate, double dividend_yield, double maturity, const HestonParams& hp,
        int num_nodes = 1500, double phi_max = 150.0);

    // Fang & Oosterlee (2008) COS method: a fixed-node Fourier-cosine
    // series expansion, sharing the SAME characteristic function as
    // price() (its j=2 branch is already the standard risk-neutral CF of
    // ln(S_T), needing no separate re-derivation -- see the .cpp). This
    // is what production Heston engines (including QuantLib's
    // AnalyticHestonEngine) use for speed; benchmarks/test_heston_
    // benchmark.py found price() (adaptive quadrature) ~13x slower than
    // QuantLib specifically because of that fixed-vs-adaptive choice --
    // this method closes nearly all of that gap (measured ~1.2x slower
    // than QuantLib, see the README's "External benchmarks") without
    // giving up price()'s self-terminating accuracy as the in-tree
    // reference. Kept as a SEPARATE method, not a replacement: two
    // independent ways to evaluate the same integral, cross-checked
    // against each other (test_heston.py), matching this project's
    // established pattern (HestonMCPricer vs. this class) of never
    // trusting one Heston implementation alone.
    //
    // The truncation range [a,b] for x=ln(S_T) is derived from the first
    // two cumulants of ln(S_T), computed NUMERICALLY via finite
    // differences on the (already-trusted) characteristic function
    // itself -- not a hand-derived closed-form cumulant formula, which
    // would be a second, unverified place for a transcription error to
    // hide. A SINGLE fixed (range, term-count) pair is not robust across
    // this pricer's full parameter space, though (validated empirically,
    // not assumed -- see the .cpp and CHANGELOG for what broke and why):
    // instead this widens the range and term count together, iteration
    // by iteration, until two successive estimates agree, and falls back
    // to price() itself on the rare parameter combinations where that
    // search doesn't converge within a bounded number of iterations --
    // so the num_terms argument below is a starting point for that
    // search, not the final resolution used. Verified to <0.1% relative
    // error against price() across a 300-case random stress sweep
    // spanning 1-day to 3-year maturities and well-behaved through badly
    // Feller-violating vol-of-vol; see test_heston.py.
    static double price_cos(double spot, double strike, double rate, double dividend_yield,
                             double maturity, OptionType type, const HestonParams& hp,
                             int num_terms = 160);

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
    //
    // Known limitation, quantified in test_heston.py: this scheme has a
    // well-documented discretization bias that grows large specifically
    // when the Feller condition is badly violated (variance keeps hitting
    // its floor between steps). At xi=3.0 against 2*kappa*theta=0.16, 300
    // steps disagrees with the (independently verified) analytic price by
    // ~40 standard errors; the bias shrinks monotonically and is within
    // 1 standard error by ~3000 steps. Don't trust a low-step-count MC
    // price against extreme (Feller-violating) parameters without either
    // increasing num_steps substantially or cross-checking against
    // HestonPricer::price.
    MCResult price(double spot, double strike, double rate, double dividend_yield,
                   double maturity, OptionType type, const HestonParams& hp, long num_paths,
                   int num_steps);

private:
    // See MonteCarloPricer/AmericanPricer: path generation constructs a
    // fresh LOCAL Philox4x64 per path (seeked to a disjoint counter
    // range), parallelized via #pragma omp parallel for -- no shared
    // generator for parallel path threads to race on. cursor_ advances
    // after each price() call so a reused instance draws fresh paths
    // rather than repeating them.
    std::uint64_t seed_;
    std::uint64_t cursor_ = 0;
};

}  // namespace bscpp
