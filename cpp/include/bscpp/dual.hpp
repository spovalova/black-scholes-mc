#pragma once

#include <array>
#include <cmath>
#include <complex>

namespace bscpp {

using cdouble = std::complex<double>;

// Number of Heston parameters this file's dual numbers differentiate
// against, in a fixed order: kappa, theta, xi, rho, v0.
constexpr int kHestonNumParams = 5;

// Forward-mode dual number carrying a COMPLEX value and its derivatives
// w.r.t. up to kHestonNumParams independent real parameters -- built
// specifically to differentiate HestonPricer::char_function (see
// heston.cpp) without literal complex-step (perturbing a parameter with
// the SAME imaginary unit `i` the characteristic function already uses
// internally for its Fourier phase factor).
//
// That reuse would be wrong, not just imprecise: HestonPricer::probability
// extracts Re[...] from the integrand at EVERY quadrature node before
// integrating, and Re() is not a holomorphic operation -- it does not
// commute with "evaluate the same complex formula at a complex-perturbed
// parameter" the way literal complex-step requires. Perturbing kappa with
// i*h would scramble exactly the real/imaginary split Re[] depends on,
// silently corrupting the derivative it's supposed to extract (worked
// through by hand before writing any code here, not discovered by trial
// and error -- see CHANGELOG for the derivation).
//
// The fix is a SECOND, independent differentiation "unit" that doesn't
// interact with the first: instead of reusing `i`, track each parameter's
// derivative explicitly in its own array slot. Re()/Im() then commute
// with differentiation exactly, because they're real-linear projections,
// not because of any special-casing. This is mathematically equivalent to
// "multicomplex-step" / multicomplex differentiation (Lantoine, Russell &
// Dargent 2012) with one extra unit per parameter, just implemented
// directly as forward-mode dual numbers rather than a literal second
// imaginary axis -- same zero-cancellation, no-finite-difference-error
// guarantee complex-step gives for ordinary real functions, generalized to
// survive the Re() step this pricer's math requires.
struct ComplexDual5 {
    cdouble value{};
    std::array<cdouble, kHestonNumParams> deriv{};

    ComplexDual5() = default;
    ComplexDual5(cdouble v) : value(v) {}                       // NOLINT(runtime/explicit)
    ComplexDual5(double v) : value(v) {}                        // NOLINT(runtime/explicit)

    // A "seeded" independent variable: value v, derivative 1 w.r.t.
    // parameter index `param` (0=kappa,1=theta,2=xi,3=rho,4=v0), 0 w.r.t.
    // every other -- marks exactly one Heston parameter as the thing
    // being differentiated. Seeding all 5 parameters this way and running
    // ONE pass through char_function computes the FULL Jacobian at once
    // (vector forward-mode AD), not 5 separate passes.
    static ComplexDual5 variable(double v, int param) {
        ComplexDual5 d(v);
        d.deriv[static_cast<size_t>(param)] = 1.0;
        return d;
    }
};

inline ComplexDual5 operator+(const ComplexDual5& a, const ComplexDual5& b) {
    ComplexDual5 r;
    r.value = a.value + b.value;
    for (int k = 0; k < kHestonNumParams; ++k) r.deriv[static_cast<size_t>(k)] =
        a.deriv[static_cast<size_t>(k)] + b.deriv[static_cast<size_t>(k)];
    return r;
}
inline ComplexDual5 operator-(const ComplexDual5& a, const ComplexDual5& b) {
    ComplexDual5 r;
    r.value = a.value - b.value;
    for (int k = 0; k < kHestonNumParams; ++k) r.deriv[static_cast<size_t>(k)] =
        a.deriv[static_cast<size_t>(k)] - b.deriv[static_cast<size_t>(k)];
    return r;
}
inline ComplexDual5 operator-(const ComplexDual5& a) {
    ComplexDual5 r;
    r.value = -a.value;
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = -a.deriv[static_cast<size_t>(k)];
    return r;
}
// Product rule: d(a*b) = da*b + a*db.
inline ComplexDual5 operator*(const ComplexDual5& a, const ComplexDual5& b) {
    ComplexDual5 r;
    r.value = a.value * b.value;
    for (int k = 0; k < kHestonNumParams; ++k) {
        const size_t sk = static_cast<size_t>(k);
        r.deriv[sk] = a.deriv[sk] * b.value + a.value * b.deriv[sk];
    }
    return r;
}
// Quotient rule: d(a/b) = (da*b - a*db) / b^2.
inline ComplexDual5 operator/(const ComplexDual5& a, const ComplexDual5& b) {
    ComplexDual5 r;
    r.value = a.value / b.value;
    const cdouble b2 = b.value * b.value;
    for (int k = 0; k < kHestonNumParams; ++k) {
        const size_t sk = static_cast<size_t>(k);
        r.deriv[sk] = (a.deriv[sk] * b.value - a.value * b.deriv[sk]) / b2;
    }
    return r;
}

// sqrt/exp/log: found via `using std::sqrt;`-style unqualified calls in
// heston.cpp's templated char_function_impl, so ordinary two-phase name
// lookup + ADL picks these overloads for ComplexDual5 arguments and
// std::sqrt/exp/log (unchanged) for plain cdouble/double ones -- the same
// customization-point idiom std::swap relies on.
inline ComplexDual5 sqrt(const ComplexDual5& x) {
    ComplexDual5 r;
    r.value = std::sqrt(x.value);
    const cdouble two_sqrt = 2.0 * r.value;
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = x.deriv[static_cast<size_t>(k)] / two_sqrt;
    return r;
}
inline ComplexDual5 exp(const ComplexDual5& x) {
    ComplexDual5 r;
    r.value = std::exp(x.value);
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = x.deriv[static_cast<size_t>(k)] * r.value;
    return r;
}
inline ComplexDual5 log(const ComplexDual5& x) {
    ComplexDual5 r;
    r.value = std::log(x.value);
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = x.deriv[static_cast<size_t>(k)] / x.value;
    return r;
}

// Real-valued counterpart: what ComplexDual5::real() (below) produces,
// and what the (real-valued, by construction) P1/P2 quadrature integrand
// and adaptive-Simpson running sum are built from.
struct RealDual5 {
    double value = 0.0;
    std::array<double, kHestonNumParams> deriv{};

    RealDual5() = default;
    RealDual5(double v) : value(v) {}                           // NOLINT(runtime/explicit)
};

inline RealDual5 operator+(const RealDual5& a, const RealDual5& b) {
    RealDual5 r;
    r.value = a.value + b.value;
    for (int k = 0; k < kHestonNumParams; ++k) r.deriv[static_cast<size_t>(k)] =
        a.deriv[static_cast<size_t>(k)] + b.deriv[static_cast<size_t>(k)];
    return r;
}
inline RealDual5 operator-(const RealDual5& a, const RealDual5& b) {
    RealDual5 r;
    r.value = a.value - b.value;
    for (int k = 0; k < kHestonNumParams; ++k) r.deriv[static_cast<size_t>(k)] =
        a.deriv[static_cast<size_t>(k)] - b.deriv[static_cast<size_t>(k)];
    return r;
}
inline RealDual5 operator*(const RealDual5& a, double s) {
    RealDual5 r;
    r.value = a.value * s;
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = a.deriv[static_cast<size_t>(k)] * s;
    return r;
}
inline RealDual5 operator*(double s, const RealDual5& a) { return a * s; }
inline RealDual5 operator/(const RealDual5& a, double s) {
    RealDual5 r;
    r.value = a.value / s;
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = a.deriv[static_cast<size_t>(k)] / s;
    return r;
}

// The one non-holomorphic operation this file exists to support: Re() is
// real-linear, so this is exact -- no derivative information is lost or
// scrambled the way it would be under literal (shared-`i`) complex-step.
inline RealDual5 real_part(const ComplexDual5& z) {
    RealDual5 r;
    r.value = z.value.real();
    for (int k = 0; k < kHestonNumParams; ++k)
        r.deriv[static_cast<size_t>(k)] = z.deriv[static_cast<size_t>(k)].real();
    return r;
}

// Magnitude of just the VALUE component -- used by the templated adaptive
// quadrature's error-control/stopping checks (heston.cpp), which must
// converge on how well the INTEGRAL'S VALUE is resolved, not its
// derivatives (comparing derivatives too would conflate two different
// notions of "converged" and has no natural single scalar to check
// against tol anyway).
inline double abs_value(double x) { return std::abs(x); }
inline double abs_value(const RealDual5& x) { return std::abs(x.value); }

}  // namespace bscpp
