#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace bscpp {

// Brent's method (Van Wijngaarden-Dekker-Brent): combines bisection, the
// secant method, and inverse quadratic interpolation. Given a valid
// bracket [a, b] with f(a) and f(b) of opposite sign, convergence is
// GUARANTEED (worst case degrades to bisection) -- unlike Newton's method,
// it never divides by a near-zero derivative, which is exactly the failure
// mode that makes plain Newton unreliable for implied vol deep ITM/OTM
// (where vega -> 0). Standard reference algorithm (e.g. Brent 1973;
// Numerical Recipes "zbrent"; the same algorithm scipy.optimize.brentq
// implements). Shared between BlackScholes::implied_vol and
// CRRPricer::implied_vol -- both price monotonically in vol, so both can
// use the identical bracketed solver rather than each carrying its own
// copy.
inline double brent_solve(const std::function<double(double)>& f, double a, double b, double tol,
                           int max_iter) {
    double fa = f(a);
    double fb = f(b);
    if (fa * fb > 0.0) {
        return std::numeric_limits<double>::quiet_NaN();  // no sign change; no root to bracket
    }
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    double d = b;  // only meaningful once mflag is false; harmless placeholder before then
    bool mflag = true;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (fb == 0.0 || std::abs(b - a) < tol) {
            return b;
        }

        double s;
        if (fa != fc && fb != fc) {
            // inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        const double lo_bound = std::min((3.0 * a + b) / 4.0, b);
        const double hi_bound = std::max((3.0 * a + b) / 4.0, b);
        const bool cond1 = (s < lo_bound || s > hi_bound);
        const bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        const bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        const bool cond4 = mflag && std::abs(b - c) < tol;
        const bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }

        const double fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    return b;  // best estimate after max_iter; caller's tol wasn't hit but this is close
}

}  // namespace bscpp
