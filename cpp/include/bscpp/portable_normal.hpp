#pragma once

#include <cmath>

#include "bscpp/philox.hpp"

namespace bscpp {

// Box-Muller transform built only from Philox4x64::operator()'s raw
// output (see philox.hpp -- bit-for-bit cross-validated against NumPy's
// Philox, not just internally self-consistent) and portable arithmetic
// (log, sqrt, cos). No std::uniform_real_distribution or
// std::normal_distribution anywhere in the chain -- both are
// implementation-defined by the C++ standard (libstdc++ and libc++
// implement genuinely different transforms), so the same seed could
// otherwise produce different Monte Carlo paths on different platforms
// even with a bit-for-bit portable underlying generator. A project whose
// CI runs across 3 OSes and whose whole premise is verified, reproducible
// numerics needs the seeded RNG path to actually BE reproducible, not
// merely pass because test tolerances are loose enough to hide the
// divergence.
inline double standard_normal(Philox4x64& rng) {
    // 53-bit uniform-in-(0,1]: Philox4x64's 64-bit output carries more
    // entropy than a double's 53-bit mantissa, so this keeps the top 53
    // bits and normalizes by 2^53. u1 must be strictly positive (log(0)
    // is undefined); the raw draw can land on exactly 0, so reject and
    // redraw rather than clamp (clamping would bias the tail).
    constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^-53
    double u1;
    do {
        u1 = static_cast<double>(rng() >> 11) * kScale;
    } while (u1 <= 0.0);
    const double u2 = static_cast<double>(rng() >> 11) * kScale;
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

}  // namespace bscpp
