#pragma once

#include <cmath>
#include <random>

namespace bscpp {

// std::normal_distribution's exact algorithm is NOT specified by the C++
// standard -- libstdc++ and libc++ implement genuinely different
// transforms, so the same seed can produce different Monte Carlo paths on
// different platforms even though std::mt19937_64 itself is bit-for-bit
// portable (the standard fully specifies the generator's raw output
// stream, just not what any std::*_distribution does with it). A project
// whose CI runs across 3 OSes and whose whole premise is verified,
// reproducible numerics needs the seeded RNG path to actually BE
// reproducible, not merely pass because test tolerances are loose enough
// to hide the divergence.
//
// This hand-rolled Box-Muller transform is built only from
// std::mt19937_64::operator()'s raw output and portable arithmetic (log,
// sqrt, cos) -- no std::uniform_real_distribution or
// std::normal_distribution anywhere in the chain -- so the exact sequence
// of normal variates for a given seed is identical across any conforming
// C++ implementation, not just "usually similar in distribution."
inline double standard_normal(std::mt19937_64& rng) {
    // 53-bit uniform-in-(0,1]: mt19937_64's 64-bit output carries more
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
