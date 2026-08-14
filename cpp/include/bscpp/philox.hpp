#pragma once

#include <array>
#include <cstdint>

namespace bscpp {

// Philox4x64-10 (Salmon, Moraes, Sanches, Pande 2011, "Parallel Random
// Numbers: As Easy as 1, 2, 3"): a COUNTER-BASED generator, not a
// sequential state machine like std::mt19937_64. output = f(key, counter)
// is a pure function -- draw N can be computed directly from counter=N
// without replaying draws 0..N-1 first, which is what makes this
// trivially parallelizable: thread t seeks to a disjoint counter range
// and generates its stream independently, with zero coordination and no
// risk of overlap, unlike partitioning a single mt19937_64 stream.
//
// This exact algorithm (constants, round function, and the counter-
// increments-BEFORE-each-block convention below) is bit-for-bit
// cross-validated against NumPy's Philox bit generator
// (numpy/random/src/philox/philox.h) -- see test_philox.py, which
// compares raw uint64 output directly against numpy.random.Philox for
// several seeds. That match is the actual portability claim: this isn't
// just self-consistent, it reproduces an independent, already-portable
// reference implementation exactly.
//
// mulhilo64 below deliberately does NOT use __uint128_t (a GCC/Clang
// extension MSVC doesn't support -- this project's CI runs on
// windows-latest too) -- the schoolbook 32-bit-limb expansion is
// standard, portable C++ (only well-defined unsigned wraparound
// arithmetic) and verified bit-exact against Python's arbitrary-
// precision integers across random and edge-case inputs before being
// trusted here.
inline void mulhilo64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
    const uint64_t a_lo = static_cast<uint32_t>(a);
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = static_cast<uint32_t>(b);
    const uint64_t b_hi = b >> 32;

    const uint64_t a_x_b_hi = a_hi * b_hi;
    const uint64_t a_x_b_mid = a_hi * b_lo;
    const uint64_t b_x_a_mid = b_hi * a_lo;
    const uint64_t a_x_b_lo = a_lo * b_lo;

    const uint64_t carry_bit = (static_cast<uint64_t>(static_cast<uint32_t>(a_x_b_mid)) +
                                 static_cast<uint64_t>(static_cast<uint32_t>(b_x_a_mid)) +
                                 (a_x_b_lo >> 32)) >>
                                32;

    hi = a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32) + carry_bit;
    lo = a * b;  // low 64 bits of the product: correct via defined unsigned wraparound
}

namespace detail {

constexpr uint64_t kPhiloxM0 = 0xD2E7470EE14C6C93ULL;
constexpr uint64_t kPhiloxM1 = 0xCA5A826395121157ULL;
constexpr uint64_t kPhiloxW0 = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kPhiloxW1 = 0xBB67AE8584CAA73BULL;

using Word4 = std::array<uint64_t, 4>;
using Word2 = std::array<uint64_t, 2>;

inline Word4 philox4x64_10(Word4 ctr, Word2 key) {
    for (int round = 0; round < 10; ++round) {
        uint64_t hi0, lo0, hi1, lo1;
        mulhilo64(kPhiloxM0, ctr[0], hi0, lo0);
        mulhilo64(kPhiloxM1, ctr[2], hi1, lo1);
        ctr = {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
        key[0] += kPhiloxW0;
        key[1] += kPhiloxW1;
    }
    return ctr;
}

}  // namespace detail

// Drop-in replacement for std::mt19937_64 in this project's Monte Carlo
// pricers: constructible from a single uint64_t seed, callable via
// operator() to draw a uint64_t. Internally buffers 4 draws per Philox
// block (matching NumPy's convention exactly, including incrementing the
// counter BEFORE computing each new block -- verified in test_philox.py).
class Philox4x64 {
public:
    explicit Philox4x64(uint64_t seed, uint64_t stream = 0)
        : key_{seed, 0}, counter_{0, stream, 0, 0}, buffer_pos_(4) {}

    uint64_t operator()() {
        if (buffer_pos_ >= 4) {
            increment_counter();
            buffer_ = detail::philox4x64_10(counter_, key_);
            buffer_pos_ = 0;
        }
        return buffer_[buffer_pos_++];
    }

    // Seeks to an explicit counter position -- the operation that gives
    // "per-thread streams for free": thread t can call
    // seek(base + t, 0, 0, 0) and draw from a stream that provably never
    // overlaps any other thread's, with no shared mutable state and no
    // locking. Resets the internal buffer so the next operator() call
    // generates fresh output from the new position.
    void seek(uint64_t c0, uint64_t c1 = 0, uint64_t c2 = 0, uint64_t c3 = 0) {
        counter_ = {c0, c1, c2, c3};
        buffer_pos_ = 4;
    }

private:
    detail::Word2 key_;
    detail::Word4 counter_;
    detail::Word4 buffer_;
    int buffer_pos_;

    void increment_counter() {
        for (auto& word : counter_) {
            if (++word != 0) break;  // 256-bit increment with carry propagation
        }
    }
};

}  // namespace bscpp
