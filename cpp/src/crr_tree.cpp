#include "bscpp/crr_tree.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "bscpp/brent.hpp"

namespace bscpp {

double CRRPricer::price(double spot, double strike, double rate, double dividend_yield,
                         double maturity, OptionType type, double vol, int num_steps) {
    const double dt = maturity / num_steps;
    const double disc = std::exp(-rate * dt);
    const double u = std::exp(vol * std::sqrt(dt));
    const double d = 1.0 / u;
    // Risk-neutral up-probability. Clamped to [0,1]: a badly-chosen
    // (vol, num_steps, maturity) combination can push this outside the
    // valid range at very coarse dt (see the header's note) -- clamping
    // keeps the tree well-defined rather than silently producing a
    // negative "probability" that corrupts every node above it, and the
    // clamp itself is the honest signal to increase num_steps.
    double p = (std::exp((rate - dividend_yield) * dt) - d) / (u - d);
    p = std::clamp(p, 0.0, 1.0);

    const auto payoff = [&](double s) {
        return type == OptionType::Call ? std::max(s - strike, 0.0) : std::max(strike - s, 0.0);
    };

    // u^j and d^j, precomputed once: node (i, j) has spot spot*u_pow[j]*
    // d_pow[i-j], and every std::pow above is replaced by a table lookup
    // -- O(N) transcendental calls total instead of O(N^2).
    std::vector<double> u_pow(num_steps + 1), d_pow(num_steps + 1);
    u_pow[0] = d_pow[0] = 1.0;
    for (int k = 1; k <= num_steps; ++k) {
        u_pow[k] = u_pow[k - 1] * u;
        d_pow[k] = d_pow[k - 1] * d;
    }

    // Terminal payoffs at maturity: node j (0..num_steps) has j up-moves,
    // (num_steps - j) down-moves, so S_T = spot * u^j * d^(num_steps - j).
    std::vector<double> values(num_steps + 1);
    for (int j = 0; j <= num_steps; ++j) {
        values[j] = payoff(spot * u_pow[j] * d_pow[num_steps - j]);
    }

    // Backward induction: at step i (num_steps-1 down to 0), node j (0..i)
    // has spot spot*u_pow[j]*d_pow[i-j]. values[] is reused in place --
    // safe because computing values[j] at step i only reads values[j] and
    // values[j+1], both still holding step (i+1)'s values at that point
    // (processing j in increasing order never overwrites an index a later
    // j' in this same pass still needs: j' reads values[j'] and
    // values[j'+1], both > the index just written).
    for (int i = num_steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            const double continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            const double s = spot * u_pow[j] * d_pow[i - j];
            values[j] = std::max(continuation, payoff(s));  // early exercise
        }
    }
    return values[0];
}

double CRRPricer::implied_vol(double spot, double strike, double rate, double dividend_yield,
                               double maturity, OptionType type, double market_price,
                               int num_steps, double tol, int max_iter) {
    const auto residual = [&](double vol) {
        return price(spot, strike, rate, dividend_yield, maturity, type, vol, num_steps) -
               market_price;
    };
    return brent_solve(residual, 1e-6, 5.0, tol, max_iter);
}

}  // namespace bscpp
