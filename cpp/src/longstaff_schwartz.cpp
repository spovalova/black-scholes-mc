#include "bscpp/longstaff_schwartz.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "bscpp/portable_normal.hpp"

namespace bscpp {

namespace {

// Solves A*x = b via Gaussian elimination with partial pivoting.
// Returns false (leaving x untouched) if A is numerically singular --
// callers must treat that as "not enough information, skip this update"
// rather than crash.
bool solve_linear_system(std::vector<std::vector<double>> a, std::vector<double> b,
                          std::vector<double>& x) {
    const int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        double best = std::abs(a[col][col]);
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > best) {
                best = std::abs(a[row][col]);
                pivot_row = row;
            }
        }
        if (best < 1e-12) {
            return false;  // singular / ill-conditioned
        }
        std::swap(a[col], a[pivot_row]);
        std::swap(b[col], b[pivot_row]);

        for (int row = col + 1; row < n; ++row) {
            const double factor = a[row][col] / a[col][col];
            for (int k = col; k < n; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    x.assign(n, 0.0);
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < n; ++k) {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    return true;
}

std::vector<double> basis_at(double x, int k) {
    std::vector<double> basis(k);
    basis[0] = 1.0;
    for (int j = 1; j < k; ++j) {
        basis[j] = basis[j - 1] * x;
    }
    return basis;
}

std::vector<std::vector<double>> simulate_paths(Philox4x64& rng, const MarketInputs& in,
                                                 long num_paths, int num_steps, double drift,
                                                 double diffusion) {
    std::vector<std::vector<double>> paths(
        static_cast<size_t>(num_paths), std::vector<double>(static_cast<size_t>(num_steps) + 1));
    for (long p = 0; p < num_paths; ++p) {
        paths[p][0] = in.spot;
        for (int t = 1; t <= num_steps; ++t) {
            const double z = standard_normal(rng);
            paths[p][t] = paths[p][t - 1] * std::exp(drift + diffusion * z);
        }
    }
    return paths;
}

// Backward-induction calibration pass: fits one regression (if enough ITM
// points) per exercise date on `paths`, returning the fitted coefficients
// per step. Does NOT decide anything about a separate pricing path set --
// that happens in a forward pass using these coefficients as a fixed policy.
std::vector<std::optional<std::vector<double>>> calibrate(
    const std::vector<std::vector<double>>& paths, const MarketInputs& in, int num_steps, int k,
    double discount_dt) {
    const long num_paths = static_cast<long>(paths.size());

    std::vector<double> cashflow(static_cast<size_t>(num_paths));
    std::vector<int> exercise_step(static_cast<size_t>(num_paths), num_steps);
    for (long p = 0; p < num_paths; ++p) {
        cashflow[p] = AmericanPricer::payoff(paths[p][num_steps], in.strike, in.type);
    }

    std::vector<std::optional<std::vector<double>>> betas(static_cast<size_t>(num_steps) + 1);

    for (int t = num_steps - 1; t >= 1; --t) {
        std::vector<long> itm;
        itm.reserve(static_cast<size_t>(num_paths));
        for (long p = 0; p < num_paths; ++p) {
            if (AmericanPricer::payoff(paths[p][t], in.strike, in.type) > 0.0) {
                itm.push_back(p);
            }
        }
        if (static_cast<int>(itm.size()) < 2 * k) {
            continue;  // not enough ITM points to trust a regression here
        }

        std::vector<std::vector<double>> xtx(k, std::vector<double>(k, 0.0));
        std::vector<double> xty(k, 0.0);

        for (long p : itm) {
            const double x = paths[p][t] / in.strike;
            const std::vector<double> basis = basis_at(x, k);
            const double disc_factor = std::pow(discount_dt, exercise_step[p] - t);
            const double y = cashflow[p] * disc_factor;

            for (int a = 0; a < k; ++a) {
                xty[a] += basis[a] * y;
                for (int b = 0; b < k; ++b) {
                    xtx[a][b] += basis[a] * basis[b];
                }
            }
        }

        std::vector<double> beta;
        if (!solve_linear_system(xtx, xty, beta)) {
            continue;  // ill-conditioned this step; leave policy unset (hold)
        }
        betas[t] = beta;

        // Update THIS calibration set's own cashflows/exercise_step so the
        // next (earlier) step's regression targets reflect the exercise
        // policy just fitted -- standard LSM backward induction. This is
        // purely internal bookkeeping for the calibration set; it never
        // touches the separate pricing path set.
        for (long p : itm) {
            const double x = paths[p][t] / in.strike;
            const std::vector<double> basis = basis_at(x, k);
            double continuation_est = 0.0;
            for (int j = 0; j < k; ++j) {
                continuation_est += beta[j] * basis[j];
            }
            const double exercise_val = AmericanPricer::payoff(paths[p][t], in.strike, in.type);
            if (exercise_val > continuation_est) {
                cashflow[p] = exercise_val;
                exercise_step[p] = t;
            }
        }
    }

    return betas;
}

}  // namespace

AmericanPricer::AmericanPricer(std::uint64_t seed)
    // Philox's stream parameter gives a principled, provably-non-
    // overlapping second stream from the same seed (see philox.hpp) --
    // replaces the previous arbitrary-magic-number seed offset
    // (seed + 1768237423ULL), which only relied on that offset being
    // "big enough" to avoid overlap in mt19937_64's sequential state.
    : rng_(seed, 0), rng_calibration_(seed, 1) {}

double AmericanPricer::payoff(double s, double strike, OptionType type) {
    if (type == OptionType::Call) {
        return std::max(s - strike, 0.0);
    }
    return std::max(strike - s, 0.0);
}

MCResult AmericanPricer::price(const MarketInputs& in, long num_paths, int num_steps,
                                int poly_degree, long num_calibration_paths) {
    if (num_calibration_paths <= 0) {
        num_calibration_paths = num_paths;
    }
    const int k = poly_degree + 1;
    const double dt = in.maturity / num_steps;
    const double drift = (in.rate - in.dividend_yield - 0.5 * in.vol * in.vol) * dt;
    const double diffusion = in.vol * std::sqrt(dt);
    const double discount_dt = std::exp(-in.rate * dt);

    // Phase 1: fit the exercise policy on an independently-seeded
    // calibration path set.
    const auto calibration_paths =
        simulate_paths(rng_calibration_, in, num_calibration_paths, num_steps, drift, diffusion);
    const auto betas = calibrate(calibration_paths, in, num_steps, k, discount_dt);

    // Phase 2: apply that FIXED policy forward along a separate pricing
    // path set -- no regression happens here, only exercise decisions.
    const auto pricing_paths = simulate_paths(rng_, in, num_paths, num_steps, drift, diffusion);

    double sum = 0.0;
    double sum_sq = 0.0;
    for (long p = 0; p < num_paths; ++p) {
        double cashflow = 0.0;
        int exercise_step = num_steps;
        bool exercised = false;

        for (int t = 1; t < num_steps && !exercised; ++t) {
            if (!betas[t].has_value()) {
                continue;  // no fitted policy at this step; hold
            }
            const double exercise_val = payoff(pricing_paths[p][t], in.strike, in.type);
            if (exercise_val <= 0.0) {
                continue;
            }
            const double x = pricing_paths[p][t] / in.strike;
            const std::vector<double> basis = basis_at(x, k);
            double continuation_est = 0.0;
            const std::vector<double>& beta = *betas[t];
            for (int j = 0; j < k; ++j) {
                continuation_est += beta[j] * basis[j];
            }
            if (exercise_val > continuation_est) {
                cashflow = exercise_val;
                exercise_step = t;
                exercised = true;
            }
        }

        if (!exercised) {
            cashflow = payoff(pricing_paths[p][num_steps], in.strike, in.type);
            exercise_step = num_steps;
        }

        const double v = cashflow * std::pow(discount_dt, exercise_step);
        sum += v;
        sum_sq += v * v;
    }

    const double mean = sum / static_cast<double>(num_paths);
    double variance = sum_sq / static_cast<double>(num_paths) - mean * mean;
    variance = std::max(variance, 0.0);
    const double std_error = std::sqrt(variance / static_cast<double>(num_paths));

    return {mean, std_error};
}

}  // namespace bscpp
