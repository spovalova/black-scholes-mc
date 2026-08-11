#include "bscpp/longstaff_schwartz.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

}  // namespace

AmericanPricer::AmericanPricer(std::uint64_t seed) : rng_(seed) {}

double AmericanPricer::payoff(double s, double strike, OptionType type) {
    if (type == OptionType::Call) {
        return std::max(s - strike, 0.0);
    }
    return std::max(strike - s, 0.0);
}

MCResult AmericanPricer::price(const MarketInputs& in, long num_paths, int num_steps,
                                int poly_degree) {
    const double dt = in.maturity / num_steps;
    const double drift = (in.rate - in.dividend_yield - 0.5 * in.vol * in.vol) * dt;
    const double diffusion = in.vol * std::sqrt(dt);
    const double discount_dt = std::exp(-in.rate * dt);

    std::normal_distribution<double> normal(0.0, 1.0);

    // paths[p][t] = simulated spot at step t, t = 0..num_steps
    std::vector<std::vector<double>> paths(
        static_cast<size_t>(num_paths), std::vector<double>(static_cast<size_t>(num_steps) + 1));
    for (long p = 0; p < num_paths; ++p) {
        paths[p][0] = in.spot;
        for (int t = 1; t <= num_steps; ++t) {
            const double z = normal(rng_);
            paths[p][t] = paths[p][t - 1] * std::exp(drift + diffusion * z);
        }
    }

    std::vector<double> cashflow(static_cast<size_t>(num_paths));
    std::vector<int> exercise_step(static_cast<size_t>(num_paths), num_steps);
    for (long p = 0; p < num_paths; ++p) {
        cashflow[p] = payoff(paths[p][num_steps], in.strike, in.type);
    }

    const int k = poly_degree + 1;  // number of basis functions

    for (int t = num_steps - 1; t >= 1; --t) {
        std::vector<long> itm;
        itm.reserve(static_cast<size_t>(num_paths));
        for (long p = 0; p < num_paths; ++p) {
            if (payoff(paths[p][t], in.strike, in.type) > 0.0) {
                itm.push_back(p);
            }
        }
        // Need meaningfully more data points than regression parameters,
        // otherwise the fit is unreliable -- just hold this step.
        if (static_cast<int>(itm.size()) < 2 * k) {
            continue;
        }

        std::vector<std::vector<double>> xtx(k, std::vector<double>(k, 0.0));
        std::vector<double> xty(k, 0.0);
        std::vector<std::vector<double>> basis_cache(itm.size(), std::vector<double>(k));

        for (size_t i = 0; i < itm.size(); ++i) {
            const long p = itm[i];
            const double x = paths[p][t] / in.strike;  // normalize for conditioning
            std::vector<double>& basis = basis_cache[i];
            basis[0] = 1.0;
            for (int j = 1; j < k; ++j) {
                basis[j] = basis[j - 1] * x;
            }
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
            continue;  // ill-conditioned this step; skip exercise update
        }

        for (size_t i = 0; i < itm.size(); ++i) {
            const long p = itm[i];
            const std::vector<double>& basis = basis_cache[i];
            double continuation_est = 0.0;
            for (int j = 0; j < k; ++j) {
                continuation_est += beta[j] * basis[j];
            }
            const double exercise_val = payoff(paths[p][t], in.strike, in.type);
            if (exercise_val > continuation_est) {
                cashflow[p] = exercise_val;
                exercise_step[p] = t;
            }
        }
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (long p = 0; p < num_paths; ++p) {
        const double disc_factor = std::pow(discount_dt, exercise_step[p]);
        const double v = cashflow[p] * disc_factor;
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
