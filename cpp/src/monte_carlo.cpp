#include "bscpp/monte_carlo.hpp"

#include <algorithm>
#include <cmath>

namespace bscpp {

MonteCarloPricer::MonteCarloPricer(std::uint64_t seed) : rng_(seed) {}

std::vector<double> MonteCarloPricer::generate_normals(long n) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> z(static_cast<size_t>(n));
    for (long i = 0; i < n; ++i) {
        z[static_cast<size_t>(i)] = dist(rng_);
    }
    return z;
}

double MonteCarloPricer::payoff(double s_t, double strike, OptionType type) {
    if (type == OptionType::Call) {
        return std::max(s_t - strike, 0.0);
    }
    return std::max(strike - s_t, 0.0);
}

MCResult MonteCarloPricer::price_with_z(const MarketInputs& in, const std::vector<double>& z,
                                         bool antithetic) {
    const double drift = (in.rate - in.dividend_yield - 0.5 * in.vol * in.vol) * in.maturity;
    const double diffusion = in.vol * std::sqrt(in.maturity);
    const double discount = std::exp(-in.rate * in.maturity);

    const long n_draws = static_cast<long>(z.size());

    // Antithetic pairs are NEGATIVELY correlated by construction -- that is
    // the entire point of the technique -- so the 2N individual samples are
    // NOT i.i.d. and pooling them into the plain i.i.d. variance formula
    // misstates the standard error (measured ~32% overstated on an ATM
    // call). The correct estimator treats each antithetic PAIR MEAN as one
    // i.i.d. observation: mean and variance are computed over the N pair
    // means, and std_error = sqrt(var_pairs / N). Verified empirically in
    // tests/test_pricing.py::test_antithetic_std_error_is_calibrated
    // (reported std_error vs. realized dispersion across independent seeds).
    double sum = 0.0;
    double sum_sq = 0.0;
    for (long i = 0; i < n_draws; ++i) {
        const double zi = z[static_cast<size_t>(i)];
        const double s_t = in.spot * std::exp(drift + diffusion * zi);
        const double p = payoff(s_t, in.strike, in.type) * discount;

        double obs = p;  // one i.i.d. observation: the sample (or the pair mean)
        if (antithetic) {
            const double s_t2 = in.spot * std::exp(drift - diffusion * zi);
            const double p2 = payoff(s_t2, in.strike, in.type) * discount;
            obs = 0.5 * (p + p2);
        }
        sum += obs;
        sum_sq += obs * obs;
    }

    const double n_obs = static_cast<double>(n_draws);
    const double mean = sum / n_obs;
    double variance = sum_sq / n_obs - mean * mean;
    variance = std::max(variance, 0.0);
    const double std_error = std::sqrt(variance / n_obs);

    return {mean, std_error};
}

MCResult MonteCarloPricer::price_european(const MarketInputs& in, long num_paths,
                                           bool antithetic) {
    const long n_draws = antithetic ? (num_paths + 1) / 2 : num_paths;
    const auto z = generate_normals(n_draws);
    return price_with_z(in, z, antithetic);
}

Greeks MonteCarloPricer::greeks_european(const MarketInputs& in, long num_paths, bool antithetic,
                                          double bump_frac) {
    const long n_draws = antithetic ? (num_paths + 1) / 2 : num_paths;
    const auto z = generate_normals(n_draws);  // shared draws => common random numbers

    auto price_at = [&](const MarketInputs& bumped) {
        return price_with_z(bumped, z, antithetic).price;
    };

    Greeks g;

    const double h_s = in.spot * bump_frac;
    MarketInputs up = in, down = in;
    up.spot += h_s;
    down.spot -= h_s;
    const double v_up_s = price_at(up);
    const double v_down_s = price_at(down);
    const double v0 = price_at(in);
    g.delta = (v_up_s - v_down_s) / (2.0 * h_s);
    g.gamma = (v_up_s - 2.0 * v0 + v_down_s) / (h_s * h_s);

    const double h_sigma = std::max(in.vol * bump_frac, 1e-4);
    up = in;
    down = in;
    up.vol += h_sigma;
    down.vol = std::max(in.vol - h_sigma, 1e-6);
    g.vega = (price_at(up) - price_at(down)) / (up.vol - down.vol);

    const double h_r = std::max(std::abs(in.rate) * bump_frac, 1e-4);
    up = in;
    down = in;
    up.rate += h_r;
    down.rate -= h_r;
    g.rho = (price_at(up) - price_at(down)) / (2.0 * h_r);

    const double h_t = std::min(in.maturity * bump_frac, in.maturity * 0.5);
    const double t_down = std::max(in.maturity - h_t, 1e-6);
    const double t_up = in.maturity + h_t;
    up = in;
    up.maturity = t_up;
    down = in;
    down.maturity = t_down;
    // -dV/dT (time-to-maturity shrinking) == dV/dt (calendar time decay).
    g.theta = (price_at(down) - price_at(up)) / (t_up - t_down);

    return g;
}

}  // namespace bscpp
