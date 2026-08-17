#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bscpp/black_scholes.hpp"
#include "bscpp/crr_tree.hpp"
#include "bscpp/heston.hpp"
#include "bscpp/longstaff_schwartz.hpp"
#include "bscpp/monte_carlo.hpp"
#include "bscpp/philox.hpp"
#include "bscpp/types.hpp"

namespace py = pybind11;
using namespace bscpp;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ Black-Scholes analytic pricer and Monte Carlo pricer, exposed to Python.";

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put);

    py::class_<Greeks>(m, "Greeks")
        .def(py::init<>())
        .def_readwrite("delta", &Greeks::delta)
        .def_readwrite("gamma", &Greeks::gamma)
        .def_readwrite("vega", &Greeks::vega)
        .def_readwrite("theta", &Greeks::theta)
        .def_readwrite("rho", &Greeks::rho)
        .def("__repr__", [](const Greeks& g) {
            return "Greeks(delta=" + std::to_string(g.delta) +
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) +
                   ", theta=" + std::to_string(g.theta) +
                   ", rho=" + std::to_string(g.rho) + ")";
        });

    py::class_<PricingResult>(m, "PricingResult")
        .def(py::init<>())
        .def_readwrite("price", &PricingResult::price)
        .def_readwrite("greeks", &PricingResult::greeks);

    py::class_<MCResult>(m, "MCResult")
        .def(py::init<>())
        .def_readwrite("price", &MCResult::price)
        .def_readwrite("std_error", &MCResult::std_error)
        .def("__repr__", [](const MCResult& r) {
            return "MCResult(price=" + std::to_string(r.price) +
                   ", std_error=" + std::to_string(r.std_error) + ")";
        });

    py::class_<MarketInputs>(m, "MarketInputs")
        .def(py::init([](double spot, double strike, double rate, double dividend_yield,
                          double vol, double maturity, OptionType type) {
                 MarketInputs in;
                 in.spot = spot;
                 in.strike = strike;
                 in.rate = rate;
                 in.dividend_yield = dividend_yield;
                 in.vol = vol;
                 in.maturity = maturity;
                 in.type = type;
                 return in;
             }),
             py::arg("spot"), py::arg("strike"), py::arg("rate"), py::arg("dividend_yield") = 0.0,
             py::arg("vol"), py::arg("maturity"), py::arg("type") = OptionType::Call)
        .def_readwrite("spot", &MarketInputs::spot)
        .def_readwrite("strike", &MarketInputs::strike)
        .def_readwrite("rate", &MarketInputs::rate)
        .def_readwrite("dividend_yield", &MarketInputs::dividend_yield)
        .def_readwrite("vol", &MarketInputs::vol)
        .def_readwrite("maturity", &MarketInputs::maturity)
        .def_readwrite("type", &MarketInputs::type);

    m.def("bs_price", &BlackScholes::price, py::arg("inputs"));
    m.def("bs_greeks", &BlackScholes::greeks, py::arg("inputs"));
    m.def("bs_price_with_greeks", &BlackScholes::price_with_greeks, py::arg("inputs"));
    m.def("bs_price_with_greeks_batch", &BlackScholes::price_with_greeks_batch, py::arg("inputs"),
          py::call_guard<py::gil_scoped_release>(),
          "Price + Greeks for a list of MarketInputs in one C++ call (avoids per-contract "
          "Python<->C++ crossing overhead when pricing a whole chain).");
    m.def("bs_implied_vol", &BlackScholes::implied_vol, py::arg("inputs"), py::arg("market_price"),
          py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100, py::arg("tol") = 1e-8);
    m.def("bs_implied_vol_batch", &BlackScholes::implied_vol_batch, py::arg("inputs"),
          py::arg("market_prices"), py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100,
          py::arg("tol") = 1e-8, py::call_guard<py::gil_scoped_release>());

    // --- NumPy-native batch variants ---
    // bs_price_with_greeks_batch/bs_implied_vol_batch above take a Python
    // list of MarketInputs OBJECTS: pybind11/stl.h converts that list by
    // constructing N individual Python MarketInputs instances (each a
    // real object allocation + refcounting) before the batch call even
    // starts -- the exact per-contract Python<->C++ crossing overhead
    // batching was supposed to eliminate, just moved one step earlier.
    // These variants take struct-of-arrays NumPy arrays directly (the
    // layout chain data is already in, via chain["strike"].to_numpy()
    // etc.) and return struct-of-arrays too, with zero Python object
    // construction per contract on either side -- `unchecked` accessors
    // are raw-buffer access, safe to use with the GIL released (no
    // Python API calls happen inside the loop), so the whole computation
    // runs GIL-free.
    m.def("bs_price_with_greeks_batch_arrays",
          [](py::array_t<double> spot, py::array_t<double> strike, py::array_t<double> rate,
             py::array_t<double> dividend_yield, py::array_t<double> vol,
             py::array_t<double> maturity, py::array_t<int> type) {
              const py::ssize_t n = spot.size();
              if (strike.size() != n || rate.size() != n || dividend_yield.size() != n ||
                  vol.size() != n || maturity.size() != n || type.size() != n) {
                  throw std::invalid_argument("all input arrays must be the same length");
              }
              auto s = spot.unchecked<1>();
              auto k = strike.unchecked<1>();
              auto r = rate.unchecked<1>();
              auto q = dividend_yield.unchecked<1>();
              auto v = vol.unchecked<1>();
              auto t = maturity.unchecked<1>();
              auto ty = type.unchecked<1>();

              py::array_t<double> price(n), delta(n), gamma(n), vega(n), theta(n), rho(n);
              auto price_m = price.mutable_unchecked<1>();
              auto delta_m = delta.mutable_unchecked<1>();
              auto gamma_m = gamma.mutable_unchecked<1>();
              auto vega_m = vega.mutable_unchecked<1>();
              auto theta_m = theta.mutable_unchecked<1>();
              auto rho_m = rho.mutable_unchecked<1>();

              {
                  py::gil_scoped_release release;
                  for (py::ssize_t i = 0; i < n; ++i) {
                      MarketInputs in{s(i), k(i),
                                       r(i), q(i),
                                       v(i), t(i),
                                       ty(i) == 0 ? OptionType::Call : OptionType::Put};
                      const PricingResult result = BlackScholes::price_with_greeks(in);
                      price_m(i) = result.price;
                      delta_m(i) = result.greeks.delta;
                      gamma_m(i) = result.greeks.gamma;
                      vega_m(i) = result.greeks.vega;
                      theta_m(i) = result.greeks.theta;
                      rho_m(i) = result.greeks.rho;
                  }
              }
              return py::make_tuple(price, delta, gamma, vega, theta, rho);
          },
          py::arg("spot"), py::arg("strike"), py::arg("rate"), py::arg("dividend_yield"),
          py::arg("vol"), py::arg("maturity"), py::arg("type"),
          "Struct-of-arrays batch price+Greeks: NumPy arrays in, NumPy arrays out "
          "((price, delta, gamma, vega, theta, rho)), type as 0=Call/1=Put. GIL released "
          "for the whole loop. See bs_price_with_greeks_batch's docstring for why this "
          "exists alongside the list-of-MarketInputs version.");

    m.def("bs_implied_vol_batch_arrays",
          [](py::array_t<double> spot, py::array_t<double> strike, py::array_t<double> rate,
             py::array_t<double> dividend_yield, py::array_t<double> vol,
             py::array_t<double> maturity, py::array_t<int> type,
             py::array_t<double> market_price, double initial_guess, int max_iter, double tol) {
              const py::ssize_t n = spot.size();
              if (strike.size() != n || rate.size() != n || dividend_yield.size() != n ||
                  vol.size() != n || maturity.size() != n || type.size() != n ||
                  market_price.size() != n) {
                  throw std::invalid_argument("all input arrays must be the same length");
              }
              auto s = spot.unchecked<1>();
              auto k = strike.unchecked<1>();
              auto r = rate.unchecked<1>();
              auto q = dividend_yield.unchecked<1>();
              auto v = vol.unchecked<1>();
              auto t = maturity.unchecked<1>();
              auto ty = type.unchecked<1>();
              auto mp = market_price.unchecked<1>();

              py::array_t<double> iv(n);
              auto iv_m = iv.mutable_unchecked<1>();

              {
                  py::gil_scoped_release release;
                  for (py::ssize_t i = 0; i < n; ++i) {
                      MarketInputs in{s(i), k(i),
                                       r(i), q(i),
                                       v(i), t(i),
                                       ty(i) == 0 ? OptionType::Call : OptionType::Put};
                      iv_m(i) = BlackScholes::implied_vol(in, mp(i), initial_guess, max_iter, tol);
                  }
              }
              return iv;
          },
          py::arg("spot"), py::arg("strike"), py::arg("rate"), py::arg("dividend_yield"),
          py::arg("vol"), py::arg("maturity"), py::arg("type"), py::arg("market_price"),
          py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100, py::arg("tol") = 1e-8,
          "Struct-of-arrays batch implied-vol solve: NumPy arrays in, NumPy array out. "
          "GIL released for the whole loop.");

    // gil_scoped_release below is applied selectively, not blanket: only
    // to calls slow enough (single-digit microseconds or more) that
    // releasing/reacquiring the GIL is a rounding error against the
    // compute it unblocks, not a call slow ENOUGH to bother -- releasing
    // it around e.g. bs_price (sub-microsecond, see benchmarks/) would
    // make that call slower, not faster, since the release/reacquire
    // pair has its own real (if small) fixed cost.
    py::class_<MonteCarloPricer>(m, "MonteCarloPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price_european", &MonteCarloPricer::price_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("greeks_european", &MonteCarloPricer::greeks_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true, py::arg("bump_frac") = 0.01,
             py::call_guard<py::gil_scoped_release>());

    py::class_<AmericanPricer>(m, "AmericanPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price", &AmericanPricer::price, py::arg("inputs"), py::arg("num_paths"),
             py::arg("num_steps"), py::arg("poly_degree") = 2, py::arg("num_calibration_paths") = 0,
             py::call_guard<py::gil_scoped_release>(),
             "American-style option price via Longstaff-Schwartz least-squares Monte Carlo, "
             "using an independently-seeded calibration path set (num_calibration_paths, "
             "defaults to num_paths) separate from the pricing path set.");

    py::class_<HestonParams>(m, "HestonParams")
        .def(py::init([](double kappa, double theta, double xi, double rho, double v0) {
                 return HestonParams{kappa, theta, xi, rho, v0};
             }),
             py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"), py::arg("v0"))
        .def_readwrite("kappa", &HestonParams::kappa)
        .def_readwrite("theta", &HestonParams::theta)
        .def_readwrite("xi", &HestonParams::xi)
        .def_readwrite("rho", &HestonParams::rho)
        .def_readwrite("v0", &HestonParams::v0)
        .def("__repr__", [](const HestonParams& p) {
            return "HestonParams(kappa=" + std::to_string(p.kappa) +
                   ", theta=" + std::to_string(p.theta) + ", xi=" + std::to_string(p.xi) +
                   ", rho=" + std::to_string(p.rho) + ", v0=" + std::to_string(p.v0) + ")";
        });

    m.def("heston_price", &HestonPricer::price, py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"),
          py::arg("params"), py::call_guard<py::gil_scoped_release>());
    m.def("heston_price_batch", &HestonPricer::price_batch, py::arg("spot"), py::arg("strikes"),
          py::arg("types"), py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"),
          py::arg("params"), py::arg("num_nodes") = 1500, py::arg("phi_max") = 150.0,
          py::call_guard<py::gil_scoped_release>(),
          "Prices a whole strike grid in one call, sharing characteristic-function "
          "evaluations across strikes -- see heston.hpp for why this is faster, not just "
          "more convenient, than calling heston_price in a loop, and for when it isn't.");
    m.def("heston_satisfies_feller_condition", &HestonPricer::satisfies_feller_condition,
          py::arg("params"));

    py::class_<HestonJacobian>(m, "HestonJacobian")
        .def_readwrite("price", &HestonJacobian::price)
        .def_readwrite("d_kappa", &HestonJacobian::d_kappa)
        .def_readwrite("d_theta", &HestonJacobian::d_theta)
        .def_readwrite("d_xi", &HestonJacobian::d_xi)
        .def_readwrite("d_rho", &HestonJacobian::d_rho)
        .def_readwrite("d_v0", &HestonJacobian::d_v0)
        .def("__repr__", [](const HestonJacobian& j) {
            return "HestonJacobian(price=" + std::to_string(j.price) +
                   ", d_kappa=" + std::to_string(j.d_kappa) +
                   ", d_theta=" + std::to_string(j.d_theta) +
                   ", d_xi=" + std::to_string(j.d_xi) + ", d_rho=" + std::to_string(j.d_rho) +
                   ", d_v0=" + std::to_string(j.d_v0) + ")";
        });
    m.def("heston_price_jacobian", &HestonPricer::price_jacobian, py::arg("spot"),
          py::arg("strike"), py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"),
          py::arg("type"), py::arg("params"), py::call_guard<py::gil_scoped_release>(),
          "price() plus its exact partials w.r.t. all 5 Heston parameters in one pass "
          "(forward-mode AD, not finite differences) -- see heston.hpp for why this needs a "
          "second, independent differentiation unit rather than literal complex-step, and "
          "bscpp.backtest.heston_calibration for how calibrate_heston uses it.");
    m.def("heston_price_jacobian_batch", &HestonPricer::price_jacobian_batch, py::arg("spot"),
          py::arg("strikes"), py::arg("types"), py::arg("rate"), py::arg("dividend_yield"),
          py::arg("maturity"), py::arg("params"), py::arg("num_nodes") = 1500,
          py::arg("phi_max") = 150.0, py::call_guard<py::gil_scoped_release>(),
          "heston_price_jacobian, batched across a strike grid the same way "
          "heston_price_batch batches heston_price -- see heston.hpp for why the per-strike "
          "version alone is NOT a win over finite differences (measured ~3.6x slower).");
    m.def("heston_price_cos", &HestonPricer::price_cos, py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"),
          py::arg("params"), py::arg("num_terms") = 160, py::call_guard<py::gil_scoped_release>(),
          "Fang & Oosterlee (2008) COS-method price -- see heston.hpp for the fixed-node-vs-"
          "adaptive-quadrature tradeoff against heston_price, and its accuracy/speed profile "
          "(cross-checked to <0.02% relative error against heston_price across a 300-case "
          "random stress sweep; falls back to heston_price itself on the rare parameter "
          "combinations where its adaptive truncation search doesn't converge).");

    py::class_<HestonMCPricer>(m, "HestonMCPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price", &HestonMCPricer::price, py::arg("spot"), py::arg("strike"), py::arg("rate"),
             py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"), py::arg("params"),
             py::arg("num_paths"), py::arg("num_steps"), py::call_guard<py::gil_scoped_release>())
        .def("price_qe", &HestonMCPricer::price_qe, py::arg("spot"), py::arg("strike"),
             py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"),
             py::arg("params"), py::arg("num_paths"), py::arg("num_steps"),
             py::call_guard<py::gil_scoped_release>(),
             "Andersen (2008) QE scheme -- see heston.hpp for why this needs far fewer "
             "num_steps than price() (full-truncation Euler) to reach comparable accuracy, "
             "especially when the Feller condition is badly violated.");

    // Testing-only: exposes raw Philox4x64 draws so test_philox.py can
    // cross-validate them bit-for-bit against numpy.random.Philox on the
    // same (seed, stream) -- not part of the public pricing API.
    m.def("_philox_raw_draws", [](std::uint64_t seed, std::uint64_t stream, int n) {
        Philox4x64 rng(seed, stream);
        std::vector<std::uint64_t> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = rng();
        return out;
    }, py::arg("seed"), py::arg("stream"), py::arg("n"));

    // Testing-only: draws n values starting from an explicit seek()
    // position, so test_philox.py can confirm seek() lands exactly where
    // the equivalent number of sequential draws would have.
    m.def("_philox_seek_draws",
          [](std::uint64_t seed, std::uint64_t c0, std::uint64_t c1, std::uint64_t c2,
             std::uint64_t c3, int n) {
              Philox4x64 rng(seed);
              rng.seek(c0, c1, c2, c3);
              std::vector<std::uint64_t> out(static_cast<size_t>(n));
              for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = rng();
              return out;
          },
          py::arg("seed"), py::arg("c0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("n"));

    m.def("crr_price", &CRRPricer::price, py::arg("spot"), py::arg("strike"), py::arg("rate"),
          py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"), py::arg("vol"),
          py::arg("num_steps") = 200, py::call_guard<py::gil_scoped_release>(),
          "American-style price via a dividend-aware Cox-Ross-Rubinstein binomial "
          "tree -- see crr_tree.hpp for why this, not Longstaff-Schwartz MC, is the "
          "chain pipeline's American pricer.");
    m.def("crr_implied_vol", &CRRPricer::implied_vol, py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"),
          py::arg("market_price"), py::arg("num_steps") = 200, py::arg("tol") = 1e-6,
          py::arg("max_iter") = 100, py::call_guard<py::gil_scoped_release>(),
          "American implied vol via Brent's method against crr_price -- NaN if "
          "market_price isn't bracketed by [1e-6, 5.0] vol, matching bs_implied_vol's "
          "contract exactly.");
}
