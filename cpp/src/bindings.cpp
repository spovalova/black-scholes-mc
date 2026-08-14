#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bscpp/black_scholes.hpp"
#include "bscpp/crr_tree.hpp"
#include "bscpp/heston.hpp"
#include "bscpp/longstaff_schwartz.hpp"
#include "bscpp/monte_carlo.hpp"
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
          "Price + Greeks for a list of MarketInputs in one C++ call (avoids per-contract "
          "Python<->C++ crossing overhead when pricing a whole chain).");
    m.def("bs_implied_vol", &BlackScholes::implied_vol, py::arg("inputs"), py::arg("market_price"),
          py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100, py::arg("tol") = 1e-8);
    m.def("bs_implied_vol_batch", &BlackScholes::implied_vol_batch, py::arg("inputs"),
          py::arg("market_prices"), py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100,
          py::arg("tol") = 1e-8);

    py::class_<MonteCarloPricer>(m, "MonteCarloPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price_european", &MonteCarloPricer::price_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true)
        .def("greeks_european", &MonteCarloPricer::greeks_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true, py::arg("bump_frac") = 0.01);

    py::class_<AmericanPricer>(m, "AmericanPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price", &AmericanPricer::price, py::arg("inputs"), py::arg("num_paths"),
             py::arg("num_steps"), py::arg("poly_degree") = 2, py::arg("num_calibration_paths") = 0,
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
          py::arg("params"));
    m.def("heston_price_batch", &HestonPricer::price_batch, py::arg("spot"), py::arg("strikes"),
          py::arg("types"), py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"),
          py::arg("params"), py::arg("num_nodes") = 1500, py::arg("phi_max") = 150.0,
          "Prices a whole strike grid in one call, sharing characteristic-function "
          "evaluations across strikes -- see heston.hpp for why this is faster, not just "
          "more convenient, than calling heston_price in a loop, and for when it isn't.");
    m.def("heston_satisfies_feller_condition", &HestonPricer::satisfies_feller_condition,
          py::arg("params"));

    py::class_<HestonMCPricer>(m, "HestonMCPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price", &HestonMCPricer::price, py::arg("spot"), py::arg("strike"), py::arg("rate"),
             py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"), py::arg("params"),
             py::arg("num_paths"), py::arg("num_steps"));

    m.def("crr_price", &CRRPricer::price, py::arg("spot"), py::arg("strike"), py::arg("rate"),
          py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"), py::arg("vol"),
          py::arg("num_steps") = 200,
          "American-style price via a dividend-aware Cox-Ross-Rubinstein binomial "
          "tree -- see crr_tree.hpp for why this, not Longstaff-Schwartz MC, is the "
          "chain pipeline's American pricer.");
    m.def("crr_implied_vol", &CRRPricer::implied_vol, py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend_yield"), py::arg("maturity"), py::arg("type"),
          py::arg("market_price"), py::arg("num_steps") = 200, py::arg("tol") = 1e-6,
          py::arg("max_iter") = 100,
          "American implied vol via Brent's method against crr_price -- NaN if "
          "market_price isn't bracketed by [1e-6, 5.0] vol, matching bs_implied_vol's "
          "contract exactly.");
}
