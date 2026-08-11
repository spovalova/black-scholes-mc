#include <pybind11/pybind11.h>

#include "bscpp/black_scholes.hpp"
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
    m.def("bs_implied_vol", &BlackScholes::implied_vol, py::arg("inputs"), py::arg("market_price"),
          py::arg("initial_guess") = 0.2, py::arg("max_iter") = 100, py::arg("tol") = 1e-8);

    py::class_<MonteCarloPricer>(m, "MonteCarloPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price_european", &MonteCarloPricer::price_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true)
        .def("greeks_european", &MonteCarloPricer::greeks_european, py::arg("inputs"),
             py::arg("num_paths"), py::arg("antithetic") = true, py::arg("bump_frac") = 0.01);

    py::class_<AmericanPricer>(m, "AmericanPricer")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
        .def("price", &AmericanPricer::price, py::arg("inputs"), py::arg("num_paths"),
             py::arg("num_steps"), py::arg("poly_degree") = 2,
             "American-style option price via Longstaff-Schwartz least-squares Monte Carlo.");
}
