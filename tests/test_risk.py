import math

import bscpp


def _sample_positions():
    return [
        bscpp.Position("AAPL call", "call", quantity=10, underlying="AAPL", spot=200, rate=0.05,
                        strike=210, vol=0.25, maturity=0.5),
        bscpp.Position("AAPL stock hedge", "stock", quantity=-500, underlying="AAPL", spot=200,
                        rate=0.05),
        bscpp.Position("MSFT put", "put", quantity=-20, underlying="MSFT", spot=400, rate=0.05,
                        strike=380, vol=0.22, maturity=0.25),
    ]


def test_stock_position_greeks_are_pure_delta():
    pos = bscpp.Position("hedge", "stock", quantity=-500, underlying="AAPL", spot=200, rate=0.05)
    g = pos.greeks()
    # dollar delta of a stock position is just its own market value: a
    # stock's raw delta is always 1, so dollar_delta = 1 * quantity * spot.
    assert g["delta"] == -500 * 200
    assert g["gamma"] == 0.0 and g["vega"] == 0.0 and g["theta"] == 0.0 and g["rho"] == 0.0
    assert g["price"] == -500 * 200


def test_option_position_greeks_scale_by_quantity():
    single = bscpp.Position("1x", "call", quantity=1, underlying="AAPL", spot=200, rate=0.05,
                             strike=210, vol=0.25, maturity=0.5)
    tenx = bscpp.Position("10x", "call", quantity=10, underlying="AAPL", spot=200, rate=0.05,
                           strike=210, vol=0.25, maturity=0.5)
    g1, g10 = single.greeks(), tenx.greeks()
    for key in ["price", "delta", "gamma", "vega", "theta", "rho"]:
        assert math.isclose(g10[key], 10 * g1[key], rel_tol=1e-9)


def test_net_greeks_sums_across_positions():
    positions = _sample_positions()
    mgr = bscpp.PortfolioRiskManager()
    net = mgr.net_greeks(positions)
    df = mgr.position_greeks(positions)
    for key in ["price", "delta", "gamma", "vega", "theta", "rho"]:
        assert math.isclose(net[key], df[key].sum(), rel_tol=1e-9)


def test_greeks_by_underlying_groups_correctly():
    positions = _sample_positions()
    mgr = bscpp.PortfolioRiskManager()
    by_name = mgr.greeks_by_underlying(positions)
    assert set(by_name["underlying"]) == {"AAPL", "MSFT"}

    aapl_row = by_name[by_name["underlying"] == "AAPL"].iloc[0]
    df = mgr.position_greeks(positions)
    expected_aapl_delta = df[df["underlying"] == "AAPL"]["delta"].sum()
    assert math.isclose(aapl_row["delta"], expected_aapl_delta, rel_tol=1e-9)


def test_check_limits_flags_portfolio_and_per_underlying_breaches():
    positions = _sample_positions()
    limits = bscpp.RiskLimits(max_abs_vega=50, per_underlying_max_abs_delta=100)
    mgr = bscpp.PortfolioRiskManager(limits)
    breaches = mgr.check_limits(positions)

    scopes_metrics = {(b.scope, b.metric) for b in breaches}
    assert ("portfolio", "vega") in scopes_metrics
    assert ("AAPL", "delta") in scopes_metrics


def test_check_limits_empty_when_no_limits_set():
    positions = _sample_positions()
    mgr = bscpp.PortfolioRiskManager()  # no limits configured
    assert mgr.check_limits(positions) == []


def test_check_limits_empty_when_within_generous_limits():
    # Net dollar delta for _sample_positions() is ~-96,935 (mostly the
    # -500-share AAPL stock hedge at spot=200 -> -$100,000 dollar delta,
    # partly offset by the option legs) -- 200,000 keeps real margin
    # rather than sitting a coincidental 3% above the actual value.
    positions = _sample_positions()
    limits = bscpp.RiskLimits(max_abs_delta=200_000, max_abs_vega=100_000, max_abs_gamma=1000,
                               max_abs_theta=100_000)
    mgr = bscpp.PortfolioRiskManager(limits)
    assert mgr.check_limits(positions) == []
