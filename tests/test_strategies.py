import math

import bscpp


def _pricer():
    return bscpp.StrategyPricer(rate=0.05)


def test_straddle_breakevens_equal_strike_plus_minus_premium():
    pricer = _pricer()
    legs = bscpp.straddle(strike=100, quantity=1)
    result = pricer.price(legs, spot=100, vol=0.2, maturity=1.0)
    _, breakevens = pricer.payoff_diagram(legs, spot=100, vol=0.2, maturity=1.0)

    assert len(breakevens) == 2
    assert math.isclose(min(breakevens), 100 - result.net_price, abs_tol=1e-6)
    assert math.isclose(max(breakevens), 100 + result.net_price, abs_tol=1e-6)


def test_vertical_call_spread_bounded_risk_and_reward():
    pricer = _pricer()
    legs = bscpp.vertical_spread("call", long_strike=95, short_strike=105, quantity=1)
    result = pricer.price(legs, spot=100, vol=0.2, maturity=1.0)
    plot_df, _ = pricer.payoff_diagram(legs, spot=100, vol=0.2, maturity=1.0)

    width = 105 - 95
    assert math.isclose(plot_df["pnl_at_expiry"].max(), width - result.net_price, abs_tol=1e-2)
    assert math.isclose(plot_df["pnl_at_expiry"].min(), -result.net_price, abs_tol=1e-6)
    # a debit spread costs money to put on
    assert result.net_price > 0


def test_strip_is_bearish_biased_strap_is_bullish_biased():
    pricer = _pricer()
    strip_result = pricer.price(bscpp.strip(strike=100), spot=100, vol=0.2, maturity=1.0)
    strap_result = pricer.price(bscpp.strap(strike=100), spot=100, vol=0.2, maturity=1.0)

    assert strip_result.net_delta < 0  # 2 puts + 1 call => net short delta
    assert strap_result.net_delta > 0  # 2 calls + 1 put => net long delta
    # both are long gamma/vega (long volatility structures)
    assert strip_result.net_gamma > 0
    assert strap_result.net_gamma > 0


def test_butterfly_max_gain_at_middle_strike():
    pricer = _pricer()
    legs = bscpp.butterfly("call", low_strike=90, mid_strike=100, high_strike=110)
    plot_df, _ = pricer.payoff_diagram(legs, spot=100, vol=0.2, maturity=0.5, spot_range=(0.7, 1.3))
    peak_row = plot_df.loc[plot_df["pnl_at_expiry"].idxmax()]
    assert abs(peak_row["spot"] - 100) < 1.0  # peak P&L should sit at the body strike


def test_strategy_pricer_accepts_per_strike_vol_dict():
    pricer = _pricer()
    legs = bscpp.strangle(call_strike=110, put_strike=90)
    vol_by_strike = {110: 0.22, 90: 0.28}  # a simple skew
    result = pricer.price(legs, spot=100, vol=vol_by_strike, maturity=0.5)
    flat_result = pricer.price(legs, spot=100, vol=0.25, maturity=0.5)
    assert result.net_price != flat_result.net_price
