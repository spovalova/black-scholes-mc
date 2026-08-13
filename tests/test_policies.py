import datetime as dt
import math

import numpy as np
import pandas as pd

from bscpp.backtest import (
    BandPolicy,
    CallablePolicy,
    DeltaPolicy,
    HedgeState,
    HedgingBacktester,
    WhalleyWilmottPolicy,
)


def _gbm_path(vol=0.30, days=60, spot=100.0, rate=0.05, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(dt.date.today(), periods=days + 1, freq="D")
    dt_frac = 1 / 365
    rets = rng.normal((rate - 0.5 * vol**2) * dt_frac, vol * math.sqrt(dt_frac), size=days)
    path = spot * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return pd.Series(path, index=dates), dates[-1].date()


def _state(delta=0.5, gamma=0.02, spot=100.0, t=0.25, cost_frac=5e-4):
    return HedgeState(t=t, spot=spot, delta=delta, gamma=gamma, vega=20.0,
                      rate=0.05, cost_frac=cost_frac)


def test_delta_policy_always_rebalances_to_delta():
    p = DeltaPolicy()
    assert p.target_shares(0.0, _state(delta=0.62)) == 0.62
    assert p.target_shares(0.9, _state(delta=0.62)) == 0.62


def test_band_policy_no_trade_inside_band_and_trades_to_edge():
    p = BandPolicy(band=0.05)
    s = _state(delta=0.50)
    assert p.target_shares(0.53, s) == 0.53          # inside: hold
    assert p.target_shares(0.58, s) == 0.55          # breach above: to upper edge
    assert p.target_shares(0.40, s) == 0.45          # breach below: to lower edge
    # zero band reduces to DeltaPolicy
    assert BandPolicy(band=0.0).target_shares(0.9, s) == 0.50


def test_whalley_wilmott_band_scalings():
    # The published asymptotics: half-width ~ Gamma^(2/3) and ~ cost^(1/3),
    # shrinking in risk aversion. Verify all three scalings numerically.
    p = WhalleyWilmottPolicy(risk_aversion=1.0)

    h_base = p.band_half_width(_state(gamma=0.02, cost_frac=5e-4))
    h_4x_gamma = p.band_half_width(_state(gamma=0.08, cost_frac=5e-4))
    assert math.isclose(h_4x_gamma / h_base, 4.0 ** (2.0 / 3.0), rel_tol=1e-9)

    h_8x_cost = p.band_half_width(_state(gamma=0.02, cost_frac=8 * 5e-4))
    assert math.isclose(h_8x_cost / h_base, 2.0, rel_tol=1e-9)  # 8^(1/3) = 2

    tight = WhalleyWilmottPolicy(risk_aversion=8.0).band_half_width(
        _state(gamma=0.02, cost_frac=5e-4))
    assert math.isclose(h_base / tight, 2.0, rel_tol=1e-9)  # lam^( -1/3 )

    # zero cost -> band collapses -> exact delta tracking
    assert p.band_half_width(_state(cost_frac=0.0)) == 0.0
    assert p.target_shares(0.9, _state(delta=0.5, cost_frac=0.0)) == 0.5


def test_band_policies_cut_transaction_costs_vs_daily():
    # The entire point of no-trade bands: with costs on, band policies must
    # trade less notional and pay materially less spread than rebalancing
    # to the exact delta every day, on the same path.
    path, expiry = _gbm_path(seed=11)
    bt = HedgingBacktester(rate=0.05, transaction_cost_bps=10.0)

    daily = bt.run(path, 100.0, expiry, 0.30, policy=DeltaPolicy())
    band = bt.run(path, 100.0, expiry, 0.30, policy=BandPolicy(band=0.05))
    ww = bt.run(path, 100.0, expiry, 0.30, policy=WhalleyWilmottPolicy(risk_aversion=1.0))

    cost_daily = daily["transaction_cost"].sum()
    cost_band = band["transaction_cost"].sum()
    cost_ww = ww["transaction_cost"].sum()

    assert cost_band < 0.7 * cost_daily
    assert cost_ww < cost_daily
    # trade count: bands should skip many rebalances entirely
    trades_daily = (daily["transaction_cost"] > 1e-12).sum()
    trades_band = (band["transaction_cost"] > 1e-12).sum()
    assert trades_band < trades_daily


def test_band_policy_attribution_identity_still_holds():
    # With a band policy, shares deliberately deviate from delta; the
    # attribution's delta_gap term must absorb that first-order P&L so the
    # residual stays a genuine higher-order term, and the accounting
    # identity (daily P&L sums to final cumulative P&L) is untouched.
    path, expiry = _gbm_path(seed=7)
    bt = HedgingBacktester(rate=0.05, transaction_cost_bps=10.0)
    result = bt.run(path, 100.0, expiry, 0.30, policy=BandPolicy(band=0.10))
    attr = bt.attribute_pnl(result)

    # telescoping identity: daily P&L sums to pv_final - pv_0 (pv_0 is the
    # initial hedge's transaction cost, booked at inception, not in day 1+)
    assert math.isclose(attr["realized_pnl"].sum(),
                         result["portfolio_value"].iloc[-1] - result["portfolio_value"].iloc[0],
                         abs_tol=1e-9)
    # the delta-gap term is genuinely active under a band policy
    assert attr["delta_gap_pnl"].abs().sum() > 0.0
    # and the decomposition still explains most of the realized P&L
    assert attr["attribution_error"].abs().sum() < 0.5 * attr["realized_pnl"].abs().sum()


def test_risk_aversion_scaling_reproduces_exact_band_multiples():
    # examples/hedging_policy_frontier_study.py sweeps band width via
    # risk_aversion = lam0 / c**3, relying on this producing EXACTLY
    # band(lam0) * c (from h ~ risk_aversion^(-1/3)). This is the load-
    # bearing identity for that study's entire multiplier grid -- verified
    # here to more decimal places than the general scaling test above, and
    # across multiple states/lam0 values.
    for state in (_state(gamma=0.02, spot=100.0, cost_frac=5e-4),
                  _state(gamma=0.11, spot=340.0, cost_frac=8e-4)):
        for lam0 in (0.001, 0.01, 0.1, 1.0):
            base = WhalleyWilmottPolicy(risk_aversion=lam0).band_half_width(state)
            for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
                h = WhalleyWilmottPolicy(risk_aversion=lam0 / c ** 3).band_half_width(state)
                assert math.isclose(h, c * base, rel_tol=1e-9)


def test_callable_policy_wraps_custom_logic():
    p = CallablePolicy(lambda held, state: 0.5 * (held + state.delta))
    assert p.target_shares(0.0, _state(delta=0.8)) == 0.4
