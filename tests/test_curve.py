import math

import pytest

from bscpp.curve import ZeroCurve, resolve_rate


def test_flat_curve_returns_constant_rate_everywhere():
    curve = ZeroCurve.flat(0.05)
    for t in [0.01, 0.5, 1.0, 5.0, 30.0]:
        assert curve.zero_rate(t) == 0.05
    assert math.isclose(curve.df(1.0), math.exp(-0.05), rel_tol=1e-12)


def test_step_interpolation_uses_left_pillar_up_to_next_tenor():
    curve = ZeroCurve(tenors=(0.25, 1.0, 5.0), rates=(0.04, 0.045, 0.05))
    assert curve.zero_rate(0.1) == 0.04  # before first tenor -> first rate
    assert curve.zero_rate(0.25) == 0.04  # at a pillar -> that pillar's rate
    assert curve.zero_rate(0.5) == 0.04  # between pillars -> left (lower) pillar
    assert curve.zero_rate(0.999) == 0.04
    assert curve.zero_rate(1.0) == 0.045
    assert curve.zero_rate(3.0) == 0.045
    assert curve.zero_rate(5.0) == 0.05
    assert curve.zero_rate(100.0) == 0.05  # past last tenor -> flat extrapolation


def test_df_matches_zero_rate_at_each_point():
    curve = ZeroCurve(tenors=(0.5, 2.0), rates=(0.03, 0.06))
    for t in [0.1, 0.5, 1.0, 2.0, 10.0]:
        expected = math.exp(-curve.zero_rate(t) * t)
        assert math.isclose(curve.df(t), expected, rel_tol=1e-12)


def test_forward_rate_recovers_flat_rate_on_a_flat_curve():
    curve = ZeroCurve.flat(0.05)
    # on a flat curve, every forward segment equals the flat zero rate
    assert math.isclose(curve.forward_rate(0.5, 1.5), 0.05, rel_tol=1e-9)


def test_forward_rate_between_pillars_is_internally_consistent():
    curve = ZeroCurve(tenors=(1.0, 2.0), rates=(0.03, 0.05))
    f = curve.forward_rate(1.0, 2.0)
    # df(2) should equal df(1) * exp(-f * 1) by construction
    assert math.isclose(curve.df(2.0), curve.df(1.0) * math.exp(-f * 1.0), rel_tol=1e-12)


def test_forward_rate_rejects_non_increasing_interval():
    curve = ZeroCurve.flat(0.05)
    with pytest.raises(ValueError):
        curve.forward_rate(1.0, 1.0)
    with pytest.raises(ValueError):
        curve.forward_rate(2.0, 1.0)


def test_curve_rejects_mismatched_or_empty_or_unsorted_input():
    with pytest.raises(ValueError):
        ZeroCurve(tenors=(1.0, 2.0), rates=(0.05,))
    with pytest.raises(ValueError):
        ZeroCurve(tenors=(), rates=())
    with pytest.raises(ValueError):
        ZeroCurve(tenors=(2.0, 1.0), rates=(0.05, 0.04))


def test_resolve_rate_passes_through_bare_float():
    assert resolve_rate(0.05, 1.0) == 0.05
    assert resolve_rate(0.03, 99.0) == 0.03


def test_resolve_rate_looks_up_curve_at_given_maturity():
    curve = ZeroCurve(tenors=(0.25, 1.0), rates=(0.04, 0.05))
    assert resolve_rate(curve, 0.1) == 0.04
    assert resolve_rate(curve, 2.0) == 0.05
