import math

import bscpp


def test_american_call_equals_european_without_dividends():
    # Classic result: with no dividends, early exercise of an American call
    # is never optimal, so its price should collapse to the European price.
    euro = bscpp.price(100, 100, 0.05, 0.2, 1.0, "call")
    amer = bscpp.price_american(100, 100, 0.05, 0.2, 1.0, "call", num_paths=40_000, num_steps=50)
    assert abs(amer.price - euro) < 5 * amer.std_error + 0.05


def test_american_put_has_early_exercise_premium():
    # American puts are always worth at least as much as their European
    # counterpart, strictly more away from the boundary cases.
    euro = bscpp.price(36, 40, 0.06, 0.2, 1.0, "put")
    amer = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=60_000, num_steps=50)
    assert amer.price > euro


def test_american_put_matches_longstaff_schwartz_2001_benchmark():
    # S0=36, K=40, r=6%, sigma=20%, T=1y is the headline example from
    # Longstaff & Schwartz (2001), "Valuing American Options by Simulation:
    # A Simple Least-Squares Approach". Published price ~= 4.478.
    result = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=80_000, num_steps=50,
                                   seed=7)
    assert math.isclose(result.price, 4.478, abs_tol=0.10)
