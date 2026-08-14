import math

import bscpp


def test_american_call_equals_european_without_dividends():
    # Classic result: with no dividends, early exercise of an American call
    # is never optimal, so CRR should converge to the closed-form European
    # price as num_steps grows.
    euro = bscpp.price(100, 100, 0.05, 0.2, 1.0, "call")
    amer = bscpp.price_american_crr(100, 100, 0.05, 0.2, 1.0, "call", num_steps=500)
    assert math.isclose(amer, euro, abs_tol=0.01)


def test_american_put_has_early_exercise_premium():
    euro = bscpp.price(36, 40, 0.06, 0.2, 1.0, "put")
    amer = bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=500)
    assert amer > euro


def test_american_put_matches_longstaff_schwartz_2001_benchmark():
    # Same S0=36, K=40, r=6%, sigma=20%, T=1y case test_american.py checks
    # against LSM -- independent implementations of this exact case
    # converge to values in the ~4.47-4.48 range.
    price = bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=1000)
    assert 4.40 < price < 4.55


def test_crr_matches_lsm_independently():
    # Two independent American pricers (deterministic tree vs. Monte Carlo
    # regression) should agree with each other, not just with a literature
    # benchmark -- the real cross-check this project's "two methods" pattern
    # (Heston MC vs. analytic, etc.) is built around.
    crr = bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=1000)
    lsm = bscpp.price_american(36, 40, 0.06, 0.2, 1.0, "put", num_paths=80_000, num_steps=50, seed=7)
    assert abs(crr - lsm.price) < 5 * lsm.std_error + 0.05


def test_crr_price_converges_as_steps_increase():
    # Binomial trees oscillate with odd/even step parity but the amplitude
    # of that oscillation should shrink as num_steps grows.
    prices = [bscpp.price_american_crr(36, 40, 0.06, 0.2, 1.0, "put", num_steps=n)
              for n in (50, 200, 800, 3200)]
    diffs = [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]
    assert diffs[-1] < diffs[0]


def test_crr_implied_vol_round_trips():
    true_vol = 0.28
    price = bscpp.crr_price(100.0, 95.0, 0.05, 0.02, 0.5, bscpp.OptionType.Put, true_vol, 300)
    iv = bscpp.crr_implied_vol(100.0, 95.0, 0.05, 0.02, 0.5, bscpp.OptionType.Put, price, 300)
    assert math.isclose(iv, true_vol, rel_tol=1e-6)


def test_crr_implied_vol_nan_on_unbracketed_price():
    # A price below intrinsic value (here, below max(K-S,0)=5 for a put
    # struck at 100 with spot 95) can't be produced by ANY volatility --
    # matches bs_implied_vol's contract of returning NaN rather than a
    # nonsense answer.
    iv = bscpp.crr_implied_vol(95.0, 100.0, 0.05, 0.0, 0.5, bscpp.OptionType.Put, 1.0, 300)
    assert iv != iv  # NaN


def test_crr_price_matches_put_call_parity_bound_direction():
    # American put-call parity is an inequality, not an equality (unlike
    # European): S - K <= C_amer - P_amer <= S - K*e^{-rT} (Merton 1973).
    # Confirms the tree respects that bound rather than testing a false
    # equality.
    spot, strike, rate, div, vol, t = 100.0, 100.0, 0.05, 0.0, 0.25, 1.0
    call = bscpp.price_american_crr(spot, strike, rate, vol, t, "call", div, num_steps=500)
    put = bscpp.price_american_crr(spot, strike, rate, vol, t, "put", div, num_steps=500)
    lower = spot - strike
    upper = spot - strike * math.exp(-rate * t)
    assert lower - 1e-6 <= call - put <= upper + 1e-6
