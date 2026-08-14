import datetime as dt
import math

import pandas as pd

import bscpp
from bscpp.backtest import Backtester, MockProvider, StripPricer, extract_forward_and_carry


def test_mock_provider_chain_shape():
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    expiration = dt.date.today() + dt.timedelta(days=45)
    chain = provider.get_option_chain("TEST", expiration)
    assert not chain.empty
    assert {"strike", "type", "bid", "ask", "last"}.issubset(chain.columns)
    assert set(chain["type"].unique()) <= {"call", "put"}


def test_strip_pricer_prices_mock_chain():
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    pricer = StripPricer(provider, rate=0.05, mc_paths=20_000)
    expiration = dt.date.today() + dt.timedelta(days=45)

    result = pricer.price_strip("TEST", expiration, strike_range=(0.9, 1.1), use_mc=True)

    assert not result.empty
    assert "bs_price" in result.columns
    assert "bs_error_vs_market" in result.columns
    # OTM-only IV solving means the ITM leg at each strike is a deliberate
    # NaN fallback (see extract_forward_and_carry / price_strip), so only
    # check the rows that were actually priced -- and confirm there ARE
    # some, and some legitimate fallbacks too, not an accident.
    priced = result.dropna(subset=["bs_price"])
    assert not priced.empty
    assert result["bs_price"].isna().any()
    # theoretical prices should be positive and in a sane range vs spot
    assert (priced["bs_price"] >= 0).all()
    # BS and MC should roughly agree per-contract
    diff = (priced["bs_price"] - priced["mc_price"]).abs()
    assert (diff < 1.0).all()


def test_backtester_run_and_summary():
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    pricer = StripPricer(provider, rate=0.05, mc_paths=5_000)
    expiration = dt.date.today() + dt.timedelta(days=45)
    backtester = Backtester(pricer)

    dates = [dt.date.today(), dt.date.today() - dt.timedelta(days=1)]
    results = backtester.run("TEST", expiration, dates, strike_range=(0.9, 1.1), use_mc=False)
    assert not results.empty

    summary = backtester.summary(results)
    assert set(summary.columns) >= {"as_of", "mean_abs_error", "mean_abs_pct_error", "n_contracts"}
    assert len(summary) == len(dates)


def test_strip_pricer_flags_iv_source_and_never_invents_ivs():
    # Regression test: rows whose IV solve fails (or that have no usable
    # quote) used to be silently priced at an invented 0.20 vol with no
    # flag -- contaminating every downstream consumer with fake IVs. Now
    # every row carries iv_source in {"quoted", "solved", "fallback"} and
    # fallback rows have NaN model_iv and NaN pricing outputs.
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    pricer = StripPricer(provider, rate=0.05, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=45)

    result = pricer.price_strip("TEST", expiration, strike_range=(0.9, 1.1), use_mc=False)

    assert "iv_source" in result.columns
    assert set(result["iv_source"].unique()) <= {"quoted", "solved", "fallback"}
    # MockProvider quotes NaN vendor IVs, so every row must be solved or fallback
    assert (result["iv_source"] != "quoted").all()
    # solved rows have finite IVs; fallback rows are NaN everywhere that matters
    solved = result[result["iv_source"] == "solved"]
    fallback = result[result["iv_source"] == "fallback"]
    assert solved["model_iv"].notna().all()
    assert fallback["model_iv"].isna().all()
    assert fallback["bs_price"].isna().all()
    assert fallback["delta"].isna().all()


def test_polygon_provider_follows_pagination():
    # Regression test for a real bug: get_option_chain fetched ONE page
    # (limit=250) and never followed Polygon's next_url cursor, silently
    # truncating any large chain. Simulate a 3-page response with a fake
    # session and assert all pages are concatenated.
    import bscpp.backtest.data_provider as dp

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(url)
            base = "https://api.polygon.io/v3/snapshot/options/SPY"
            page = {
                base: {"results": [{"n": i} for i in range(250)],
                       "next_url": base + "?cursor=page2"},
                base + "?cursor=page2": {"results": [{"n": i} for i in range(250, 500)],
                                          "next_url": base + "?cursor=page3"},
                base + "?cursor=page3": {"results": [{"n": i} for i in range(500, 600)]},
            }
            return FakeResponse(page[url])

    session = FakeSession()
    provider = dp.PolygonProvider(api_key="test-key", session=session)
    data = provider._get("/v3/snapshot/options/SPY")

    assert len(data["results"]) == 600  # 250 + 250 + 100, all three pages
    assert len(session.calls) == 3


def _synthetic_chain(rows):
    """rows: list of (strike, type, price). Builds a minimal chain frame
    with bid==ask==price (zero spread, mid == price exactly)."""
    return pd.DataFrame([
        {"strike": k, "type": t, "bid": p, "ask": p, "last": p}
        for k, t, p in rows
    ])


def test_extract_forward_and_carry_recovers_known_forward():
    # C - P = e^{-rT}(F-K) with F = S*e^{(r-q)T} -- build exact BS prices
    # at a known (r, q) and check the parity-implied forward/carry match.
    spot, rate, div, vol, t_years, strike = 100.0, 0.05, 0.02, 0.20, 1.0, 100.0
    call = bscpp.price(spot, strike, rate, vol, t_years, "call", div)
    put = bscpp.price(spot, strike, rate, vol, t_years, "put", div)
    chain = _synthetic_chain([(strike, "call", call), (strike, "put", put)])

    forward, carry = extract_forward_and_carry(chain, spot, t_years, rate)
    expected_forward = spot * math.exp((rate - div) * t_years)
    assert math.isclose(forward, expected_forward, rel_tol=1e-9)
    assert math.isclose(carry, rate - div, abs_tol=1e-9)


def test_extract_forward_and_carry_picks_min_abs_diff_strike():
    # Three strikes; only the middle one's call/put pair is exactly
    # consistent with the assumed (r, q) -- the other two are deliberately
    # perturbed (simulating bid-ask noise away from the money). The
    # min-|C-P| strike should still recover the true forward closely,
    # confirming the selection logic picks the informative strike.
    spot, rate, div, vol, t_years = 100.0, 0.05, 0.02, 0.20, 1.0
    true_forward = spot * math.exp((rate - div) * t_years)
    rows = []
    for strike, noise in [(80.0, 5.0), (100.0, 0.0), (120.0, -5.0)]:
        call = bscpp.price(spot, strike, rate, vol, t_years, "call", div) + noise
        put = bscpp.price(spot, strike, rate, vol, t_years, "put", div)
        rows.append((strike, "call", call))
        rows.append((strike, "put", put))
    chain = _synthetic_chain(rows)

    forward, _ = extract_forward_and_carry(chain, spot, t_years, rate)
    assert math.isclose(forward, true_forward, rel_tol=1e-6)  # unperturbed (K=100) pair wins


def test_extract_forward_and_carry_nan_without_paired_quotes():
    chain = _synthetic_chain([(100.0, "call", 5.0), (110.0, "call", 2.0)])  # calls only
    forward, carry = extract_forward_and_carry(chain, 100.0, 1.0, 0.05)
    assert forward != forward  # NaN
    assert carry != carry


def test_strip_pricer_solves_otm_only():
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    pricer = StripPricer(provider, rate=0.05, mc_paths=1)
    expiration = dt.date.today() + dt.timedelta(days=45)
    result = pricer.price_strip("TEST", expiration, strike_range=(0.9, 1.1), use_mc=False)

    assert "implied_forward" in result.columns and "implied_carry" in result.columns
    forward = result["implied_forward"].iloc[0]
    assert forward == forward  # a real chain has paired call/put quotes -> not NaN

    calls = result[result["type"] == "call"]
    puts = result[result["type"] == "put"]
    # calls at/above the forward are solved (or fallback only for lack of a
    # mid); calls BELOW the forward (ITM) are never solved, only fallback.
    itm_calls = calls[calls["strike"] < forward]
    otm_calls = calls[calls["strike"] >= forward]
    assert (itm_calls["iv_source"] == "fallback").all()
    assert (otm_calls["iv_source"] == "solved").any()

    itm_puts = puts[puts["strike"] > forward]
    otm_puts = puts[puts["strike"] <= forward]
    assert (itm_puts["iv_source"] == "fallback").all()
    assert (otm_puts["iv_source"] == "solved").any()


def test_strip_pricer_american_mode_reports_crr_price_and_higher_put_iv_solve():
    provider = MockProvider(rate=0.05, spot=100.0, base_vol=0.22)
    european = StripPricer(provider, rate=0.05, mc_paths=1, american=False)
    american = StripPricer(provider, rate=0.05, mc_paths=1, american=True)
    expiration = dt.date.today() + dt.timedelta(days=180)  # long-dated: premium is easy to see

    euro_result = european.price_strip("TEST", expiration, strike_range=(0.9, 1.1), use_mc=False)
    amer_result = american.price_strip("TEST", expiration, strike_range=(0.9, 1.1), use_mc=False)

    assert "crr_price" not in euro_result.columns
    assert "crr_price" in amer_result.columns and "crr_error_vs_market" in amer_result.columns

    solved_amer = amer_result[amer_result["iv_source"] == "solved"]
    assert not solved_amer.empty
    # crr_price at the solved IV should closely reproduce the market mid
    # it was solved from -- the same self-consistency check bs_price gets
    # implicitly in the European case.
    assert (solved_amer["crr_error_vs_market"].abs() < 0.05).all()

    # American puts are worth MORE than European at the same vol (early-
    # exercise premium), so to match the SAME market mid, the American-
    # consistent solve infers a LOWER implied vol than the European solve
    # does -- this is the actual mismatch the CRR tree exists to remove.
    solved_puts_euro = euro_result[(euro_result["type"] == "put") &
                                    (euro_result["iv_source"] == "solved")]
    solved_puts_amer = amer_result[(amer_result["type"] == "put") &
                                    (amer_result["iv_source"] == "solved")]
    merged = solved_puts_euro[["strike", "model_iv"]].merge(
        solved_puts_amer[["strike", "model_iv"]], on="strike", suffixes=("_euro", "_amer"))
    assert not merged.empty
    assert (merged["model_iv_amer"] < merged["model_iv_euro"]).all()
