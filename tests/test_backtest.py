import datetime as dt

from bscpp.backtest import Backtester, MockProvider, StripPricer


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
    # theoretical prices should be positive and in a sane range vs spot
    assert (result["bs_price"] >= 0).all()
    # BS and MC should roughly agree per-contract
    diff = (result["bs_price"] - result["mc_price"]).abs()
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
