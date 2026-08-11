import datetime as dt

from bscpp.backtest import Backtester, MockProvider, StripPricer


def test_mock_provider_chain_shape():
    provider = MockProvider(spot=100.0, base_vol=0.22)
    expiration = dt.date.today() + dt.timedelta(days=45)
    chain = provider.get_option_chain("TEST", expiration)
    assert not chain.empty
    assert {"strike", "type", "bid", "ask", "last"}.issubset(chain.columns)
    assert set(chain["type"].unique()) <= {"call", "put"}


def test_strip_pricer_prices_mock_chain():
    provider = MockProvider(spot=100.0, base_vol=0.22)
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
    provider = MockProvider(spot=100.0, base_vol=0.22)
    pricer = StripPricer(provider, rate=0.05, mc_paths=5_000)
    expiration = dt.date.today() + dt.timedelta(days=45)
    backtester = Backtester(pricer)

    dates = [dt.date.today(), dt.date.today() - dt.timedelta(days=1)]
    results = backtester.run("TEST", expiration, dates, strike_range=(0.9, 1.1), use_mc=False)
    assert not results.empty

    summary = backtester.summary(results)
    assert set(summary.columns) >= {"as_of", "mean_abs_error", "mean_abs_pct_error", "n_contracts"}
    assert len(summary) == len(dates)
