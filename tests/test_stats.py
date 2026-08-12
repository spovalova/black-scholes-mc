import numpy as np
import pytest

from bscpp.stats import (
    dependent_correlation_ci,
    effective_sample_size,
    newey_west_tstat,
    stationary_block_bootstrap,
)


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def test_effective_sample_size_iid_close_to_n():
    x = np.random.default_rng(1).normal(size=2000)
    ess = effective_sample_size(x)
    assert 0.8 * len(x) <= ess <= len(x)


def test_effective_sample_size_ar1_matches_theory():
    # AR(1) with phi: theoretical ESS/N = (1-phi)/(1+phi). phi=0.9 -> ~5.3%.
    phi = 0.9
    x = _ar1(20_000, phi, seed=2)
    ess_frac = effective_sample_size(x) / len(x)
    theory = (1 - phi) / (1 + phi)
    assert 0.5 * theory < ess_frac < 2.0 * theory
    # and it must be drastically below 1 -- the whole point
    assert ess_frac < 0.15


def test_newey_west_reduces_to_ordinary_tstat_for_iid():
    x = np.random.default_rng(3).normal(loc=0.5, size=500)
    hac = newey_west_tstat(x)
    ordinary = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert abs(hac.tstat - ordinary) / abs(ordinary) < 0.15


def test_newey_west_widens_se_under_positive_autocorrelation():
    x = _ar1(3000, 0.8, seed=4) + 0.1
    hac = newey_west_tstat(x)
    naive_se = x.std(ddof=0) / np.sqrt(len(x))
    assert hac.se > 1.5 * naive_se  # HAC must penalize dependence, hard


def test_block_bootstrap_ci_covers_iid_mean():
    # Coverage check: on i.i.d. data the 95% CI for the mean should cover
    # the true mean in roughly 95% of repetitions. 150 reps is enough to
    # distinguish ~0.95 from broken (e.g. <0.8) without being slow.
    rng = np.random.default_rng(5)
    hits = 0
    reps = 150
    for r in range(reps):
        x = rng.normal(loc=1.0, size=200)
        res = stationary_block_bootstrap(x, n_boot=400, seed=r)
        hits += res.ci_low <= 1.0 <= res.ci_high
    assert hits / reps > 0.85


def test_block_bootstrap_wider_ci_for_dependent_data():
    # Same marginal variance, but AR(1) dependence: the block bootstrap
    # (with adequate block length) must produce a WIDER CI than it does
    # for i.i.d. data of the same length -- that width is the honesty.
    n = 800
    iid = np.random.default_rng(6).normal(size=n)
    dep = _ar1(n, 0.8, seed=7)
    dep = dep / dep.std() * iid.std()

    w_iid = (lambda r: r.ci_high - r.ci_low)(
        stationary_block_bootstrap(iid, n_boot=600, avg_block_len=10, seed=0))
    w_dep = (lambda r: r.ci_high - r.ci_low)(
        stationary_block_bootstrap(dep, n_boot=600, avg_block_len=10, seed=0))
    assert w_dep > 1.5 * w_iid


def test_dependent_correlation_ci_detects_real_association():
    rng = np.random.default_rng(8)
    n = 300
    x = _ar1(n, 0.5, seed=9)
    y = 0.8 * x + rng.normal(scale=0.5, size=n)  # genuinely associated
    res = dependent_correlation_ci(x, y, n_boot=600, avg_block_len=5)
    assert res.estimate > 0.5
    assert res.ci_low > 0.0  # CI excludes zero: block-robust evidence


def test_dependent_correlation_ci_no_false_certainty_on_noise():
    # Two INDEPENDENT AR(1) series: spurious sample correlation is common,
    # and the block-bootstrap CI should usually include 0. Check over a few
    # seeds that the CI includes zero in the clear majority of cases.
    includes_zero = 0
    trials = 10
    for s in range(trials):
        x = _ar1(200, 0.7, seed=100 + s)
        y = _ar1(200, 0.7, seed=200 + s)
        res = dependent_correlation_ci(x, y, n_boot=400, avg_block_len=8, seed=s)
        includes_zero += res.ci_low <= 0.0 <= res.ci_high
    assert includes_zero >= 7


def test_result_reprs_carry_caveats():
    # Design rule: the numbers can't be quoted without their assumptions.
    x = np.random.default_rng(10).normal(size=100)
    assert "n_effective" in repr(newey_west_tstat(x))
    assert "n_effective" in repr(stationary_block_bootstrap(x, n_boot=100))


def test_input_validation():
    with pytest.raises(ValueError):
        newey_west_tstat([1.0, 2.0])
    with pytest.raises(ValueError):
        stationary_block_bootstrap([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        dependent_correlation_ci([1.0] * 5, [1.0] * 5)
