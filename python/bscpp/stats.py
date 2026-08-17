"""Honest inference for serially-dependent, overlapping financial data.

Why this module exists: the single most common way quantitative research
in this field invalidates itself is not bad math but bad counterfactuals --
treating overlapping backtest windows as independent observations, quoting
i.i.d. p-values on autocorrelated P&L series, and reporting N=50 when the
effective sample size is 8. (This project's own validation study had
exactly that flaw before this module existed; so did published VRP papers
for years.) The tools here are the standard remedies:

- `effective_sample_size`: N / (1 + 2*sum(autocorrelations)) -- how many
  INDEPENDENT observations your correlated sample is actually worth.
- `newey_west_tstat`: t-statistic for a mean with HAC (heteroskedasticity-
  and autocorrelation-consistent) standard errors (Newey & West 1987),
  with the standard automatic lag choice.
- `stationary_block_bootstrap`: Politis & Romano (1994) stationary
  bootstrap -- resamples blocks of geometrically-distributed length,
  preserving short-range dependence, for confidence intervals on any
  statistic of a dependent series.
- `dependent_correlation_ci`: block-bootstrap CI for a correlation between
  two aligned, serially-dependent series (the honest replacement for
  `scipy.stats.pearsonr`'s i.i.d. p-value on overlapping windows).

Design rule, applied throughout: nothing here returns a bare p-value.
Every result object carries its assumptions (lags used, block length,
effective N) so a report can't quote the number without its caveats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _autocorrelations(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(1, max_lag + 1)])


def effective_sample_size(x, max_lag: int | None = None) -> float:
    """Effective number of independent observations in a dependent series.

    ESS = N / (1 + 2 * sum_{k=1..K} rho_k), truncating the sum at the
    first negative autocorrelation (Geyer's initial-positive-sequence
    heuristic) or at max_lag. For i.i.d. data ESS ~= N; for an AR(1) with
    coefficient phi the theoretical value is N*(1-phi)/(1+phi) -- e.g.
    phi=0.9 leaves ~5% of nominal N. Clipped to [1, N].
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float(n)
    if max_lag is None:
        max_lag = min(n - 2, max(10, int(np.sqrt(n) * 2)))
    rho = _autocorrelations(x, max_lag)
    s = 0.0
    for r in rho:
        if r <= 0.0:
            break
        s += r
    ess = n / (1.0 + 2.0 * s)
    return float(np.clip(ess, 1.0, n))


@dataclass
class HACResult:
    mean: float
    se: float
    tstat: float
    n: int
    n_effective: float
    lags: int

    def __repr__(self) -> str:  # keeps the caveat attached to the number
        return (f"HACResult(mean={self.mean:.6g}, tstat={self.tstat:.3f}, "
                f"se={self.se:.4g}, n={self.n}, n_effective={self.n_effective:.1f}, "
                f"hac_lags={self.lags})")


def newey_west_tstat(x, lags: int | None = None) -> HACResult:
    """HAC t-statistic for the mean of a serially-dependent series.

    Newey-West (1987) long-run variance with Bartlett kernel weights
    w_k = 1 - k/(L+1); automatic lag choice L = floor(4*(n/100)^(2/9))
    (the standard rule of thumb) unless overridden. Reduces to the
    ordinary t-stat when lags=0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 observations")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = min(lags, n - 2)

    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc)) / n
    lrv = gamma0
    for k in range(1, lags + 1):
        gamma_k = float(np.dot(xc[: n - k], xc[k:])) / n
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    lrv = max(lrv, 1e-300)

    se = float(np.sqrt(lrv / n))
    mean = float(x.mean())
    return HACResult(mean=mean, se=se, tstat=mean / se, n=n,
                     n_effective=effective_sample_size(x), lags=lags)


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    level: float
    n_boot: int
    avg_block_len: float
    n: int
    n_effective: float

    def __repr__(self) -> str:
        return (f"BootstrapResult(estimate={self.estimate:.6g}, "
                f"ci=[{self.ci_low:.6g}, {self.ci_high:.6g}] @{self.level:.0%}, "
                f"n={self.n}, n_effective={self.n_effective:.1f}, "
                f"avg_block_len={self.avg_block_len:.1f}, n_boot={self.n_boot})")


def _stationary_bootstrap_indices(n: int, avg_block_len: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """One resample of indices via the Politis-Romano stationary bootstrap.

    Blocks start uniformly at random and continue with probability
    1 - 1/avg_block_len (geometric lengths, mean avg_block_len), wrapping
    circularly. The resulting resampled series is stationary, unlike
    fixed-block schemes.

    Fully vectorized (no per-element Python loop, unlike an earlier
    version of this function): a naive per-index loop cost ~1.7M Python-
    level RNG calls for a single 2000-resample CI on a 420-observation
    series -- the dominant cost of every CI in this project, since
    stationary_block_bootstrap/dependent_correlation_ci call this once
    per resample. Equivalent construction, no explicit loop:

      1. B[i] ~ Bernoulli(p), B[0] forced True -- B[i]=True marks that
         position i STARTS a fresh block (a new uniform-random restart
         point), matching "continue w.p. 1-p, else restart" from the
         original formulation exactly (block lengths are still
         Geometric(p), mean avg_block_len).
      2. block_id[i] = (number of Trues in B[0..i]) - 1 -- which block
         each position belongs to.
      3. Each block gets ONE fresh random start (vectorized, one draw per
         block instead of one per position that restarts).
      4. idx[i] = (that block's start + i's offset from its block's first
         position) mod n -- circular continuation, exactly as before.

    Verified (test_stats.py) to preserve the statistical properties this
    function's callers depend on (block-bootstrap CI coverage on i.i.d.
    data, wider CIs under positive dependence) -- not just assumed
    equivalent from the derivation above.
    """
    n = int(n)
    p = 1.0 / max(avg_block_len, 1.0)
    starts_new_block = rng.random(n) < p
    starts_new_block[0] = True

    block_id = np.cumsum(starts_new_block) - 1
    block_first_pos = np.flatnonzero(starts_new_block)
    n_blocks = block_first_pos.size

    block_starts = rng.integers(0, n, size=n_blocks)
    offset_within_block = np.arange(n) - block_first_pos[block_id]
    idx = (block_starts[block_id] + offset_within_block) % n
    return idx.astype(np.int64)


def cluster_bootstrap_indices(cluster_row_idx: list[np.ndarray], avg_block_len: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Row indices (into the ORIGINAL data) for one stationary-bootstrap
    resample at the CLUSTER level, not the observation level.

    For panel data where units sharing a cluster id (e.g. the same
    calendar period, across many co-moving tickers) are contemporaneously
    dependent -- not just serially dependent within one unit's own time
    series -- a plain block bootstrap over a flattened, single-axis-
    ordered array is blind to that dependence: if the array is ordered
    (ticker, date) and blocked along that concatenation, same-date rows
    from DIFFERENT tickers can sit many lags apart, invisible to a block
    short enough to capture genuine serial dependence. This resamples
    whole clusters together instead -- applying the SAME Politis-Romano
    scheme (geometric block lengths of CONSECUTIVE clusters, mean
    avg_block_len) one level up, via _stationary_bootstrap_indices on the
    cluster axis, then expanding each drawn cluster back out to all of
    its member rows (with repeats when a cluster is drawn more than once,
    exactly as with-replacement cluster resampling should behave).

    cluster_row_idx: cluster_row_idx[k] = the row indices (into the
    original data) belonging to cluster k, for clusters 0..n_clusters-1
    in a fixed, caller-defined order (typically sorted by the cluster
    key, e.g. calendar date) -- precompute this ONCE outside any
    resampling loop; only the (cheap) cluster-level index draw and the
    concatenation below repeat per resample.
    """
    n_clusters = len(cluster_row_idx)
    drawn_clusters = _stationary_bootstrap_indices(n_clusters, avg_block_len, rng)
    return np.concatenate([cluster_row_idx[k] for k in drawn_clusters])


def _default_block_len(x: np.ndarray) -> float:
    # Politis-White-style order: block length ~ n^(1/3), floored at 2 --
    # a robust default when the user has no better prior on the dependence
    # length (e.g. set it to the window overlap for rolling-window studies).
    return max(2.0, len(x) ** (1.0 / 3.0))


def stationary_block_bootstrap(x, stat_fn=None, n_boot: int = 2000,
                                avg_block_len: float | None = None,
                                level: float = 0.95,
                                seed: int | None = 0) -> BootstrapResult:
    """Percentile CI for stat_fn(x) under the stationary block bootstrap.

    stat_fn: maps a 1-D array to a scalar; defaults to the mean. For
    rolling-window studies set avg_block_len to at least the number of
    overlapping steps between consecutive observations.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        raise ValueError("need at least 5 observations")
    if stat_fn is None:
        stat_fn = np.mean
    if avg_block_len is None:
        avg_block_len = _default_block_len(x)

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(x[_stationary_bootstrap_indices(n, avg_block_len, rng)])

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return BootstrapResult(estimate=float(stat_fn(x)), ci_low=float(lo), ci_high=float(hi),
                           level=level, n_boot=n_boot, avg_block_len=float(avg_block_len),
                           n=n, n_effective=effective_sample_size(x))


def dependent_correlation_ci(x, y, n_boot: int = 2000,
                              avg_block_len: float | None = None,
                              level: float = 0.95,
                              seed: int | None = 0) -> BootstrapResult:
    """Block-bootstrap CI for corr(x, y) between two aligned dependent series.

    The honest replacement for pearsonr's i.i.d. p-value on overlapping-
    window data: pairs (x_i, y_i) are resampled in BLOCKS (same indices for
    both series, preserving both the cross-correlation and each series'
    serial dependence). If the CI excludes 0 you have block-robust evidence
    of association; the reported n_effective (of x) tells you how many
    independent observations that evidence is really based on.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    n = len(x)
    if n < 8:
        raise ValueError("need at least 8 paired observations")
    if avg_block_len is None:
        avg_block_len = _default_block_len(x)

    def corr_at(idx: np.ndarray) -> float:
        xs, ys = x[idx], y[idx]
        sx, sy = xs.std(), ys.std()
        if sx == 0.0 or sy == 0.0:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = corr_at(_stationary_bootstrap_indices(n, avg_block_len, rng))

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    full = float(np.corrcoef(x, y)[0, 1])
    return BootstrapResult(estimate=full, ci_low=float(lo), ci_high=float(hi),
                           level=level, n_boot=n_boot, avg_block_len=float(avg_block_len),
                           n=n, n_effective=effective_sample_size(x))
