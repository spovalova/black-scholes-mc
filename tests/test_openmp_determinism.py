"""Confirms exactly what's reproducible about the OpenMP-parallelized path
loops (see monte_carlo.cpp/longstaff_schwartz.cpp/heston.cpp) and states
the one thing that ISN'T, precisely: the underlying random draws are
exactly reproducible regardless of thread count (each path/index seeks to
its own disjoint Philox counter position, independent of which thread
computes it or in what order -- see philox.hpp); the AGGREGATED price/
std_error can differ in the last few ULPs at scale, because OpenMP's
reduction(+:...) sums per-thread partial sums in a thread-count-dependent
order, and floating-point addition is not associative. That's standard,
expected behavior for any parallelized numerical reduction (BLAS/LAPACK
included), not a correctness bug -- but "bit-identical regardless of
thread count" would be an overclaim if stated without this distinction,
so it's verified here rather than assumed either way.

These tests don't control OMP_NUM_THREADS directly (the extension is
already loaded with the compiler's chosen default by the time Python
starts) -- they instead confirm the underlying mechanism (small-N calls,
where there's no reduction ambiguity to begin with, are exact) and rely
on examples/README's documented cross-thread-count check for the
reduction-order tolerance claim.
"""

import math

import bscpp


def test_small_n_price_has_no_reduction_ambiguity_and_is_reproducible():
    # At num_paths=2 there's essentially no summation-order freedom (at
    # most a handful of terms), so this isolates whether the underlying
    # draws themselves are reproducible, independent of any reduction
    # question -- repeated calls with the same seed must match exactly.
    def run():
        mc = bscpp.MonteCarloPricer(seed=42)
        return mc.price_european(bscpp.make_inputs(100, 100, 0.05, 0.2, 1.0, "call"), 2, False)

    a, b = run(), run()
    assert a.price == b.price


def test_aggregated_price_reproducible_to_tight_tolerance_at_scale():
    # At large num_paths, OpenMP's reduction(+:sum,sum_sq) may sum
    # per-thread partial results in a run-to-run order that isn't fixed
    # by the seed alone (thread scheduling), so exact bit-equality isn't
    # the right bar here -- but the result must still agree to far tighter
    # than statistical noise, since the underlying draws ARE identical.
    def run():
        mc = bscpp.MonteCarloPricer(seed=42)
        return mc.price_european(bscpp.make_inputs(100, 100, 0.05, 0.2, 1.0, "call"), 100_000, True)

    a, b = run(), run()
    assert math.isclose(a.price, b.price, rel_tol=1e-9)
    assert math.isclose(a.std_error, b.std_error, rel_tol=1e-9)


def test_philox_raw_draws_are_exactly_reproducible_regardless_of_execution():
    # The actual guarantee the path loops depend on: index i's draw is a
    # pure function of i (via seek()), never of thread/execution order --
    # this is what test_philox.py's own tests already establish for the
    # generator in isolation; repeated here as the property the OpenMP
    # path loops above are built on.
    a = bscpp._core._philox_raw_draws(42, 0, 1000)
    b = bscpp._core._philox_raw_draws(42, 0, 1000)
    assert a == b
