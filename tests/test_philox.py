"""Cross-validates bscpp's Philox4x64-10 against numpy.random.Philox --
the actual portability claim (see cpp/include/bscpp/philox.hpp): this
isn't just internally self-consistent, it reproduces an independent,
already-portable reference implementation exactly.
"""

import numpy as np
import pytest

import bscpp._core as core


def _numpy_philox_raw(key: int, n: int) -> list:
    bg = np.random.Philox(key=key)
    return [int(x) for x in bg.random_raw(n)]


@pytest.mark.parametrize("seed", [0, 1, 42, 12345, 2**32 - 1, 2**63 - 1, 0xDEADBEEF])
def test_philox_matches_numpy_stream_zero(seed):
    # bscpp maps (seed, stream=0) -> numpy's Philox(key=[seed, 0]),
    # counter starting at [0,0,0,0]: identical construction, so the raw
    # output streams must match bit-for-bit, not just "look random."
    bscpp_draws = core._philox_raw_draws(seed, 0, 16)
    numpy_draws = _numpy_philox_raw(seed, 16)
    assert bscpp_draws == numpy_draws


def test_philox_different_streams_from_same_seed_are_independent():
    # The whole point of the stream parameter: two streams from the same
    # seed must not just "differ" but land in genuinely disjoint counter
    # ranges (stream maps to counter word 1, so streams 0 and 1 are 2^64
    # blocks apart -- overlap is not just unlikely, it's astronomically
    # far from where either stream will ever actually draw).
    stream0 = core._philox_raw_draws(42, 0, 1000)
    stream1 = core._philox_raw_draws(42, 1, 1000)
    assert len(set(stream0) & set(stream1)) == 0


def test_philox_seek_to_initial_position_matches_a_fresh_generator():
    # seek(0,0,0,0) resets to exactly the state a freshly-constructed
    # generator starts in (counter=[0,0,0,0], same seed/stream) -- the
    # baseline correctness check before trusting seek() for anything else.
    fresh = core._philox_raw_draws(7, 0, 12)
    seeked = core._philox_seek_draws(7, 0, 0, 0, 0, 12)
    assert seeked == fresh


def test_philox_seek_to_distinct_counters_gives_disjoint_streams():
    # The actual property the future OpenMP parallelization depends on:
    # two threads seeking to different counters get streams that don't
    # overlap, so no coordination is needed between them.
    a = core._philox_seek_draws(7, 0, 0, 0, 0, 500)
    b = core._philox_seek_draws(7, 1_000_000, 0, 0, 0, 500)
    assert len(set(a) & set(b)) == 0


def test_philox_reproducible_within_and_across_calls():
    a = core._philox_raw_draws(999, 0, 50)
    b = core._philox_raw_draws(999, 0, 50)
    assert a == b
