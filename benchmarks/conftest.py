"""Shared helper for the external benchmark suite (see benchmarks/README.md)."""


def ql_force_recompute(option, quote, base_value):
    """Wraps a QuantLib option's NPV() call so pytest-benchmark times a
    genuine recomputation on every call, not a cache hit.

    QuantLib's LazyObject model caches NPV() until a watched quote
    changes -- confirmed by direct measurement, ~6x faster (241ns vs
    1440ns) for a cached call vs. a genuinely recomputed one on a plain
    European option. Perturbing `quote` by an alternating +-1e-10 forces
    real recomputation every call while leaving the priced value
    unchanged to any tolerance that matters -- the same thing a
    calibration loop repricing against a moving market quote would
    trigger naturally, so this is the fair comparison, not a workaround.
    """
    toggle = [True]

    def call():
        toggle[0] = not toggle[0]
        quote.setValue(base_value + (1e-10 if toggle[0] else -1e-10))
        return option.NPV()

    return call
