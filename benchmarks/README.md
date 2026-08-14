# External benchmarks

Compares this project's pricers against two established, independent
references on identical inputs: [QuantLib](https://www.quantlib.org/)
(analytic BS, CRR binomial American, analytic Heston) and
[vollib](http://vollib.org/) (analytic BS/BSM price and implied vol).

This exists because "3.1x faster than my own previous version"
(`heston_price_batch`'s own CHANGELOG entry) is a self-referential claim
-- it says nothing about whether the *result* is fast in any absolute
sense, only that it improved. These benchmarks are the answer to "fast
compared to what."

**Not run as part of the normal test suite.** QuantLib and vollib aren't
core dependencies (they're a comparison harness, not something this
project's own pricers depend on), and wall-clock timings are inherently
noisy/hardware-dependent -- not something to assert pass/fail on in CI.
Run explicitly:

```
pip install -e ".[benchmark]"
pytest benchmarks/ --benchmark-only --benchmark-columns=min,mean,stddev,rounds
```

## Results

Hardware and exact numbers are in the README's "External benchmarks"
section (not duplicated here, to avoid two sources of truth going stale
independently) -- re-run the command above and update both if the
numbers materially change.

## Correctness, not just speed

Every benchmark file asserts agreement with the reference (QuantLib/
vollib) within a stated tolerance before timing anything -- a fast wrong
answer isn't a result. See each file's `test_*_correctness` function.
