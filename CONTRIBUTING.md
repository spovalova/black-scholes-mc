# Contributing to bscpp

This started as a personal project to learn quantitative finance from the
ground up -- pricing models, Greeks, vol surfaces, hedging P&L -- by
building them rather than just reading about them. It's not run by a team
and there's no roadmap committee, but issues and pull requests are welcome.

A few things that'll make a PR easier to review, in the spirit of how this
project has been built so far:

- **Verify against something you didn't write.** A formula or algorithm
  should be checked against the published paper, a production
  implementation (QuantLib is the usual reference here), or an independent
  method (e.g. an analytic result cross-checked by Monte Carlo) -- not
  just against its own tests, which can pass while encoding the same
  mistake twice.
- **Precondition your criteria, not just your formulas.** If a check is
  only meaningful under some condition (e.g. `w(k) > 0` for the SVI
  arbitrage criterion), enforce it explicitly rather than assuming the
  input will satisfy it.
- **State what you didn't test.** A stress test or benchmark that covers
  most of the input space is more useful documented honestly (with the
  gap named) than implied to be exhaustive.
- **Regression tests reproduce the bug, not just the fix.** Where
  practical, show the test fails against the old behavior and passes
  against the new one.

Commit messages: a one-line imperative summary is fine for mechanical
changes. A real bug fix is worth a body that states what was wrong, what
it changes numerically, and why it wasn't caught before -- that context is
what makes the commit useful to read later, not just the diff.

Run the fast test suite before opening a PR:

```bash
pip install -e .
pytest tests/ -q -m "not slow"
```

Thanks for reading this far.
