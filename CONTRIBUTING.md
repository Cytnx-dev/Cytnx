# Contributing to Cytnx

Thanks for considering a contribution. This guide covers the housekeeping
that is easy to miss when a change touches metadata shared across the
build, packaging, and docs. For build/test instructions, see
[Readme.md](Readme.md). Maintainers cutting a tagged release should follow
[RELEASING.md](RELEASING.md) instead.

## Updating the minimum supported Python version

The minimum Python version is declared independently in a few places, since
none of them can import it from a single source:

- **`pyproject.toml`** — `requires-python` under `[project]`. This is the
  version cibuildwheel reads to decide which interpreters to build PyPI
  wheels for.
- **`conda_build/conda_build_config.yaml`** — the `python:` list. This file
  is plain YAML and is read by conda-build *before* `meta.yaml` is rendered,
  so it cannot import the bound from `pyproject.toml` the way `meta.yaml`
  does for the package version.
- **`docs/source/adv_install.rst`** — the `python >= X.Y` requirement.

When raising (or lowering) the minimum, update all three. There is no
automated check for this — please grep for the old version string (e.g.
`3.9`) across the repository before opening the PR to catch anything missed.

