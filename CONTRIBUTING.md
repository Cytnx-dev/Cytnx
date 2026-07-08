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

## Regenerating the Python type stubs

The `cytnx` Python package ships PEP 561 type stubs (`cytnx/cytnx/*.pyi`)
committed to the repository and shipped unchanged in every wheel and conda
package. They are generated from the built `cytnx.cytnx` pybind11 extension
by `tools/generate_stubs.py`, not written by hand.

**If your change touches any `pybind/*.cpp` binding** — a new function, a
changed signature, a different default value, a different overload set — the
committed stubs go stale and must be regenerated as part of the same PR.

Stub generation is only reproducible when the tools that produce it are
pinned, since both silently change the emitted annotations across versions:

- `pybind11 ==3.0.4` — controls the type annotations baked into the compiled
  extension (`[build-system].requires` in `pyproject.toml`).
- `pybind11-stubgen ==2.5.5` — walks the built extension and renders the
  `.pyi` files (`dev` extra in `pyproject.toml`).

Do not bump either version incidentally; if a bump is needed, do it
deliberately in its own commit and regenerate the stubs in the same PR so the
diff shows exactly what the version change affected.

To regenerate:

1. Build the extension and install the pinned dev tools together, through
   the editable install (not a direct `cmake` configure/build — that
   bypasses `[build-system].requires` and can silently pick up an
   unpinned `pybind11`, since `CMakeLists.txt` accepts any installed
   `pybind11 >= 3.0.0` and falls back to `FetchContent`-ing `v3.0.1` if
   none is found):
   ```sh
   pip install --editable '.[dev]' --config-settings=build-dir=build/python
   ```
2. Regenerate the committed stubs:
   ```sh
   python tools/generate_stubs.py
   ```
   The extension is auto-discovered under `build/` for the running
   interpreter's ABI; pass `--extension` to point at a specific `.so`/`.pyd`
   if more than one build is present. Run this with the lowest supported
   interpreter (Python 3.10) so the emitted syntax stays parseable
   everywhere the package is installed.
3. Review the diff under `cytnx/cytnx/*.pyi` and commit it alongside the
   binding change that caused it.

`mypy.stubtest` compares the committed stubs against the live runtime module
and catches mismatches (missing members, incompatible defaults, overloads
that can never match). It is not yet wired into CI, so run it manually after
regenerating:

```sh
python -m mypy.stubtest cytnx.cytnx
```

