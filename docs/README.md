# Cytnx_doc
The documentation of the Cytnx tensor network library


# Compilation instructions
Documentation dependencies live in the root `pyproject.toml` as the `docs`
optional group. Use `uv` (recommended) or `pip` to install them.


# Build locally for preview
Build the HTML user guide into `docs/build/html/`. Run these commands from the
repository root.

With `uv` (recommended):

```bash
uv sync --extra docs
uv run --no-sync sphinx-build -M html docs/source docs/build
```

With `pip`:

```bash
pip install -e ".[docs]"
sphinx-build -M html docs/source docs/build
```

`sphinx-build` accepts any `-M` mode supported by Sphinx, for example:

```bash
uv run --no-sync sphinx-build -M latexpdf docs/source docs/build  # PDF via LaTeX
uv run --no-sync sphinx-build -M help     docs/source docs/build  # full list of -M targets
uv run --no-sync sphinx-multiversion docs/source docs/build       # one build per git ref, output in docs/build/<ref>/
```

The commands above also build the cytnx C++ extension, which is not actually
required to render the user guide. To install only the doc dependencies and
skip the C++ build, pass `--no-install-project` to `uv sync` (or set up a
separate `pip` virtual environment):

```bash
uv sync --extra docs --no-install-project
uv run --no-sync sphinx-build -M html docs/source docs/build
```


If you want to run the unit test in the documentation, see [here](./tests/README.md).
