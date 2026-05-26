# Cytnx_doc
The documentation of the Cytnx tensor network library


# Compilation instructions
Use uv (or other python package management system that support pyproject.toml)


# Build locally for preview
Build the HTML user guide into `build/html/`:

```bash
uv sync
uv run sphinx-build -M html source build
```

`sphinx-build` accepts any `-M` mode supported by Sphinx, for example:

```bash
uv run sphinx-build -M latexpdf source build  # PDF via LaTeX
uv run sphinx-build -M help     source build  # full list of -M targets
uv run sphinx-multiversion source build       # one build per git ref, output in build/<ref>/
```


If you want to run the unit test in the documentation, see [here](./tests/README.md).
