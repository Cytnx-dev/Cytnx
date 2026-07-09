---
name: google-python-style
description: >-
  Enforce the Google Python Style Guide when writing, reviewing, or refactoring
  Python code in Cytnx — the `cytnx/` package, `pytests/`, `tools/`, and
  `example/`. Use when creating new .py files or symbols, naming, writing
  docstrings or type hints, structuring imports, or reviewing a Python diff for
  style. Covers naming, imports, Google-format docstrings, type annotations,
  error handling, and doctests.
---

# Google Python style (as adopted by Cytnx)

Cytnx's Python surface — the `cytnx/` package that wraps the compiled extension,
`pytests/`, `tools/`, and `example/` — follows the **Google Python Style Guide**
(<https://google.github.io/styleguide/pyguide.html>), which builds on PEP 8.
This skill distills the rules that matter for review.

Match the file you are editing when it is already consistent; apply the rules
below to new files and new code. Do not reformat unrelated code as churn.

## Naming

- **Modules / packages**: `lower_snake_case`.
- **Classes / exceptions / type vars**: `PascalCase` (`CapWords`).
- **Functions / methods / variables / arguments**: `lower_snake_case`.
- **Module-level constants**: `CONSTANT_CASE` (`MAX_BOND_DIM`).
- **Internal names**: a **single leading underscore** (`_helper`) marks module-
  or class-internal API; `__double_leading` only for genuine name mangling. Use a
  trailing underscore solely to dodge a keyword clash (`class_`, `id_`).
- Avoid single-character names except loop counters/iterators; never `l`, `O`,
  `I`.

## Imports

- **Import modules/packages, not individual symbols** — `from cytnx import
  linalg`, then call `linalg.Svd(...)`. Google style qualifies at the call site.
  (Typing symbols are the usual exception.)
- One import per line; grouped and ordered: (1) `__future__`, (2) standard
  library, (3) third-party, (4) Cytnx / first-party — alphabetical within a
  group.
- No wildcard `from x import *`.

## Docstrings — Google format

Every public module, class, and function gets a docstring: triple double-quotes,
an imperative one-line summary first, then the **Google sections** (`Args:`,
`Returns:`, `Raises:`, `Yields:`).

```python
def truncate(tensor, max_dim):
    """Truncate the bond to at most ``max_dim`` singular values.

    Args:
        tensor: The UniTensor to truncate.
        max_dim: Maximum bond dimension to keep; must be positive.

    Returns:
        A new UniTensor with the truncated bond.

    Raises:
        CytnxError: If ``max_dim`` is not positive.
    """
```

## Type annotations

- Annotate public function signatures (PEP 484); prefer built-in generics
  (`list[int]`, `dict[str, int]`).
- Do not annotate `self`/`cls`. For heavy or optional types use
  `from __future__ import annotations` (string-form annotations).

## Structure & correctness

- **4-space indent, no tabs.** Follow Google's 80-column guideline.
- Use `is` / `is not` for `None` — never `== None`.
- Prefer f-strings; avoid `%` and `.format` in new code.
- Catch **specific** exceptions, never bare `except:`; raise `cytnx.CytnxError`
  (the type surfaced from the C++ `cytnx_error_msg`) where a Cytnx error is the
  right kind.
- Use context managers (`with`) for resources; use comprehensions only for
  simple, side-effect-free transforms.
- **Doctests run in CI**: `pytest pytests/ --doctest-modules`. Keep docstring
  examples correct and deterministic.

## Formatting

There is **no Python autoformatter** wired into `pre-commit` (only clang-format
for C/C++/CUDA, plus generic trailing-whitespace / end-of-file fixers). So keep
whitespace, quotes, and wrapping PEP 8 / Google-clean **by hand** — don't rely on
a formatter to fix it. Still, review substance (naming, imports, docstrings,
typing, correctness) over pure layout nits.

## Review checklist

- [ ] snake_case functions/vars, `PascalCase` classes, `CONSTANT_CASE` constants.
- [ ] Modules imported and qualified; grouped/ordered; no wildcard imports.
- [ ] Public symbols carry Google-format docstrings (`Args`/`Returns`/`Raises`).
- [ ] Public signatures annotated; `None` compared with `is`.
- [ ] Specific exceptions; `CytnxError` where appropriate.
- [ ] Doctests pass under `--doctest-modules`.
