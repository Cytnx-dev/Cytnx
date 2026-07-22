#!/usr/bin/env python3
"""Generate the committed PEP 561 type stubs for the cytnx extension.

The cytnx Python package is a pybind11 extension (``cytnx.cytnx``) that carries
no Python-level type information, so IDEs and static type checkers cannot
introspect the bindings. This script runs pybind11-stubgen against a built
extension and writes the resulting ``.pyi`` files into ``cytnx/cytnx/`` so they
can be committed to the repository and shipped, unchanged, in every wheel and
conda package.

Stubs are deliberately NOT generated during the package build. Generation has
to import the freshly built, architecture-specific extension, which is
impossible when cross-compiling and brittle in build sandboxes (e.g. conda's
MKL build, where the build prefix may not expose libatomic). A ``.pyi`` file
describes the public API only, so it is independent of platform, architecture,
and CPython version: generating it once on any native build and committing the
result is sufficient for all of them. Keeping the committed stubs honest with
a CI ``mypy.stubtest`` job that compares them against the real runtime module
is planned, not yet wired into CI.

Usage
-----
    # 1. Build the extension (any native CPU preset is fine), e.g.
    cmake --preset openblas-cpu -B build/python
    cmake --build build/python --target pycytnx

    # 2. Install pybind11-stubgen and cytnx's own import-time deps into the
    #    interpreter that runs this script (it imports cytnx.cytnx). The
    #    `dev` extra in pyproject.toml lists both:
    pip install -e .[dev]

    # 3. Regenerate the committed stubs:
    python tools/generate_stubs.py

By default the extension is auto-discovered under ``build/`` for the running
interpreter's ABI; pass ``--extension`` to point at a specific ``.so``/``.pyd``.
Run the script with the lowest supported interpreter (Python 3.10+) so the
emitted syntax stays parseable everywhere the package is installed.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

# The dotted extension module pybind11-stubgen introspects. It is a pybind11
# submodule package, so stubgen emits a directory cytnx/cytnx/ with
# __init__.pyi plus one .pyi per nested submodule.
MODULE = "cytnx.cytnx"
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "cytnx"
STUB_DIR = PACKAGE_DIR / "cytnx"

# Some bindings expose raw C++ types in their docstrings (e.g. `cytnx::Tensor`,
# `boost::intrusive_ptr<...>`). pybind11-stubgen cannot parse these and emits a
# bare `...` in their place, which is invalid as a type annotation and is
# rejected by mypy/pyright/stubtest. Rewrite those `...` annotations to
# `typing.Any` so the committed stubs are valid for every consumer. The fix is
# lossy (the precise return/parameter type is lost for the handful of affected
# bindings); the durable fix is to remove raw C++ types from the binding
# docstrings, after which these substitutions become no-ops.
_RETURN_ELLIPSIS = re.compile(r"-> \.\.\.:")
_ANNOTATION_ELLIPSIS = re.compile(r": \.\.\.([,)])")
# A whole subscript that is just `...`, e.g. `Sequence[...]`, where the element
# type was an unparseable C++ type. `\[\.\.\.\]` cannot match `Callable[..., X]`
# (two elements), so the legitimate `...` in a Callable parameter list is left
# untouched.
_SUBSCRIPT_ELLIPSIS = re.compile(r"\[\.\.\.\]")

# pybind11 3.0.4 renders a cytnx_complex128 parameter as this exact union.
# typeshed's builtin `complex` has no `__complex__` (real-valued dunders like
# `__float__` don't satisfy `SupportsComplex`, which requires `__complex__`
# specifically), so `SupportsComplex` alone rejects a plain Python complex
# literal even though the binding accepts it directly. Add `complex`
# explicitly so the annotation covers the full accepted range.
_SUPPORTS_COMPLEX_UNION = re.compile(
    r"(?<!complex \| )typing\.SupportsComplex \| typing\.SupportsFloat \| typing\.SupportsIndex"
)


def sanitize(text: str) -> str:
    text = _RETURN_ELLIPSIS.sub("-> typing.Any:", text)
    text = _ANNOTATION_ELLIPSIS.sub(r": typing.Any\1", text)
    text = _SUBSCRIPT_ELLIPSIS.sub("[typing.Any]", text)
    text = _SUPPORTS_COMPLEX_UNION.sub(
        "complex | typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex", text
    )
    # Match the repo's pre-commit hooks (trailing-whitespace, end-of-file-fixer)
    # so regeneration stays idempotent and the committed stubs do not get
    # rewritten on the next commit.
    text = "".join(line.rstrip() + "\n" for line in text.splitlines())
    return text


def find_installed_extension() -> Path | None:
    """Return the extension of *this checkout's* editable install of ``cytnx.cytnx``.

    An editable install (``pip install --editable .``) places the freshly built
    extension on the import path, so this matches exactly what the developer
    just compiled without pointing at a build directory. Only trust it when the
    installed ``cytnx`` package actually resolves to this checkout's
    ``cytnx/`` source directory, so an unrelated ``cytnx`` install elsewhere on
    the path (e.g. a released wheel) does not shadow a real local build.
    """
    origin = None
    try:
        spec = importlib.util.find_spec(MODULE)
        parent = sys.modules.get("cytnx")
        if (
            spec is not None
            and spec.origin
            and spec.origin != "built-in"
            and parent is not None
            and str(PACKAGE_DIR) in list(getattr(parent, "__path__", ()))
        ):
            candidate = Path(spec.origin)
            if candidate.is_file():
                origin = candidate
    except (ImportError, ValueError):
        pass

    if origin is None:
        # Resolving "cytnx.cytnx" imports the parent `cytnx` package as a side
        # effect. When that ran cytnx/__init__.py to completion against a real
        # (but ultimately rejected here, e.g. unrelated) extension, its
        # `from .Storage_conti import *`-style imports ran too, and their
        # `@add_method` decorators patched that extension's `Storage`/
        # `Tensor`/... classes; those submodules then stay cached in
        # sys.modules. If main() later imports a distinct copy of the correct
        # extension, it would reuse the cached submodules without rerunning
        # the decorators, silently dropping the patched methods from the
        # regenerated stubs. Drop every "cytnx"/"cytnx.*" entry so nothing
        # from this rejected probe survives. (Skipped when accepted above:
        # the accepted extension is exactly what a later import should reuse,
        # so its already-correct state must be left alone.) This only resets
        # the import system, not pybind11's per-process type registrations;
        # if the rejected extension registered pybind11 types under the same
        # qualified names, a later staged import of a different .so can still
        # raise "generic_type: type ... is already registered!" instead of
        # silently producing wrong stubs.
        for name in [n for n in sys.modules if n == "cytnx" or n.startswith("cytnx.")]:
            del sys.modules[name]

    return origin


def find_build_extension() -> Path | None:
    """Return a built cytnx extension under ``build/`` for this interpreter's ABI."""
    # EXT_SUFFIX is e.g. ".cpython-311-x86_64-linux-gnu.so"; the extension file
    # is named cytnx<EXT_SUFFIX> because the submodule is itself named `cytnx`.
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or importlib.machinery.EXTENSION_SUFFIXES[0]
    candidates = sorted((REPO_ROOT / "build").rglob(f"cytnx{ext_suffix}"))
    if not candidates:
        return None
    # Prefer the most recently modified build if several presets are present.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_extension() -> Path:
    """Locate a built cytnx extension matching the running interpreter's ABI.

    Prefer the installed ``cytnx.cytnx`` (an editable install exposes the
    just-built extension), then fall back to scanning ``build/``.
    """
    extension = find_installed_extension() or find_build_extension()
    if extension is None:
        raise SystemExit(
            "No cytnx extension found. Install cytnx so `cytnx.cytnx` is "
            "importable (e.g. `pip install --editable .`), or build it under "
            f"{REPO_ROOT / 'build'} (e.g. `cmake --build build/python --target "
            "pycytnx`) with the interpreter you are running this script with, "
            "or pass --extension explicitly."
        )
    return extension


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--extension",
        type=Path,
        default=None,
        help="Path to the built cytnx extension (.so/.pyd). Auto-discovered under build/ if omitted.",
    )
    args = parser.parse_args()

    extension = args.extension.resolve() if args.extension else find_extension()
    if not extension.is_file():
        raise SystemExit(f"Extension not found: {extension}")

    try:
        import pybind11_stubgen
    except ImportError:
        raise SystemExit(
            "pybind11-stubgen is not importable. Install the generation deps "
            "(see the `dev` extra in pyproject.toml):\n"
            "    pip install -e .[dev]"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Stage the pure-Python sources next to a copy of the freshly built
        # extension so the dotted name `cytnx.cytnx` resolves to this build,
        # independent of whatever cytnx may be installed in the environment.
        stage = tmp_path / "stage"
        shutil.copytree(PACKAGE_DIR, stage / "cytnx")
        shutil.copy2(extension, stage / "cytnx" / extension.name)

        out = tmp_path / "out"
        print(f"Generating stubs for {MODULE} from {extension}")
        # Drive pybind11-stubgen through its public entry point in this very
        # interpreter rather than re-launching it as `python -m` in a
        # subprocess: the CLI is a thin wrapper around `main()`, and the
        # subprocess only existed to control sys.path. Prepend the stage so
        # `import cytnx.cytnx` resolves to the staged build; the script is a
        # one-shot tool, so leaving the staged package imported is harmless.
        sys.path.insert(0, str(stage))
        try:
            pybind11_stubgen.main([MODULE, "-o", str(out)])
        finally:
            sys.path.remove(str(stage))

        generated = out / "cytnx" / "cytnx"
        if not (generated / "__init__.pyi").is_file():
            raise SystemExit(f"pybind11-stubgen did not produce {generated}/__init__.pyi")

        # Replace the committed stubs wholesale so removed bindings do not leave
        # stale .pyi behind.
        if STUB_DIR.exists():
            shutil.rmtree(STUB_DIR)
        shutil.copytree(generated, STUB_DIR)
        for pyi in STUB_DIR.rglob("*.pyi"):
            pyi.write_text(sanitize(pyi.read_text()))

    written = sorted(p.relative_to(REPO_ROOT) for p in STUB_DIR.rglob("*.pyi"))
    print(f"Wrote {len(written)} stub file(s) to {STUB_DIR.relative_to(REPO_ROOT)}/:")
    for p in written:
        print(f"  {p}")
    print(f"\nReview and commit the changes under {STUB_DIR.relative_to(REPO_ROOT)}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
