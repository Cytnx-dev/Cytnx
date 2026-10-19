"""Version label of the current checkout, shared by the documentation builds.

``version.cmake`` is the single source of truth for the release version: it
feeds the PyPI/conda package version (via scikit-build-core),
``cytnx.__version__``, the Sphinx user guide and the Doxygen API reference.
The documentation label adds git information when the checkout is not a
release, so the ``dev`` docs built from ``master`` say which commit they
describe:

* on the release tag ``vMAJOR.MINOR.PATCH``:        ``1.1.1``
* elsewhere, e.g. the ``dev`` docs built from master: ``1.1.1.dev580+g393c5c7``
  (580 commits after the last release tag, at commit 393c5c7)
* without git metadata, e.g. a source tarball:        ``1.1.1``
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_CMAKE = REPO_ROOT / "version.cmake"


def release_version() -> str:
    """Return ``MAJOR.MINOR.PATCH`` from version.cmake.

    The pattern mirrors the one scikit-build-core uses in pyproject.toml's
    ``[tool.scikit-build.metadata.version]`` block, so the docs accept exactly
    the version.cmake the package build accepts.
    """
    text = VERSION_CMAKE.read_text()
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"set\(\w+?VERSION_{field}\s+(\d+)\)", text)
        if not match:
            raise ValueError(f"could not parse {field} version from {VERSION_CMAKE}")
        parts.append(match.group(1))
    return ".".join(parts)


def version_label() -> str:
    """Return the release version, with a ``.devN+g<sha>`` suffix off a release tag."""
    version = release_version()
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "v[0-9]*"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return version
    match = re.fullmatch(r"(v\S+)-(\d+)-g([0-9a-f]+)", described)
    if not match:
        return version
    tag, distance, commit = match.groups()
    if tag == f"v{version}" and distance == "0":
        return version
    return f"{version}.dev{distance}+g{commit}"


if __name__ == "__main__":
    print(version_label())
