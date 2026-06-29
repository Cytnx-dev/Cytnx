#!/usr/bin/env python3
"""Stamp pyproject.toml and emit the build version tag for a nightly release.

The script

  1. reads `MAJOR.MINOR.PATCH` from version.cmake using the same regex
     that the `[tool.scikit-build.metadata.version]` block of
     pyproject.toml uses, so there is only one source of truth for how
     the version is parsed;
  2. derives a PEP 440 dev version of the form
     `MAJOR.MINOR.PATCH.devYYYYMMDDHHMM` (UTC stamp). When the
     `CYTNX_VERSION_TAG` environment variable is already set and
     non-empty, its value is used as the dev suffix verbatim instead
     of generating a fresh one. This lets the surrounding CI job
     compute the tag once and supply it to every matrix build so all
     wheels for a workflow run carry the same version, even when the
     matrix jobs cross a minute boundary;
  3. rewrites pyproject.toml in place so cibuildwheel produces wheels
     for the `cytnx` PyPI project with that static dev version
     (`pip install --pre cytnx` will pick up the nightly; `pip install
     cytnx` continues to install the latest stable, mirroring the
     numpy / scipy convention); and
  4. appends `CYTNX_VERSION_TAG=.devYYYYMMDDHHMM` to `$GITHUB_ENV` so
     downstream steps that were not given the tag explicitly can
     still forward it into CMake. The C++ compile definition
     `CYTNX_VERSION` (which becomes `cytnx.__version__`) is built
     from the numeric CMake version plus this tag, keeping
     `cytnx.__version__` aligned with the wheel filename.

This is intended to run inside a fresh CI checkout before cibuildwheel.
It mutates the working tree and is not idempotent.

Requires `tomlkit` (declared in pyproject.toml's `release-tools`
dependency-group) so the rewrite preserves comments and formatting on
round-trip.
"""

import datetime
import os
import pathlib
import re
import sys

import tomlkit

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_CMAKE = REPO_ROOT / "version.cmake"


def load_version_regex(doc: tomlkit.TOMLDocument) -> re.Pattern[str]:
    pattern = doc["tool"]["scikit-build"]["metadata"]["version"]["regex"]
    return re.compile(str(pattern).strip())


def read_base_version(version_re: re.Pattern[str]) -> str:
    text = VERSION_CMAKE.read_text()
    match = version_re.search(text)
    if not match:
        sys.exit(f"could not parse MAJOR/MINOR/PATCH from {VERSION_CMAKE}")
    return f"{match['major']}.{match['minor']}.{match['patch']}"


def build_dev_tag() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M")
    return f".dev{stamp}"


def resolve_dev_tag() -> str:
    # Prefer a caller-supplied tag so concurrent invocations agree on
    # one version; fall back to a freshly-generated UTC stamp for
    # standalone use.
    return os.environ.get("CYTNX_VERSION_TAG") or build_dev_tag()


def rewrite_pyproject(doc: tomlkit.TOMLDocument, version: str) -> None:
    project = doc["project"]

    dynamic = list(project.get("dynamic", []))
    if "version" not in dynamic:
        sys.exit('expected "version" in [project].dynamic in pyproject.toml')
    dynamic.remove("version")
    if dynamic:
        project["dynamic"] = dynamic
    else:
        del project["dynamic"]
    project["version"] = version

    skb_metadata = doc["tool"]["scikit-build"]["metadata"]
    if "version" not in skb_metadata:
        sys.exit("expected [tool.scikit-build.metadata.version] in pyproject.toml")
    del skb_metadata["version"]

    PYPROJECT.write_text(tomlkit.dumps(doc))


def emit_github_env(tag: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        # Outside Actions; print so the operator can pass it manually.
        print(f"CYTNX_VERSION_TAG={tag}")
        return
    with open(github_env, "a", encoding="utf-8") as f:
        f.write(f"CYTNX_VERSION_TAG={tag}\n")


def main() -> None:
    doc = tomlkit.parse(PYPROJECT.read_text())
    version_re = load_version_regex(doc)
    base = read_base_version(version_re)
    tag = resolve_dev_tag()
    version = f"{base}{tag}"

    rewrite_pyproject(doc, version)
    emit_github_env(tag)

    print(f"stamped pyproject.toml: cytnx=={version}")


if __name__ == "__main__":
    main()
