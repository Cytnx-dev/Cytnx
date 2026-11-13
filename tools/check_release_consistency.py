#!/usr/bin/env python3
"""Check that the release version metadata is internally consistent.

A Cytnx release is described by four pieces of metadata that must agree;
when they drift, the published artifacts contradict each other:

  * ``version.cmake`` carries the numeric ``MAJOR.MINOR.PATCH`` that
    scikit-build-core stamps onto the PyPI and conda packages and onto
    ``cytnx.__version__``. The release tag itself is *not* the source of
    the package version.
  * ``docs/site_root/versions.json`` lists the documentation slugs served
    from the ``gh-pages`` branch. The Sphinx version switcher and the
    landing-page redirect build URLs as ``<site_root>/<slug>/``, so every
    slug must name a directory that ``.github/workflows/docs.yml`` really
    publishes.
  * the ``vMAJOR.MINOR.PATCH`` git tag triggers the release workflows.
    ``docs.yml`` deploys the HTML to a directory named after the tag with
    the leading ``v`` stripped (``v1.1.0`` -> ``gh-pages/1.1.0/``), so the
    matching ``versions.json`` slug must be ``1.1.0``, never ``v1.1.0``.
  * ``CITATION.cff`` carries the software ``version`` and the
    ``date-released`` for it. GitHub's "Cite this repository" widget and
    every downstream citation tool read these, so a stale ``version`` tells
    users to cite a release that is not the one they installed. No build
    step reads this file, which is exactly how it sat at ``1.0.0`` through
    three later releases.

Checks performed:

  1. Every numbered-release slug in ``versions.json`` is numeric (no
     leading ``v``). A ``v``-prefixed slug links to a directory
     ``docs.yml`` never creates and serves a 404.
  2. The ``version.cmake`` version appears as a ``versions.json`` slug, so
     the docs for the version being shipped are reachable from the
     switcher.
  3. ``CITATION.cff``'s ``version`` equals the ``version.cmake``
     version, and its ``date-released`` parses as a ``YYYY-MM-DD`` date.
     The date's *value* is deliberately not checked: at release-prep time
     the tag does not exist yet, so nothing in the tree records when the
     release actually went out.
  4. With ``--tag vX.Y.Z`` (passed by CI on a tag push): the tag without
     its leading ``v`` equals the ``version.cmake`` version -- otherwise
     the PyPI and conda package version would not match the release tag --
     and is present as a ``versions.json`` slug.
  5. The documentation builds carry no version literal: ``docs.doxygen``
     leaves ``PROJECT_NUMBER`` empty, and ``docs/source/conf.py`` assigns
     ``version`` and ``release`` only from ``release_version()`` and
     ``version_label()`` and builds any ``html_title`` from ``release``.
     Both take the version from ``version.cmake`` through
     ``docs/cytnx_version.py``; a literal would override it and go stale,
     which is how the API reference showed ``v1.0.0`` through three later
     releases.

``tools/bump_version.py`` rewrites the three metadata files and runs this
check. Run it on its own before drafting a release::

    python3 tools/check_release_consistency.py

It prints every problem it finds and exits non-zero if any remain.
"""

import argparse
import ast
import datetime
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_CMAKE = REPO_ROOT / "version.cmake"
VERSIONS_JSON = REPO_ROOT / "docs" / "site_root" / "versions.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"
DOXYFILE = REPO_ROOT / "docs.doxygen"
SPHINX_CONF = REPO_ROOT / "docs" / "source" / "conf.py"

# Slugs that are not numbered releases: they are exempt from the
# "must be numeric" rule and never need to equal version.cmake.
ALIAS_SLUGS = {"dev", "stable", "latest"}


def read_cmake_version() -> str:
    text = VERSION_CMAKE.read_text()
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        # Mirror the exact pattern scikit-build-core uses in pyproject.toml's
        # [tool.scikit-build.metadata.version] block to read version.cmake, so
        # this check accepts precisely what the build accepts. Keeping it strict
        # -- case-sensitive `set(`, a CYTNX_-style prefix, a bare integer with
        # no quotes -- means a version.cmake the build itself cannot parse fails
        # here too, rather than being green-lit by a more permissive regex.
        match = re.search(rf"set\(\w+?VERSION_{field}\s+(\d+)\)", text)
        if not match:
            sys.exit(f"could not parse {field} version from {VERSION_CMAKE}")
        parts.append(match.group(1))
    return ".".join(parts)


def read_slugs() -> list[str]:
    try:
        entries = json.loads(VERSIONS_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"could not read {VERSIONS_JSON}: {exc}")
    if not isinstance(entries, list):
        sys.exit(f"{VERSIONS_JSON} must contain a JSON array")
    slugs = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            sys.exit(f"{VERSIONS_JSON} entry {i} is not a JSON object")
        if "name" not in entry or "version" not in entry:
            sys.exit(
                f'{VERSIONS_JSON} entry {i} must have both "name" and "version" '
                f"(the switcher renders both fields)"
            )
        slugs.append(str(entry["version"]))
    return slugs


def read_citation_field(field: str) -> str:
    """Return a top-level scalar field from CITATION.cff.

    Parsed with a regex rather than PyYAML so this script keeps running on a
    bare runner with no pip install step, the way the workflow invokes it.
    The pattern is anchored at the start of a line, so it reads only
    top-level keys: an indented key of the same name nested under, say, a
    future ``preferred-citation`` block is not mistaken for the software's
    own metadata, and ``cff-version`` is not mistaken for ``version``.
    """
    text = CITATION_CFF.read_text()
    match = re.search(
        rf"""^{re.escape(field)}:[ \t]*(?:'([^']*)'|"([^"]*)"|([^\s#]+))[ \t]*$""",
        text,
        re.MULTILINE,
    )
    if not match:
        sys.exit(f"could not parse top-level '{field}' from {CITATION_CFF}")
    return next(group for group in match.groups() if group is not None)


# Module-level names in docs/source/conf.py that carry the version, and the
# docs/cytnx_version.py function each must be assigned from.
SPHINX_VERSION_SOURCES = {"version": "release_version", "release": "version_label"}


def check_sphinx_version_source() -> list[str]:
    """Return errors for ``conf.py`` version assignments not taken from cytnx_version.

    Inspects the syntax tree instead of matching text, so any spelling of a
    literal -- plain or f-string, ``str(1.0)``, a later reassignment -- is
    caught. ``version`` and ``release`` must each be assigned exactly from
    the matching call, and ``html_title``, if set, must use ``release``.
    """
    tree = ast.parse(SPHINX_CONF.read_text(), filename=str(SPHINX_CONF))
    assigned: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)

    errors = []
    for name, function in SPHINX_VERSION_SOURCES.items():
        values = assigned.get(name, [])
        all_expected_calls = all(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == function
            and not value.args
            and not value.keywords
            for value in values
        )
        if not values or not all_expected_calls:
            errors.append(
                f"docs/source/conf.py must set `{name} = {function}()` (from "
                f"docs/cytnx_version.py) and assign {name} nowhere else, so the "
                f"user guide shows the version from version.cmake."
            )
    for value in assigned.get("html_title", []):
        uses_release = any(
            isinstance(node, ast.Name) and node.id == "release" for node in ast.walk(value)
        )
        if not uses_release:
            errors.append(
                "docs/source/conf.py sets html_title without using `release`; a "
                "version written into the title goes stale."
            )
    return errors


def looks_like_v_prefixed_release(slug: str) -> bool:
    return slug.startswith("v") and slug[1:2].isdigit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Cytnx release version metadata."
    )
    parser.add_argument(
        "--tag",
        help="release git tag being published, e.g. v1.1.0 "
        "(enables the tag-vs-metadata checks)",
    )
    args = parser.parse_args()

    cmake_version = read_cmake_version()
    slugs = read_slugs()
    errors: list[str] = []

    # Check 1: numbered-release slugs must be numeric.
    for slug in slugs:
        if slug in ALIAS_SLUGS:
            continue
        if looks_like_v_prefixed_release(slug):
            errors.append(
                f'versions.json slug "{slug}" has a leading "v". docs.yml '
                f"publishes release docs to a directory named after the tag "
                f'with the "v" stripped, so this links to a 404. Use '
                f'"{slug[1:]}" instead.'
            )

    # Check 2: the version being built must have a docs slug.
    if cmake_version not in slugs:
        errors.append(
            f"version.cmake is {cmake_version} but docs/site_root/versions.json "
            f'has no entry with "version": "{cmake_version}". Add '
            f'{{"name": "{cmake_version}", "version": "{cmake_version}"}} so the '
            f"docs for this release are reachable from the version switcher."
        )

    # Check 3: the citation metadata must name the version in the tree.
    citation_version = read_citation_field("version")
    if citation_version != cmake_version:
        errors.append(
            f"CITATION.cff says version {citation_version} but version.cmake is "
            f"{cmake_version}. The citation metadata would tell users to cite a "
            f"release other than the one they installed. Set CITATION.cff's "
            f"version to {cmake_version} and its date-released to the date that "
            f"release ships."
        )

    citation_date = read_citation_field("date-released")
    try:
        datetime.date.fromisoformat(citation_date)
    except ValueError:
        errors.append(
            f"CITATION.cff date-released is {citation_date!r}, which is not a "
            f"YYYY-MM-DD date. Citation tools parse this field as a date."
        )

    # Check 4: on a tag push, the tag must agree with both.
    if args.tag:
        tag = args.tag
        tag_version = tag[1:] if tag.startswith("v") else tag
        if tag_version != cmake_version:
            errors.append(
                f"git tag {tag} implies version {tag_version} but version.cmake "
                f"is {cmake_version}. The PyPI and conda packages take their "
                f"version from version.cmake, so the published package would not "
                f"match the release tag. Update version.cmake to {tag_version} "
                f"(or retag) before releasing."
            )
        if tag_version not in slugs:
            errors.append(
                f"git tag {tag} has no matching docs slug in "
                f'docs/site_root/versions.json (expected "version": '
                f'"{tag_version}").'
            )

    # Check 5: the documentation builds must not carry a version literal.
    doxygen_number = re.search(r"^PROJECT_NUMBER\s*=(.*)$", DOXYFILE.read_text(), re.MULTILINE)
    if doxygen_number and doxygen_number.group(1).strip():
        errors.append(
            f"docs.doxygen sets PROJECT_NUMBER = {doxygen_number.group(1).strip()}. "
            f"Leave it empty: docs/build_api_docs.py sets the version label from "
            f"version.cmake, and a literal here goes stale."
        )
    errors.extend(check_sphinx_version_source())

    if errors:
        print("Release metadata is inconsistent:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    summary = (
        f"Release metadata OK: version.cmake={cmake_version}, "
        f"CITATION.cff={citation_version} ({citation_date})"
    )
    if args.tag:
        summary += f", tag={args.tag}"
    summary += f", versions.json slugs={slugs}"
    print(summary)


if __name__ == "__main__":
    main()
