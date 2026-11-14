#!/usr/bin/env python3
"""Set the release version in every file that carries it.

Usage::

    python3 tools/bump_version.py 1.2.0 [--date YYYY-MM-DD]

Three committed files carry the version, and the release workflows and
citation tools read them at the tagged commit (see RELEASING.md):

* ``version.cmake`` -- ``MAJOR``, ``MINOR`` and ``PATCH``. The package
  version, ``cytnx.__version__`` and both documentation builds derive their
  version from this file.
* ``docs/site_root/versions.json`` -- the docs slug of the release, which
  the version switcher and the documentation landing page link to.
* ``CITATION.cff`` -- ``version`` and ``date-released`` (today by default).

The script rewrites all three and then runs
``tools/check_release_consistency.py``, the same check CI runs on the
release-prep pull request.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_CMAKE = REPO_ROOT / "version.cmake"
VERSIONS_JSON = REPO_ROOT / "docs" / "site_root" / "versions.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"
CONSISTENCY_CHECK = REPO_ROOT / "tools" / "check_release_consistency.py"


def replace_once(text: str, pattern: str, replacement: str, where: pathlib.Path) -> str:
    """Substitute exactly one match of ``pattern``; fail if there is not exactly one."""
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        sys.exit(f"expected exactly one match of {pattern!r} in {where}, found {count}")
    return text


def bump_version_cmake(version: str) -> None:
    text = VERSION_CMAKE.read_text()
    for field, number in zip(("MAJOR", "MINOR", "PATCH"), version.split(".")):
        text = replace_once(
            text, rf"^(set\(\w+?VERSION_{field}\s+)\d+(\))", rf"\g<1>{number}\g<2>", VERSION_CMAKE
        )
    VERSION_CMAKE.write_text(text)


def add_docs_slug(version: str) -> None:
    entries = json.loads(VERSIONS_JSON.read_text())
    if any(entry.get("version") == version for entry in entries):
        print(f"{VERSIONS_JSON.relative_to(REPO_ROOT)} already lists {version}")
        return
    entries.append({"name": version, "version": version})
    VERSIONS_JSON.write_text(json.dumps(entries, indent=2) + "\n")


def bump_citation(version: str, date: datetime.date) -> None:
    text = CITATION_CFF.read_text()
    text = replace_once(text, r"^version:.*$", f"version: {version}", CITATION_CFF)
    text = replace_once(
        text, r"^date-released:.*$", f"date-released: '{date.isoformat()}'", CITATION_CFF
    )
    CITATION_CFF.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the Cytnx release version everywhere.")
    parser.add_argument("version", help="new release version, MAJOR.MINOR.PATCH (no leading v)")
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
        help="release date for CITATION.cff, YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        sys.exit(f"version must be MAJOR.MINOR.PATCH without a leading v, got {args.version!r}")

    bump_version_cmake(args.version)
    add_docs_slug(args.version)
    bump_citation(args.version, args.date)
    for path in (VERSION_CMAKE, VERSIONS_JSON, CITATION_CFF):
        print(f"updated {path.relative_to(REPO_ROOT)}")

    subprocess.run([sys.executable, str(CONSISTENCY_CHECK)], check=True)


if __name__ == "__main__":
    main()
