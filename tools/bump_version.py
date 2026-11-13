#!/usr/bin/env python3
"""Set the release version in every file that carries it.

Usage::

    python3 tools/bump_version.py 1.2.0 [--date YYYY-MM-DD]   # release prep
    python3 tools/bump_version.py 1.2.1 --next                # after the release

Three committed files carry the version, and the release workflows and
citation tools read them at the tagged commit (see RELEASING.md):

* ``version.cmake`` -- ``MAJOR``, ``MINOR`` and ``PATCH``. The package
  version, ``cytnx.__version__`` and both documentation builds derive their
  version from this file.
* ``docs/site_root/versions.json`` -- the docs slug of the release, which
  the version switcher and the documentation landing page link to.
* ``CITATION.cff`` -- ``version`` and ``date-released`` (today by default).

For a release, the script rewrites all three. With ``--next`` it rewrites
``version.cmake`` alone: right after a release ships, ``version.cmake``
moves past it (the patch bump by default, see RELEASING.md step 4) while
``versions.json`` and ``CITATION.cff`` keep describing the release that
exists. Nightly wheels and the dev docs are stamped from ``version.cmake``,
and PEP 440 orders ``X.Y.Z.devN`` below ``X.Y.Z``, so without this step
every nightly would lose to the release in ``pip`` (#1204).

Either way the script then runs ``tools/check_release_consistency.py``,
the same check CI runs on every pull request.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_release_consistency import newest_release, parse_release, read_slugs  # noqa: E402

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


def write_if_changed(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` and report whether the file changed."""
    if path.read_text() == text:
        print(f"unchanged {path.relative_to(REPO_ROOT)}", flush=True)
        return
    path.write_text(text)
    print(f"updated {path.relative_to(REPO_ROOT)}", flush=True)


def bump_version_cmake(version: str) -> None:
    text = VERSION_CMAKE.read_text()
    for field, number in zip(("MAJOR", "MINOR", "PATCH"), version.split(".")):
        text = replace_once(
            text, rf"^(set\(\w+?VERSION_{field}\s+)\d+(\))", rf"\g<1>{number}\g<2>", VERSION_CMAKE
        )
    write_if_changed(VERSION_CMAKE, text)


def add_docs_slug(version: str) -> None:
    entries = json.loads(VERSIONS_JSON.read_text())
    if any(entry.get("version") == version for entry in entries):
        print(f"unchanged {VERSIONS_JSON.relative_to(REPO_ROOT)}", flush=True)
        return
    entries.append({"name": version, "version": version})
    write_if_changed(VERSIONS_JSON, json.dumps(entries, indent=2) + "\n")


def bump_citation(version: str, date: datetime.date) -> None:
    text = CITATION_CFF.read_text()
    text = replace_once(text, r"^version:.*$", f"version: {version}", CITATION_CFF)
    text = replace_once(
        text, r"^date-released:.*$", f"date-released: '{date.isoformat()}'", CITATION_CFF
    )
    write_if_changed(CITATION_CFF, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the Cytnx release version everywhere.")
    parser.add_argument("version", help="new release version, MAJOR.MINOR.PATCH (no leading v)")
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        help="release date for CITATION.cff, YYYY-MM-DD (default: today); not with --next",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="post-release bump: set version.cmake to a version above the newest release "
        "(the patch bump by default) and leave versions.json and CITATION.cff at the "
        "release that shipped",
    )
    args = parser.parse_args()
    if parse_release(args.version) is None:
        sys.exit(f"version must be MAJOR.MINOR.PATCH without a leading v, got {args.version!r}")
    if args.next and args.date is not None:
        parser.error("--date sets CITATION.cff, which --next leaves alone")

    if args.next:
        latest_release = newest_release(read_slugs())
        if latest_release is not None and parse_release(args.version) <= parse_release(
            latest_release
        ):
            sys.exit(
                f"--next needs a version above the newest release {latest_release}, "
                f"got {args.version}"
            )
        bump_version_cmake(args.version)
    else:
        bump_version_cmake(args.version)
        add_docs_slug(args.version)
        bump_citation(args.version, args.date or datetime.date.today())

    sys.exit(subprocess.run([sys.executable, str(CONSISTENCY_CHECK)]).returncode)


if __name__ == "__main__":
    main()
