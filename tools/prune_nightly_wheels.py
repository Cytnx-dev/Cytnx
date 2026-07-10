#!/usr/bin/env python3
"""Delete old nightly releases from the anaconda.org nightly channel.

anaconda.org enforces a 10GB storage quota per organisation
(cytnx-nightly-wheels). Since a nightly wheel set is uploaded on every push
to master, the channel fills up in a matter of days (#1021). This script
keeps the channel bounded by deleting every nightly release beyond the
newest ``--keep`` versions, regardless of how the excess accumulated --
whether from a burst of merges or from running for the first time against
an already-over-quota channel.

Only versions matching the stamp produced by
tools/prepare_nightly_release.py (``MAJOR.MINOR.PATCH.devYYYYMMDDHHMM``)
are ever considered for deletion. Anything else on the channel is left
untouched and reported as unexpected, since deleting it could destroy a
release this script was never meant to manage.

Requires ``ANACONDA_API_TOKEN`` in the environment with delete permission
on the channel. Uses only the standard library so no extra dependency is
needed in the release-tools group.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

API_BASE = "https://api.anaconda.org"
ORGANIZATION = "cytnx-nightly-wheels"
PACKAGE = "cytnx"
NIGHTLY_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.dev(\d{12})$")


def fetch_versions() -> list[str]:
    url = f"{API_BASE}/package/{ORGANIZATION}/{PACKAGE}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return list(json.load(response)["versions"])


def sort_key(version: str) -> tuple[int, int, int, str]:
    major, minor, patch, dev_stamp = NIGHTLY_VERSION_RE.match(version).groups()
    return (int(major), int(minor), int(patch), dev_stamp)


def select_versions_to_delete(versions: list[str], keep: int) -> list[str]:
    """Return the nightly versions to delete, oldest excess first.

    Non-nightly-shaped versions are never selected and do not count
    against ``keep``, so a stray non-dev release can't push a real
    nightly out of the retained window.
    """
    nightly_versions = []
    for version in versions:
        if NIGHTLY_VERSION_RE.match(version):
            nightly_versions.append(version)
        else:
            print(f"skipping non-nightly version on channel: {version}", file=sys.stderr)

    nightly_versions.sort(key=sort_key)
    excess = len(nightly_versions) - keep
    return nightly_versions[:excess] if excess > 0 else []


def delete_version(version: str, token: str) -> None:
    url = f"{API_BASE}/release/{ORGANIZATION}/{PACKAGE}/{version}"
    request = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": f"token {token}"}
    )
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            print(f"{version}: already gone, skipping")
            return
        if error.code in (401, 403):
            sys.exit(
                f"{version}: permission denied ({error.code}) deleting release -- "
                "the token needs delete permission on the channel, not just upload"
            )
        raise
    print(f"{version}: deleted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        type=int,
        required=True,
        help="number of newest nightly versions to retain",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the versions that would be deleted without deleting them",
    )
    args = parser.parse_args()
    if args.keep < 0:
        parser.error("--keep must be a non-negative integer")

    versions = fetch_versions()
    to_delete = select_versions_to_delete(versions, args.keep)

    if not to_delete:
        print(f"nothing to prune: {len(versions)} version(s) on channel, keeping {args.keep}")
        return 0

    if args.dry_run:
        for version in to_delete:
            print(f"{version}: would delete (dry run)")
        return 0

    token = os.environ.get("ANACONDA_API_TOKEN")
    if not token:
        sys.exit("ANACONDA_API_TOKEN is required to delete releases")

    for version in to_delete:
        delete_version(version, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
