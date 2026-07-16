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
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API_BASE = "https://api.anaconda.org"
ORGANIZATION = "cytnx-nightly-wheels"
NIGHTLY_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.dev(\d{12})$")
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def _urlopen_with_retry(request: urllib.request.Request, *, timeout: int) -> bytes:
    """Retry a full request/response cycle on transient network failures.

    anaconda.org occasionally stalls a response -- whether while establishing
    the connection or while streaming the body -- past the socket timeout
    (surfaced as a bare ``TimeoutError``, since urllib only wraps
    connection-establishment failures in ``URLError``), drops the connection
    mid-request (``ConnectionError``), or closes it after sending only part
    of the body (``http.client.IncompleteRead``, which subclasses
    ``HTTPException`` rather than ``OSError`` and so isn't caught by
    ``ConnectionError``/``URLError``). The body is read to completion inside
    the same attempt so a stall during any of these phases is retried the
    same way. ``HTTPError`` is a real response from the server (e.g.
    404/401/403) and is never retried -- it propagates on the first attempt.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            urllib.error.URLError,
        ) as error:
            if attempt == MAX_ATTEMPTS:
                raise
            print(
                f"attempt {attempt}/{MAX_ATTEMPTS} failed ({error!r}), retrying",
                file=sys.stderr,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable: the last attempt always returns or raises")


def fetch_versions(package: str) -> list[str]:
    url = f"{API_BASE}/package/{ORGANIZATION}/{package}"
    try:
        body = _urlopen_with_retry(urllib.request.Request(url), timeout=30)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            # The package is not on the channel yet -- e.g. the first
            # nightly upload for a brand-new package, where this prune runs
            # before any wheel has created it. Nothing to prune.
            print(f"{package}: not on the channel yet, nothing to prune")
            return []
        raise
    return list(json.loads(body)["versions"])


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


def delete_version(package: str, version: str, token: str) -> None:
    url = f"{API_BASE}/release/{ORGANIZATION}/{package}/{version}"
    request = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": f"token {token}"}
    )
    try:
        _urlopen_with_retry(request, timeout=30)
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
    parser.add_argument(
        "--package",
        default="cytnx",
        help="package name on the cytnx-nightly-wheels channel to prune (default: cytnx)",
    )
    args = parser.parse_args()
    if args.keep < 0:
        parser.error("--keep must be a non-negative integer")

    versions = fetch_versions(args.package)
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
        delete_version(args.package, version, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
