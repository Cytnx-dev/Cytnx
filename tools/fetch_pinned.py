#!/usr/bin/env python3
"""Download a pinned bootstrap artifact and verify it against its SHA-256.

Some CI steps have to download a tool before any trustworthy package manager
exists to fetch it, for example micromamba inside a fresh manylinux container
or a CA bundle on a runner whose own trust store may be out of date. Those
downloads cannot rely on HTTPS alone: the client making them may trust a
compromised certificate authority, as every CA bundle older than certifi
2026.10.19 trusts the compromised eMudhra intermediate CA certificates. This
script pins each such artifact to an exact URL and SHA-256 digest, so a
substituted download fails here rather than running.

Usage:
    python3 tools/fetch_pinned.py ca-bundle DEST
        Writes certifi's cacert.pem to DEST, for use as the CA bundle of
        clients whose own bundle cannot be trusted (curl --cacert,
        micromamba --cacert-path, SSL_CERT_FILE, REQUESTS_CA_BUNDLE).
    python3 tools/fetch_pinned.py micromamba DEST
        Writes the micromamba executable for the running platform to DEST.

The pins live in this file only. When bumping one, take the digest from a
source independent of the download itself (PyPI's JSON API for certifi, the
conda-forge repodata for micromamba) and confirm the downloaded file matches.

Runs under the oldest python3 a manylinux_2_28 container provides, so it uses
only the standard library and no syntax newer than Python 3.6.
"""

import argparse
import hashlib
import io
import os
import platform
import sys
import tarfile
import typing
import urllib.request
import zipfile


class Artifact(typing.NamedTuple):
    """One pinned download and the single member extracted from it."""

    url: str
    sha256: str
    member: str


# certifi 2026.10.19 is the first release without the root certificate that
# signed the compromised eMudhra intermediates.
CA_BUNDLE = Artifact(
    url=(
        "https://files.pythonhosted.org/packages/c4/51/"
        "29046e48f693c54e5b7412216d447a1d8362c9653148d98c02cdf65fa733/"
        "certifi-2026.10.19-py3-none-any.whl"
    ),
    sha256="42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b",
    member="certifi/cacert.pem",
)

# micromamba 2.9.0, keyed by conda subdir. micromamba verifies HTTPS against
# the system CA bundle unless given --cacert-path, so callers pass it the
# bundle above.
MICROMAMBA = {
    "linux-64": Artifact(
        url="https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2",
        sha256="8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd",
        member="bin/micromamba",
    ),
    "linux-aarch64": Artifact(
        url="https://conda.anaconda.org/conda-forge/linux-aarch64/micromamba-2.9.0-0.tar.bz2",
        sha256="e705ffeed90ce0659eb546e4b1e1028c9eaf0bc9cc854867b19ac5ce0ba5852f",
        member="bin/micromamba",
    ),
    "osx-64": Artifact(
        url="https://conda.anaconda.org/conda-forge/osx-64/micromamba-2.9.0-0.tar.bz2",
        sha256="0426ecdc41636d369f57b8fe6acbf4385a69eca45b56d9ee7d3a840a9965d44f",
        member="bin/micromamba",
    ),
    "osx-arm64": Artifact(
        url="https://conda.anaconda.org/conda-forge/osx-arm64/micromamba-2.9.0-0.tar.bz2",
        sha256="500f5074feb8d02c4296ef9921c3650ed2874171805a9fbb8fbb53896433646b",
        member="bin/micromamba",
    ),
}

DOWNLOAD_TIMEOUT_SECONDS = 120


def conda_subdir() -> str:
    """Return the conda subdir naming the running platform.

    Raises:
        SystemExit: If the platform has no pinned micromamba.
    """
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine == "x86_64":
        return "linux-64"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "linux-aarch64"
    if system == "Darwin" and machine == "x86_64":
        return "osx-64"
    if system == "Darwin" and machine == "arm64":
        return "osx-arm64"
    sys.exit(f"fetch_pinned: no pinned micromamba for {system} {machine}")


def download_verified(artifact: Artifact) -> bytes:
    """Download an artifact and check its digest.

    The download goes through urllib with the default trust store, which may
    be outdated; the digest check, not HTTPS, is what authenticates the bytes.

    Args:
        artifact: The pinned download.

    Returns:
        The downloaded bytes.

    Raises:
        SystemExit: If the digest does not match the pin.
    """
    with urllib.request.urlopen(artifact.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != artifact.sha256:
        sys.exit(
            f"fetch_pinned: SHA-256 mismatch for {artifact.url}\n"
            f"  expected {artifact.sha256}\n"
            f"  got      {digest}"
        )
    return payload


def extract_member(payload: bytes, artifact: Artifact) -> bytes:
    """Return the contents of the artifact's member from a wheel or tar.bz2.

    Only the named member is read, so nothing else in the archive reaches the
    file system.

    Args:
        payload: The verified archive bytes.
        artifact: The pinned download the bytes came from.

    Returns:
        The member's contents.
    """
    if artifact.url.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return archive.read(artifact.member)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:bz2") as archive:
        member = archive.extractfile(artifact.member)
        if member is None:
            sys.exit(f"fetch_pinned: {artifact.member} is not a regular file in {artifact.url}")
        return member.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=["ca-bundle", "micromamba"])
    parser.add_argument("dest", help="path of the file to write")
    args = parser.parse_args()

    if args.name == "ca-bundle":
        artifact = CA_BUNDLE
        mode = 0o644
    else:
        artifact = MICROMAMBA[conda_subdir()]
        mode = 0o755

    contents = extract_member(download_verified(artifact), artifact)
    dest_dir = os.path.dirname(os.path.abspath(args.dest))
    os.makedirs(dest_dir, exist_ok=True)
    with open(args.dest, "wb") as out:
        out.write(contents)
    os.chmod(args.dest, mode)
    print(f"fetch_pinned: wrote {args.dest} from {artifact.url}")


if __name__ == "__main__":
    main()
