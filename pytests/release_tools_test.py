"""Tests for the release-version tooling: bump_version, the consistency check, cytnx_version.

Each test runs the scripts as a user would, on a copy of the files they read
in a temporary directory, so the real checkout is never modified.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

RELEASE_FILES = (
    "tools/bump_version.py",
    "tools/check_release_consistency.py",
    "docs/cytnx_version.py",
    "version.cmake",
    "docs/site_root/versions.json",
    "CITATION.cff",
    "docs.doxygen",
    "docs/source/conf.py",
)


@pytest.fixture
def release_tree(tmp_path):
    """Return a temporary copy of every file the release tools read or write."""
    for relative in RELEASE_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    return tmp_path


def run_script(tree, script, *args):
    # Stop git at the temporary tree, so it is never read as part of an
    # enclosing repository.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)}
    return subprocess.run(
        [sys.executable, str(tree / script), *args],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
    )


def snapshot(tree):
    return {relative: (tree / relative).read_text() for relative in RELEASE_FILES}


def test_version_regex_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")
    sys.path.insert(0, str(REPO_ROOT / "docs"))
    try:
        import cytnx_version
    finally:
        sys.path.pop(0)
    with open(REPO_ROOT / "pyproject.toml", "rb") as pyproject:
        metadata = tomllib.load(pyproject)["tool"]["scikit-build"]["metadata"]["version"]
    assert cytnx_version.VERSION_CMAKE_REGEX == metadata["regex"]


def test_version_label_without_git_is_the_release_version(release_tree):
    (release_tree / "version.cmake").write_text(
        "set(CYTNX_VERSION_MAJOR 3)\nset(CYTNX_VERSION_MINOR 14)\nset(CYTNX_VERSION_PATCH 15)\n"
    )
    result = run_script(release_tree, "docs/cytnx_version.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3.14.15"


def test_consistency_check_passes_on_the_checkout(release_tree):
    result = run_script(release_tree, "tools/check_release_consistency.py")
    assert result.returncode == 0, result.stderr


def test_bump_updates_all_three_files(release_tree):
    result = run_script(release_tree, "tools/bump_version.py", "12.3.40", "--date", "2031-02-03")
    assert result.returncode == 0, result.stderr

    assert (release_tree / "version.cmake").read_text() == (
        "set(CYTNX_VERSION_MAJOR 12)\nset(CYTNX_VERSION_MINOR 3)\nset(CYTNX_VERSION_PATCH 40)\n"
    )
    entries = json.loads((release_tree / "docs/site_root/versions.json").read_text())
    assert entries[-1] == {"name": "12.3.40", "version": "12.3.40"}
    assert [entry["version"] for entry in entries].count("12.3.40") == 1
    citation = (release_tree / "CITATION.cff").read_text()
    assert re.search(r"^version: 12\.3\.40$", citation, re.MULTILINE)
    assert re.search(r"^date-released: '2031-02-03'$", citation, re.MULTILINE)
    # The nested citation of the paper keeps its own fields.
    original_citation = (REPO_ROOT / "CITATION.cff").read_text()
    paper = original_citation[original_citation.index("preferred-citation:") :]
    assert citation.endswith(paper)

    lines = result.stdout.splitlines()
    assert lines[:3] == [
        "updated version.cmake",
        "updated docs/site_root/versions.json",
        "updated CITATION.cff",
    ]
    assert lines[3].startswith("Release metadata OK: version.cmake=12.3.40")


def test_bump_is_idempotent(release_tree):
    arguments = ("12.3.40", "--date", "2031-02-03")
    assert run_script(release_tree, "tools/bump_version.py", *arguments).returncode == 0
    before = snapshot(release_tree)

    result = run_script(release_tree, "tools/bump_version.py", *arguments)

    assert result.returncode == 0, result.stderr
    assert snapshot(release_tree) == before
    assert not any(line.startswith("updated") for line in result.stdout.splitlines())


@pytest.mark.parametrize("version", ["v1.2.0", "1.2", "1.2.0rc1", "1.2.0.0"])
def test_bump_rejects_a_malformed_version(release_tree, version):
    before = snapshot(release_tree)

    result = run_script(release_tree, "tools/bump_version.py", version)

    assert result.returncode != 0
    assert "MAJOR.MINOR.PATCH" in result.stderr
    assert snapshot(release_tree) == before


def test_check_rejects_a_doxygen_version_literal(release_tree):
    doxyfile = release_tree / "docs.doxygen"
    text, count = re.subn(
        r"^PROJECT_NUMBER\s*=.*$",
        'PROJECT_NUMBER = "v1.0.0"',
        doxyfile.read_text(),
        flags=re.MULTILINE,
    )
    assert count == 1
    doxyfile.write_text(text)

    result = run_script(release_tree, "tools/check_release_consistency.py")

    assert result.returncode == 1
    assert "PROJECT_NUMBER" in result.stderr


@pytest.mark.parametrize(
    ("old", "new", "reported"),
    [
        ("release = version_label()", 'release = "1.0.0"', "release"),
        ("release = version_label()", 'release = f"1.0.0"', "release"),
        ("version = release_version()", "version = str(1.0)", "version"),
        ("version = release_version()", "version = release_version('x')", "version"),
        ("release = version_label()", "", "release"),
        (
            "release = version_label()",
            "release = version_label()\nrelease = '1.0.0'",
            "release",
        ),
        ('html_title = f"{project} {release}"', 'html_title = "Cytnx 1.0.0"', "html_title"),
    ],
)
def test_check_rejects_a_sphinx_version_not_from_cytnx_version(release_tree, old, new, reported):
    conf = release_tree / "docs/source/conf.py"
    text = conf.read_text()
    assert re.search(rf"^{re.escape(old)}$", text, re.MULTILINE)
    conf.write_text(re.sub(rf"^{re.escape(old)}$", new, text, flags=re.MULTILINE))

    result = run_script(release_tree, "tools/check_release_consistency.py")

    assert result.returncode == 1
    assert reported in result.stderr
