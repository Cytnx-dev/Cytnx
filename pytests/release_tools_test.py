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
import zipfile
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


def write_version_cmake(tree, version):
    major, minor, patch = version.split(".")
    (tree / "version.cmake").write_text(
        f"set(CYTNX_VERSION_MAJOR {major})\n"
        f"set(CYTNX_VERSION_MINOR {minor})\n"
        f"set(CYTNX_VERSION_PATCH {patch})\n"
    )


def newest_slug(tree):
    entries = json.loads((tree / "docs/site_root/versions.json").read_text())
    releases = [e["version"] for e in entries if re.fullmatch(r"\d+\.\d+\.\d+", e["version"])]
    return max(releases, key=lambda v: tuple(int(p) for p in v.split(".")))


def test_check_accepts_version_cmake_ahead_of_newest_release(release_tree):
    write_version_cmake(release_tree, "99.0.0")

    result = run_script(release_tree, "tools/check_release_consistency.py")

    assert result.returncode == 0, result.stderr
    assert "version.cmake=99.0.0" in result.stdout


def test_check_rejects_version_cmake_behind_newest_release(release_tree):
    write_version_cmake(release_tree, "0.0.1")

    result = run_script(release_tree, "tools/check_release_consistency.py")

    assert result.returncode == 1
    assert "already lists" in result.stderr
    assert "--next" in result.stderr


def test_check_rejects_citation_of_a_version_other_than_the_newest_release(release_tree):
    citation = release_tree / "CITATION.cff"
    text, count = re.subn(r"^version: .*$", "version: 0.0.1", citation.read_text(), flags=re.M)
    assert count == 1
    citation.write_text(text)

    result = run_script(release_tree, "tools/check_release_consistency.py")

    assert result.returncode == 1
    assert "CITATION.cff says version 0.0.1" in result.stderr


def test_check_on_tag_requires_all_three_to_name_the_tag(release_tree):
    assert run_script(release_tree, "tools/bump_version.py", "12.3.40").returncode == 0

    checker = "tools/check_release_consistency.py"
    assert run_script(release_tree, checker, "--tag", "v12.3.40").returncode == 0
    mismatched = run_script(release_tree, checker, "--tag", "v12.3.41")
    assert mismatched.returncode == 1
    assert "version.cmake" in mismatched.stderr
    assert "versions.json" in mismatched.stderr
    assert "CITATION.cff" in mismatched.stderr


def test_next_bump_advances_only_version_cmake(release_tree):
    before = snapshot(release_tree)

    result = run_script(release_tree, "tools/bump_version.py", "99.0.0", "--next")

    assert result.returncode == 0, result.stderr
    after = snapshot(release_tree)
    assert after["version.cmake"] == (
        "set(CYTNX_VERSION_MAJOR 99)\nset(CYTNX_VERSION_MINOR 0)\nset(CYTNX_VERSION_PATCH 0)\n"
    )
    assert after["docs/site_root/versions.json"] == before["docs/site_root/versions.json"]
    assert after["CITATION.cff"] == before["CITATION.cff"]
    assert result.stdout.splitlines()[0] == "updated version.cmake"


@pytest.mark.parametrize("case", ["below", "equal"])
def test_next_bump_rejects_a_version_not_above_the_newest_release(release_tree, case):
    version = "0.0.1" if case == "below" else newest_slug(release_tree)
    before = snapshot(release_tree)

    result = run_script(release_tree, "tools/bump_version.py", version, "--next")

    assert result.returncode != 0
    assert "newest release" in result.stderr
    assert snapshot(release_tree) == before


def test_next_bump_refuses_a_release_date(release_tree):
    before = snapshot(release_tree)

    arguments = ("99.0.0", "--next", "--date", "2031-02-03")
    result = run_script(release_tree, "tools/bump_version.py", *arguments)

    assert result.returncode != 0
    assert "--date" in result.stderr
    assert snapshot(release_tree) == before


def test_nightly_stamped_after_next_bump_outranks_the_newest_release(release_tree):
    Version = pytest.importorskip("packaging.version").Version
    newest = newest_slug(release_tree)
    assert run_script(release_tree, "tools/bump_version.py", "99.0.0", "--next").returncode == 0

    base = run_script(release_tree, "docs/cytnx_version.py").stdout.strip()

    assert base == "99.0.0"
    assert Version(f"{base}.dev202611130000") > Version(newest)
    # The failure mode from #1204: a nightly of the released version loses to it.
    assert Version(f"{newest}.dev202611130000") < Version(newest)


def make_simple_index(root, package, version):
    """Create a one-project PEP 503 index holding an empty wheel of ``package``."""
    project_dir = root / package
    project_dir.mkdir(parents=True)
    wheel_name = f"{package.replace('-', '_')}-{version}-py3-none-any.whl"
    dist_info = f"{package.replace('-', '_')}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
    wheel_info = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    with zipfile.ZipFile(project_dir / wheel_name, "w") as wheel:
        wheel.writestr(f"{dist_info}/METADATA", metadata)
        wheel.writestr(f"{dist_info}/WHEEL", wheel_info)
        wheel.writestr(f"{dist_info}/RECORD", "")
    wheel_link = f'<a href="{wheel_name}">{wheel_name}</a>'
    project_link = f'<a href="{package}/">{package}</a>'
    (project_dir / "index.html").write_text(f"<html><body>{wheel_link}</body></html>")
    (root / "index.html").write_text(f"<html><body>{project_link}</body></html>")
    return root.as_uri()


def pip_resolves(package, release_index, nightly_index):
    """Return the version ``pip install --pre`` would pick with both indexes present."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "--quiet",
            "--isolated", "--disable-pip-version-check", "--report", "-", "--pre",
            "--index-url", release_index, "--extra-index-url", nightly_index, package,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    (chosen,) = json.loads(result.stdout)["install"]
    return chosen["metadata"]["version"]


@pytest.mark.parametrize("package", ["cytnx", "cytnx-cuda"])
def test_pip_selects_the_nightly_only_when_it_outranks_the_release(tmp_path, package):
    """The resolution #1204 is about, run against pip itself on local indexes."""
    pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
    if pip.returncode:
        pytest.skip("pip is not installed for this interpreter")
    pip_version = tuple(int(part) for part in pip.stdout.split()[1].split(".")[:2])
    if pip_version < (22, 2):
        pytest.skip("pip --dry-run --report needs pip 22.2 or newer")
    release = make_simple_index(tmp_path / "release", package, "1.1.1")
    stale_nightly = make_simple_index(tmp_path / "stale", package, "1.1.1.dev202610191223")
    bumped_nightly = make_simple_index(tmp_path / "bumped", package, "1.1.2.dev202611130000")

    assert pip_resolves(package, release, stale_nightly) == "1.1.1"
    assert pip_resolves(package, release, bumped_nightly) == "1.1.2.dev202611130000"
