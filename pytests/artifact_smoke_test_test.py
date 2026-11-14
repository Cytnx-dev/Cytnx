"""Tests for tools/artifact_smoke_test.py, the release artifact gate.

The script only earns its place in the conda recipe and the wheel workflows if
it exits non-zero on a bad artifact, so the failure paths are tested here on
every pull request rather than discovered at release time.
"""

import pathlib
import subprocess
import sys

import cytnx

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "tools" / "artifact_smoke_test.py"


def _run(*args, cwd):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_passes_for_the_installed_version(tmp_path):
    result = _run("--expected-version", cytnx.__version__, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "numerical smoke test passed" in result.stdout
    assert "artifact_smoke_test: OK" in result.stdout


def test_fails_on_a_version_mismatch(tmp_path):
    result = _run("--expected-version", "0.0.0", cwd=tmp_path)
    assert result.returncode != 0
    assert "expected version '0.0.0' everywhere" in result.stderr


def test_fails_on_a_malformed_version_cmake(tmp_path):
    version_cmake = tmp_path / "version.cmake"
    version_cmake.write_text(
        "set(CYTNX_VERSION_MAJOR 2)\n"
        "set(CYTNX_VERSION_MINOR 0)\n"
        "set(CYTNX_VERSION_PATCH 0rc1)\n"
    )
    result = _run("--version-cmake", str(version_cmake), cwd=tmp_path)
    assert result.returncode != 0
    assert "could not parse PATCH version" in result.stderr
