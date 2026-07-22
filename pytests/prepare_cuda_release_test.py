from pathlib import Path
import sys

import pytest

tomlkit = pytest.importorskip("tomlkit")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prepare_cuda_release as release  # noqa: E402


def _manifest():
    return tomlkit.parse((REPO_ROOT / "pyproject.toml").read_text())


def test_linux_cuda_release_rewrite_preserves_runpath_pipeline():
    document = _manifest()
    original_before_all = document["tool"]["cibuildwheel"]["linux"]["before-all"]

    release.rewrite_pyproject(document, target_platform="linux")

    assert document["project"]["name"] == "cytnx-cuda"
    assert document["tool"]["scikit-build"]["cmake"]["args"] == [
        "--preset=openblas-cuda",
        f"-DCMAKE_INSTALL_RPATH={release.CUDA_INSTALL_RPATH}",
    ]
    linux = document["tool"]["cibuildwheel"]["linux"]
    assert linux["before-all"].startswith(original_before_all + " && ")
    assert "cibuildwheel_before_all_cuda.sh" in linux["before-all"]
    assert "auditwheel repair" in linux["repair-wheel-command"]


def test_windows_cuda_release_rewrite_uses_external_pypi_dlls():
    document = _manifest()
    original_linux = tomlkit.dumps(document["tool"]["cibuildwheel"]["linux"])

    release.rewrite_pyproject(document, target_platform="windows")

    assert document["project"]["name"] == "cytnx-cuda"
    dependencies = document["project"]["dependencies"]
    assert all(
        "sys_platform == 'win32'" in dependency
        for dependency in dependencies
        if dependency.split()[0] in {
            "nvidia-cuda-runtime",
            "nvidia-cublas",
            "nvidia-cusparse",
            "nvidia-curand",
            "nvidia-cusolver",
            "cutensor-cu13",
        }
    )
    assert all(
        "sys_platform == 'linux'" in dependency
        and "win32" not in dependency
        for dependency in dependencies
        if dependency.split()[0] in {"cutensornet-cu13", "custatevec-cu13"}
    )
    assert document["tool"]["scikit-build"]["cmake"]["args"] == [
        "--preset=openblas-cuda",
        "-DUSE_CUQUANTUM=OFF",
        "-DUSE_HPTT=OFF",
        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
    ]
    assert (
        document["tool"]["cibuildwheel"]["windows"]["repair-wheel-command"]
        == "python {project}/tools/repair_windows_wheel.py --wheel {wheel} "
        "--dest-dir {dest_dir}"
    )
    assert tomlkit.dumps(document["tool"]["cibuildwheel"]["linux"]) == original_linux


def test_cuda_release_rewrite_rejects_unknown_platform():
    with pytest.raises(ValueError, match="unsupported CUDA wheel platform"):
        release.rewrite_pyproject(_manifest(), target_platform="plan9")
