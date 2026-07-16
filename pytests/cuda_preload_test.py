import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cytnx"))

import _cuda_preload  # noqa: E402


def test_is_cuda_build_true_when_marker_present(tmp_path):
    (tmp_path / "vinfo.tmp").write_text("cuda\n")
    assert _cuda_preload.is_cuda_build(str(tmp_path)) is True


def test_is_cuda_build_false_when_marker_absent(tmp_path):
    (tmp_path / "vinfo.tmp").write_text("UNI_TEST\n")
    assert _cuda_preload.is_cuda_build(str(tmp_path)) is False


def test_is_cuda_build_false_when_vinfo_missing(tmp_path):
    assert _cuda_preload.is_cuda_build(str(tmp_path)) is False


def _fake_spec(locations):
    spec = MagicMock()
    spec.submodule_search_locations = locations
    return spec


def test_discover_lib_paths_empty_when_no_namespace_packages():
    with patch("_cuda_preload.importlib.util.find_spec", return_value=None):
        assert _cuda_preload._discover_lib_paths() == []


def test_discover_lib_paths_finds_files_under_namespace_dirs(tmp_path):
    nvidia_lib = tmp_path / "nvidia" / "cu13" / "lib"
    nvidia_lib.mkdir(parents=True)
    (nvidia_lib / "libcudart.so.13").write_bytes(b"")
    (nvidia_lib / "not_a_lib.txt").write_bytes(b"")

    cutensor_lib = tmp_path / "cutensor" / "lib"
    cutensor_lib.mkdir(parents=True)
    (cutensor_lib / "libcutensor.so.2").write_bytes(b"")

    def fake_find_spec(name):
        if name == "nvidia":
            return _fake_spec([str(tmp_path / "nvidia")])
        if name == "cutensor":
            return _fake_spec([str(tmp_path / "cutensor")])
        if name == "cuquantum":
            return None
        raise AssertionError(f"unexpected find_spec({name!r})")

    with patch("_cuda_preload.importlib.util.find_spec", side_effect=fake_find_spec):
        paths = _cuda_preload._discover_lib_paths()

    names = sorted(p.name for p in paths)
    assert names == ["libcudart.so.13", "libcutensor.so.2"]


def test_preload_noop_when_nothing_to_load():
    loader = MagicMock()
    _cuda_preload.preload(lib_paths=[], loader=loader)
    loader.assert_not_called()


def test_preload_loads_every_library():
    loader = MagicMock()
    paths = [Path("/opt/a.so"), Path("/opt/b.so")]
    _cuda_preload.preload(lib_paths=paths, loader=loader)
    assert loader.call_count == 2
    loader.assert_any_call(paths[0])
    loader.assert_any_call(paths[1])


def test_preload_retries_out_of_order_dependency():
    # "b.so" depends on "a.so": loading it first fails with OSError (mirrors
    # a real dlopen failing to resolve an unsatisfied NEEDED entry), then
    # succeeds once "a.so" has loaded on a later pass.
    a, b = Path("/opt/a.so"), Path("/opt/b.so")
    loaded = set()

    def loader(path):
        if path == b and a not in loaded:
            raise OSError(f"cannot open shared object file: {path}")
        loaded.add(path)

    _cuda_preload.preload(lib_paths=[b, a], loader=loader)
    assert loaded == {a, b}


def test_preload_raises_importerror_when_stuck():
    bogus = Path("/opt/does-not-exist.so")

    def loader(path):
        raise OSError("cannot open shared object file: No such file or directory")

    with pytest.raises(ImportError, match="does-not-exist.so"):
        _cuda_preload.preload(lib_paths=[bogus], loader=loader)
