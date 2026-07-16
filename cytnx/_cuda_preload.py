"""Preload NVIDIA/cuTENSOR/cuQuantum shared libraries before importing cytnx's
compiled extension.

cytnx-cuda's extension is linked against versioned sonames (libcudart.so.13,
libcublas.so.13, libcutensor.so.2, ...) provided by the nvidia-*/
cutensor-cu13/cutensornet-cu13/custatevec-cu13 pip packages (see
tools/prepare_cuda_release.py) rather than vendored into the wheel. Those
packages install their libraries under site-packages namespace directories
(nvidia/cu13/lib, cutensor/lib, cuquantum/lib) that are not on the dynamic
linker's default search path, so importing the extension without
LD_LIBRARY_PATH set would abort with an ImportError naming the missing
soname. Loading each library explicitly with RTLD_GLOBAL first -- so the
extension's own NEEDED entries resolve against libraries already mapped into
the process by soname -- avoids requiring LD_LIBRARY_PATH at all (the
approach PyTorch's nvidia-* wheel dependencies use).

cytnx/__init__.py calls preload() before `from . import cytnx`, gated on the
"cuda" marker CMakeLists.txt writes to vinfo.tmp only for USE_CUDA builds, so
this is a no-op import on CPU-only installs.
"""

import ctypes
import importlib.util
import pathlib

# (namespace package, path segments under it) for each package family that
# installs shared libraries this needs preloaded. nvidia-cuda-runtime,
# nvidia-cublas, nvidia-cusparse, nvidia-curand, nvidia-cusolver, and their
# transitive nvidia-cuda-nvrtc/nvidia-nvjitlink dependencies all install into
# the shared "nvidia" namespace package's cu13/lib directory; cutensor-cu13
# installs into its own "cutensor" namespace package; cutensornet-cu13 and
# custatevec-cu13 both install into the shared "cuquantum" namespace package.
_LIB_LOCATIONS = (
    ("nvidia", ("cu13", "lib")),
    ("cutensor", ("lib",)),
    ("cuquantum", ("lib",)),
)


def _discover_lib_paths() -> list[pathlib.Path]:
    """Find every shared library under the namespace dirs in _LIB_LOCATIONS.

    Returns:
        Paths of the .so files to preload, in no particular order.
    """
    paths = []
    for top_name, rel_parts in _LIB_LOCATIONS:
        spec = importlib.util.find_spec(top_name)
        if spec is None or spec.submodule_search_locations is None:
            continue
        for location in spec.submodule_search_locations:
            lib_dir = pathlib.Path(location).joinpath(*rel_parts)
            if lib_dir.is_dir():
                paths.extend(sorted(lib_dir.glob("*.so*")))
    return paths


def preload() -> None:
    """Load every discovered library with RTLD_GLOBAL.

    Libraries are tried in a fixed-point loop rather than a hardcoded order:
    a library whose own NEEDED entries aren't satisfied yet raises OSError
    and is retried after the rest of the pass, so any dependency ordering
    among these packages resolves itself without this module having to track
    it explicitly.

    Raises:
        ImportError: A pass completed without loading any of the still-
            pending libraries, so no ordering will make them load; usually
            means the pip packages are missing or version-mismatched.
    """
    pending = _discover_lib_paths()
    last_error = None
    while pending:
        still_pending = []
        loaded_this_pass = False
        for lib_path in pending:
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded_this_pass = True
            except OSError as exc:
                still_pending.append(lib_path)
                last_error = exc
        if not loaded_this_pass:
            names = ", ".join(p.name for p in still_pending)
            raise ImportError(
                "cytnx: failed to preload CUDA runtime libraries: "
                f"{names}. This usually means the nvidia-*/cutensor-cu13/"
                "cutensornet-cu13/custatevec-cu13 pip packages are missing "
                "or version-mismatched; see tools/prepare_cuda_release.py "
                "for the pinned versions cytnx-cuda was built against."
            ) from last_error
        pending = still_pending
