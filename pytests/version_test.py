import re
from pathlib import Path

import cytnx

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_version_cmake():
    text = (_REPO_ROOT / "version.cmake").read_text()
    parts = {
        key: re.search(rf"set\(CYTNX_VERSION_{key}\s+(\d+)\)", text).group(1)
        for key in ("MAJOR", "MINOR", "PATCH")
    }
    return f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"


def test_version_matches_version_cmake():
    assert cytnx.__version__ == _read_version_cmake()
