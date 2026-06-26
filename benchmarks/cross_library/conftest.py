"""Shared pytest configuration for the cross-library benchmark suite.

`--benchmark-only` (pytest-benchmark) already restricts a run to tests using
the `benchmark` fixture. This adds the mirror-image behavior for
`--memray` (pytest-memray), which has no built-in equivalent: when
`--memray` is passed, every test not carrying the `cytnx_memory` marker
(registered in pyproject.toml, applied to every `*_memory` test in this
directory) is skipped, so a `--memray` run only exercises peak-memory tests.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--memray", default=False):
        return
    skip_non_memory = pytest.mark.skip(reason="Skipping non-memory test (--memray active).")
    for item in items:
        if not item.get_closest_marker("cytnx_memory"):
            item.add_marker(skip_non_memory)
