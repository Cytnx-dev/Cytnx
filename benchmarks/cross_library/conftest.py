"""Shared pytest configuration for the cross-library benchmark suite.

`--benchmark-only` (pytest-benchmark) already restricts a run to tests using
the `benchmark` fixture. This adds the mirror-image behavior for
`--memray` (pytest-memray), which has no built-in equivalent: when
`--memray` is passed, every test that requests the `benchmark` fixture is
skipped, so a `--memray` run only exercises the `*_memory` tests.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--memray", default=False):
        return
    skip_benchmark = pytest.mark.skip(reason="Skipping non-memory test (--memray active).")
    for item in items:
        if "benchmark" in item.fixturenames:
            item.add_marker(skip_benchmark)
