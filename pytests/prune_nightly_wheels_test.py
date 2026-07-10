import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from prune_nightly_wheels import select_versions_to_delete  # noqa: E402


def test_fewer_than_keep_deletes_nothing():
    versions = ["1.2.0.dev202601010000", "1.2.0.dev202601020000"]
    assert select_versions_to_delete(versions, keep=5) == []


def test_exactly_keep_deletes_nothing():
    versions = [f"1.2.0.dev20260101{i:04d}" for i in range(5)]
    assert select_versions_to_delete(versions, keep=5) == []


def test_excess_deletes_oldest_first():
    versions = [
        "1.2.0.dev202601010000",  # oldest, should be deleted
        "1.2.0.dev202601020000",
        "1.2.0.dev202601030000",
        "1.2.0.dev202601040000",
        "1.2.0.dev202601050000",
        "1.2.0.dev202601060000",  # newest
    ]
    assert select_versions_to_delete(versions, keep=5) == ["1.2.0.dev202601010000"]


def test_already_over_quota_deletes_all_excess_at_once():
    versions = [f"1.2.0.dev20260101{i:04d}" for i in range(12)]
    to_delete = select_versions_to_delete(versions, keep=5)
    assert len(to_delete) == 7
    assert to_delete == versions[:7]


def test_non_nightly_versions_are_never_selected_or_counted():
    versions = [
        "1.2.0",  # stable release, must never be touched
        "1.2.0.dev202601010000",
        "1.2.0.dev202601020000",
        "1.2.0.dev202601030000",
    ]
    assert select_versions_to_delete(versions, keep=2) == ["1.2.0.dev202601010000"]


def test_orders_across_base_version_bump():
    versions = [
        "0.10.0.dev202601010000",  # newer base version, older-looking string
        "0.9.9.dev202601020000",
    ]
    assert select_versions_to_delete(versions, keep=1) == ["0.9.9.dev202601020000"]
