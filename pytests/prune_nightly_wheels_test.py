import http.client
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from prune_nightly_wheels import _urlopen_with_retry, select_versions_to_delete  # noqa: E402


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


def _fake_response(body: bytes) -> MagicMock:
    response = MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


def _response_failing_read(error: Exception) -> MagicMock:
    response = MagicMock()
    response.read.side_effect = error
    response.__enter__.return_value = response
    return response


@patch("prune_nightly_wheels.time.sleep")
@patch("prune_nightly_wheels.urllib.request.urlopen")
def test_retry_succeeds_after_transient_timeouts(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [TimeoutError(), TimeoutError(), _fake_response(b"body")]
    assert _urlopen_with_retry(Mock(), timeout=30) == b"body"
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("prune_nightly_wheels.time.sleep")
@patch("prune_nightly_wheels.urllib.request.urlopen")
def test_retry_succeeds_after_incomplete_read(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [
        _response_failing_read(http.client.IncompleteRead(b"partial")),
        _fake_response(b"body"),
    ]
    assert _urlopen_with_retry(Mock(), timeout=30) == b"body"
    assert mock_urlopen.call_count == 2
    assert mock_sleep.call_count == 1


@patch("prune_nightly_wheels.time.sleep")
@patch("prune_nightly_wheels.urllib.request.urlopen")
def test_retry_raises_after_exhausting_attempts(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        _urlopen_with_retry(Mock(), timeout=30)
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("prune_nightly_wheels.time.sleep")
@patch("prune_nightly_wheels.urllib.request.urlopen")
def test_http_error_is_not_retried(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = urllib.error.HTTPError(
        "url", 404, "not found", hdrs=None, fp=None
    )
    with pytest.raises(urllib.error.HTTPError):
        _urlopen_with_retry(Mock(), timeout=30)
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()
