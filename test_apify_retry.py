"""Unit tests for fetch_tweets.apify_request retry behavior.
Run:
  APIFY_TOKEN=dummy OPENROUTER_API_KEY=dummy python3 -m pytest test_apify_retry.py -v

Why a separate file: keeps llm_scoring tests (test_scoring.py) focused on LLM logic,
and Apify retry tests focused on HTTP/network resilience.
"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

os.environ.setdefault("APIFY_TOKEN", "dummy")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

import fetch_tweets


def _http_error(code: int) -> HTTPError:
    """Build a real HTTPError with a readable body (for the .read() in error path)."""
    body = b'{"error": {"message": "test"}}'
    return HTTPError("http://test", code, "test", {}, BytesIO(body))


def _success_response(payload: bytes = b'{"data": {"id": "t1"}}'):
    """Build a urlopen() context-manager-style mock that returns payload."""
    resp = MagicMock()
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: None
    resp.read = lambda: payload
    return resp


class TestApifyRetry:
    def test_succeeds_first_try(self):
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep") as mock_sleep:
            mock_open.return_value = _success_response(b'{"data": {"id": "abc"}}')
            result = fetch_tweets.apify_request("GET", "/test")
        assert result == {"data": {"id": "abc"}}
        assert mock_open.call_count == 1
        assert mock_sleep.call_count == 0

    def test_retries_on_502_then_succeeds(self):
        """The exact 502 case that killed our 4/24 and 4/26 evening cron runs."""
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep") as mock_sleep:
            mock_open.side_effect = [
                _http_error(502),
                _success_response(b'{"data": {"id": "xyz"}}'),
            ]
            result = fetch_tweets.apify_request("POST", "/acts/foo/runs", {"a": 1})
        assert result == {"data": {"id": "xyz"}}
        assert mock_open.call_count == 2
        # Slept once (1s backoff between attempt 1 and 2)
        mock_sleep.assert_called_once_with(1)

    def test_retries_on_503_then_504_then_succeeds(self):
        """All 5xx codes in APIFY_RETRY_HTTP_CODES should retry."""
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep") as mock_sleep:
            mock_open.side_effect = [
                _http_error(503),
                _http_error(504),
                _success_response(),
            ]
            result = fetch_tweets.apify_request("GET", "/test")
        assert result == {"data": {"id": "t1"}}
        assert mock_open.call_count == 3
        # Exponential backoff: 1s, then 2s
        assert mock_sleep.call_args_list == [((1,),), ((2,),)]

    def test_does_not_retry_on_401(self):
        """4xx errors are real client errors — fail fast, don't waste time."""
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep") as mock_sleep:
            mock_open.side_effect = _http_error(401)
            try:
                fetch_tweets.apify_request("GET", "/test")
                assert False, "expected HTTPError to propagate"
            except HTTPError as e:
                assert e.code == 401
        assert mock_open.call_count == 1
        assert mock_sleep.call_count == 0

    def test_does_not_retry_on_404(self):
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep"):
            mock_open.side_effect = _http_error(404)
            try:
                fetch_tweets.apify_request("GET", "/test")
                assert False
            except HTTPError as e:
                assert e.code == 404
        assert mock_open.call_count == 1

    def test_gives_up_after_max_retries_on_persistent_502(self):
        """If 502 persists across all attempts, raise the final HTTPError."""
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep"):
            mock_open.side_effect = [_http_error(502)] * 5  # extra to be safe
            try:
                fetch_tweets.apify_request("GET", "/test", max_retries=2)
                assert False, "expected final HTTPError"
            except HTTPError as e:
                assert e.code == 502
        # max_retries=2 → 3 total attempts (initial + 2 retries)
        assert mock_open.call_count == 3

    def test_retries_on_network_urlerror(self):
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep"):
            mock_open.side_effect = [
                URLError("connection refused"),
                _success_response(),
            ]
            result = fetch_tweets.apify_request("GET", "/test")
        assert result == {"data": {"id": "t1"}}
        assert mock_open.call_count == 2

    def test_retries_on_timeout(self):
        with patch("fetch_tweets.urlopen") as mock_open, \
             patch("fetch_tweets.time.sleep"):
            mock_open.side_effect = [
                TimeoutError("read timeout"),
                _success_response(),
            ]
            result = fetch_tweets.apify_request("GET", "/test")
        assert result == {"data": {"id": "t1"}}
        assert mock_open.call_count == 2
