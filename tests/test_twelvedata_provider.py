"""
Tests for the Twelve Data provider — added as a real, authenticated
fallback market data source after a real deployment found both
yfinance and Stooq returning HTTP 429s from Render's shared cloud IP
range (confirmed even after adding a realistic browser User-Agent,
which ruled out headers as the cause).
"""
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def test_no_api_key_fails_gracefully_not_a_crash():
    """A local setup with no TWELVE_DATA_API_KEY set must not error —
    yfinance and Stooq keep working there exactly as before, since
    they aren't blocked from a home internet connection."""
    from backend.config import settings
    from backend.data.twelvedata_provider import TwelveDataProvider

    original_key = settings.TWELVE_DATA_API_KEY
    settings.TWELVE_DATA_API_KEY = ""
    try:
        provider = TwelveDataProvider()
        result = provider.get_historical("AAPL", period="1y")
        assert result.success is False
        assert "API key" in result.error
    finally:
        settings.TWELVE_DATA_API_KEY = original_key


def test_get_historical_success_maps_real_api_shape_correctly():
    """Verifies the actual, documented Twelve Data /time_series response
    shape (string-typed OHLCV values under a 'values' list) is parsed
    into the same DataFrame shape every other provider returns."""
    from backend.config import settings
    from backend.data.twelvedata_provider import TwelveDataProvider

    original_key = settings.TWELVE_DATA_API_KEY
    settings.TWELVE_DATA_API_KEY = "test_key_123"
    try:
        real_shaped_response = {
            "meta": {"symbol": "AAPL", "interval": "1day", "currency": "USD"},
            "values": [
                {"datetime": "2024-01-02", "open": "185.64", "high": "186.95", "low": "185.01", "close": "185.64", "volume": "82488700"},
                {"datetime": "2024-01-03", "open": "184.22", "high": "185.88", "low": "183.43", "close": "184.25", "volume": "58414500"},
            ],
            "status": "ok",
        }
        with patch("requests.get", return_value=_mock_response(json_data=real_shaped_response)) as mock_get:
            provider = TwelveDataProvider()
            result = provider.get_historical("AAPL", period="1y")

            assert result.success is True
            assert list(result.data.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
            assert len(result.data) == 2
            assert result.data["close"].iloc[0] == pytest.approx(185.64)
            assert result.data["volume"].iloc[0] == 82488700
            # Values must be genuinely numeric, not left as strings from the API
            assert result.data["close"].dtype.kind == "f"

            call_kwargs = mock_get.call_args.kwargs
            assert call_kwargs["params"]["apikey"] == "test_key_123"
            assert call_kwargs["params"]["adjust"] == "splits"
    finally:
        settings.TWELVE_DATA_API_KEY = original_key


def test_api_error_status_fails_gracefully_with_real_message():
    """Twelve Data reports errors via a status field in a 200 response,
    not always an HTTP error code — this must be checked explicitly."""
    from backend.config import settings
    from backend.data.twelvedata_provider import TwelveDataProvider

    original_key = settings.TWELVE_DATA_API_KEY
    settings.TWELVE_DATA_API_KEY = "test_key_123"
    try:
        error_response = {"code": 429, "message": "You have run out of API credits for the day.", "status": "error"}
        with patch("requests.get", return_value=_mock_response(json_data=error_response)):
            provider = TwelveDataProvider()
            result = provider.get_historical("AAPL", period="1y")
            assert result.success is False
            assert "API credits" in result.error
    finally:
        settings.TWELVE_DATA_API_KEY = original_key


def test_network_failure_fails_gracefully():
    from backend.config import settings
    from backend.data.twelvedata_provider import TwelveDataProvider
    import requests

    original_key = settings.TWELVE_DATA_API_KEY
    settings.TWELVE_DATA_API_KEY = "test_key_123"
    try:
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("network down")):
            provider = TwelveDataProvider()
            result = provider.get_historical("AAPL", period="1y")
            assert result.success is False
    finally:
        settings.TWELVE_DATA_API_KEY = original_key


def test_registered_as_a_third_fallback_provider():
    """Confirms this provider is actually wired into the app's existing
    multi-provider fallback chain, not just sitting unused."""
    from backend.data.market_data_service import MarketDataService
    service = MarketDataService()
    provider_names = [p.name for p in service.providers]
    assert "twelvedata" in provider_names
    # Ordered after the existing two so local setups (where yfinance/Stooq
    # already work fine) barely ever need to reach it.
    assert provider_names.index("twelvedata") > provider_names.index("yfinance")
