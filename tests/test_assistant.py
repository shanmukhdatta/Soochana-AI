"""Tests for the Bujji AI Financial Assistant endpoint and service."""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import httpx

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_assistant_chat_no_api_key(client):
    """When GROQ_API_KEY is not set, returns unavailable status rather than failing."""
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
        res = client.post("/assistant/chat", json={"message": "What is RiskPulse AI?"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "unavailable"
        assert "temporarily unavailable" in data["reply"]


def test_assistant_chat_validation_error(client):
    """Empty message should trigger standard Pydantic validation error."""
    res = client.post("/assistant/chat", json={"message": ""})
    assert res.status_code == 422


def test_assistant_chat_mocked_success(client):
    """Mocked Groq API call returns structured response and preserves mode."""
    mock_groq_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "RiskPulse AI is an institutional research platform combining FinBERT and XGBoost.",
                }
            }
        ]
    }

    mock_resp = httpx.Response(200, json=mock_groq_response, request=httpx.Request("POST", "https://api.groq.com"))

    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GROQ_MODEL": "openai/gpt-oss-20b"}):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = client.post(
                "/assistant/chat",
                json={"message": "Explain RiskPulse AI", "mode": "quick", "history": []},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"
            assert "RiskPulse AI" in data["reply"]
            assert data["mode"] == "quick"
            assert "AI-generated explanations" in data["disclaimer"]


def test_assistant_chat_mocked_rate_limit(client):
    """Mocked 429 status code returns rate_limited status without leaking secrets."""
    mock_resp = httpx.Response(429, text="Rate limit exceeded", request=httpx.Request("POST", "https://api.groq.com"))

    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = client.post("/assistant/chat", json={"message": "What is ECL?", "mode": "analyst"})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "rate_limited"
            assert "rate limit" in data["reply"].lower()


def test_assistant_chat_mocked_timeout(client):
    """Network timeout gracefully yields error status."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}):
        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timed out")):
            res = client.post("/assistant/chat", json={"message": "What is ECL?", "mode": "quick"})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "error"
            assert "timed out" in data["reply"].lower()
