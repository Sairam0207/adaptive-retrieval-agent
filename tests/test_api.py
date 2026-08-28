from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import main


def test_health() -> None:
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_query_rejects_blank_question() -> None:
    client = TestClient(main.app)
    with patch.object(main.settings, "api_key", ""):
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422


def test_query_requires_api_key_when_configured() -> None:
    client = TestClient(main.app)
    with patch.object(main.settings, "api_key", "secret123"):
        resp = client.post("/query", json={"question": "What is FastAPI?"})
        assert resp.status_code == 401

        resp = client.post(
            "/query",
            json={"question": "What is FastAPI?"},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401


def test_query_returns_503_on_agent_failure() -> None:
    client = TestClient(main.app)
    with (
        patch.object(main.settings, "api_key", ""),
        patch.object(main, "ask", side_effect=RuntimeError("both models down")),
    ):
        resp = client.post("/query", json={"question": "What is FastAPI?"})
        assert resp.status_code == 503
