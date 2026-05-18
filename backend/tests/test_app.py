"""
test_app.py — FastAPI HTTP endpoint tests.

Tests the full HTTP layer: request parsing → RAGSystem → response serialization.
Uses FastAPI's TestClient with real ChromaDB (persistent) + real Anthropic API.

Must be run from the backend/ directory (StaticFiles uses relative path ../frontend).

Run: uv run pytest tests/test_app.py -v -m integration
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch


# ── Unit smoke tests (mocked AI, no API calls) ────────────────────────────────

@pytest.fixture(scope="module")
def mocked_app_client():
    """
    TestClient with ChromaDB and Anthropic both mocked.
    Used for fast unit-level HTTP tests.
    """
    from starlette.testclient import TestClient

    for mod in list(sys.modules):
        if mod in ("app", "rag_system", "ai_generator", "vector_store",
                   "search_tools", "session_manager", "document_processor", "config"):
            del sys.modules[mod]

    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = MagicMock()

    with patch("chromadb.PersistentClient", return_value=mock_chroma), \
         patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"), \
         patch("anthropic.Anthropic") as mock_cls:

        mock_api = MagicMock()
        mock_cls.return_value = mock_api
        mock_resp = MagicMock()
        mock_resp.stop_reason = "end_turn"
        mock_resp.content = [MagicMock(text="Test answer", type="text")]
        mock_api.messages.create.return_value = mock_resp

        import app as app_module
        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client, mock_api


class TestQueryEndpointUnit:
    """Fast unit tests for the /api/query HTTP endpoint."""

    def test_returns_200_on_success(self, mocked_app_client):
        """POST /api/query must return HTTP 200 when the backend succeeds."""
        client, _ = mocked_app_client
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_response_has_answer_field(self, mocked_app_client):
        """Response JSON must have a non-empty 'answer' field."""
        client, _ = mocked_app_client
        resp = client.post("/api/query", json={"query": "What is Python?"})
        body = resp.json()
        assert "answer" in body
        assert isinstance(body["answer"], str) and body["answer"]

    def test_response_has_sources_list(self, mocked_app_client):
        """Response JSON must have a 'sources' list (may be empty)."""
        client, _ = mocked_app_client
        resp = client.post("/api/query", json={"query": "Question"})
        body = resp.json()
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_response_has_session_id(self, mocked_app_client):
        """Response JSON must have a non-empty 'session_id' string."""
        client, _ = mocked_app_client
        resp = client.post("/api/query", json={"query": "Question"})
        body = resp.json()
        assert "session_id" in body
        assert isinstance(body["session_id"], str) and body["session_id"]

    def test_provided_session_id_preserved(self, mocked_app_client):
        """When request includes session_id, response must echo it back."""
        client, _ = mocked_app_client
        resp = client.post(
            "/api/query", json={"query": "Question", "session_id": "session_42"}
        )
        assert resp.json()["session_id"] == "session_42"

    def test_missing_query_returns_422(self, mocked_app_client):
        """Request without 'query' field must return 422 (Pydantic validation)."""
        client, _ = mocked_app_client
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_api_error_returns_500(self, mocked_app_client):
        """When the backend raises, endpoint must return 500 with detail."""
        client, mock_api = mocked_app_client
        mock_api.messages.create.side_effect = Exception("boom")
        resp = client.post("/api/query", json={"query": "Question"})
        assert resp.status_code == 500
        assert "detail" in resp.json()
        # Restore for other tests
        mock_resp = MagicMock()
        mock_resp.stop_reason = "end_turn"
        mock_resp.content = [MagicMock(text="Test answer", type="text")]
        mock_api.messages.create.side_effect = None
        mock_api.messages.create.return_value = mock_resp


# ── Integration test: full real stack ─────────────────────────────────────────

@pytest.mark.integration
class TestQueryEndpointIntegration:
    """
    Full end-to-end test: real ChromaDB (persistent, with course data) + real Anthropic API.
    Reproduces exactly what the browser experiences when submitting a query.
    """

    @pytest.fixture(scope="class")
    def real_app_client(self):
        """TestClient backed by the real on-disk ChromaDB and real Anthropic API."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        from starlette.testclient import TestClient

        for mod in list(sys.modules):
            if mod in ("app", "rag_system", "ai_generator", "vector_store",
                       "search_tools", "session_manager", "document_processor", "config"):
                del sys.modules[mod]

        import app as app_module
        return TestClient(app_module.app, raise_server_exceptions=False)

    def test_general_question_returns_200(self, real_app_client):
        """
        A general knowledge question must return HTTP 200.

        Failure means: the HTTP layer or model name is still broken.
        """
        resp = real_app_client.post(
            "/api/query", json={"query": "What is machine learning?"}
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}.\n"
            f"Response: {resp.text}\n"
            f"This is the same error the browser sees as 'Error: Query failed'."
        )
        body = resp.json()
        assert isinstance(body["answer"], str) and body["answer"]

    def test_content_question_returns_200(self, real_app_client):
        """
        A course-content question (triggers tool use) must return HTTP 200.

        Failure here while test_general_question_returns_200 passes means:
        the bug is specifically in the tool-use execution path.
        """
        resp = real_app_client.post(
            "/api/query", json={"query": "What topics are covered in the courses?"}
        )
        assert resp.status_code == 200, (
            f"Content query failed with {resp.status_code}.\n"
            f"Response body: {resp.text}\n"
            f"The tool-use path is broken — check _handle_tool_execution in ai_generator.py."
        )
        body = resp.json()
        assert isinstance(body["answer"], str) and body["answer"]

    def test_sources_field_is_valid_list(self, real_app_client):
        """Sources must be a list (may be empty for non-content questions)."""
        resp = real_app_client.post(
            "/api/query", json={"query": "What is Python?"}
        )
        if resp.status_code == 200:
            assert isinstance(resp.json()["sources"], list)
