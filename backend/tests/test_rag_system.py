"""
test_rag_system.py — Unit and integration tests for RAGSystem.query().

RAGSystem.query() is the main orchestrator method:
  1. Wraps query in a prompt
  2. Fetches conversation history (if session_id provided)
  3. Calls AIGenerator.generate_response() with tools
  4. Collects sources from ToolManager
  5. Resets sources
  6. Updates session history
  7. Returns (response_text, sources_list)

Unit tests mock both chromadb.PersistentClient and anthropic.Anthropic.
Integration tests use real Anthropic API (requires ANTHROPIC_API_KEY).

IMPORTANT: The integration test `test_rag_query_fails_with_invalid_model_name`
is a REGRESSION ANCHOR that documents the production bug — it remains valid even
after fixing config.py, since it directly uses the bad model name.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch


# ── Unit test fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def rag_system_with_mocks(mock_anthropic_client):
    """
    RAGSystem with chromadb.PersistentClient and anthropic.Anthropic mocked.
    Also mocks SentenceTransformerEmbeddingFunction to avoid loading the model.
    Overrides ANTHROPIC_MODEL to the known-good 'claude-sonnet-4-6'.

    Returns (rag_system, mock_anthropic_client).
    """
    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = MagicMock()

    with patch("chromadb.PersistentClient", return_value=mock_chroma), \
         patch(
             "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
         ):
        # Import inside the patch context so the patches apply to VectorStore.__init__
        # Clear cached modules so re-import picks up fresh state
        for mod in ["rag_system", "vector_store", "ai_generator", "search_tools",
                    "session_manager", "document_processor", "config"]:
            sys.modules.pop(mod, None)

        from config import Config
        from rag_system import RAGSystem

        cfg = Config()
        cfg.ANTHROPIC_API_KEY = "sk-test"
        cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
        cfg.CHROMA_PATH = "/tmp/test_rag_unit"

        rag = RAGSystem(cfg)

    return rag, mock_anthropic_client


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestRAGSystemQueryUnit:
    """Unit tests: RAGSystem.query() logic without real API or DB calls."""

    def test_query_returns_two_element_tuple(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        query() must return a 2-tuple of (str, list).

        Failure means: app.py's `answer, sources = rag_system.query(...)` crashes
        with ValueError (too many / too few values to unpack).
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Python is great!")

        result = rag.query("What is Python?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_returns_ai_generator_response_text(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        The first element of the return tuple must be the text from AIGenerator.

        Failure means: response is empty or contains the raw prompt instead of
        Claude's answer.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Variables store data.")

        response, _ = rag.query("What are variables?")
        assert response == "Variables store data."

    def test_query_wraps_user_query_in_prompt(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        RAGSystem.query() prepends 'Answer this question about course materials:'
        to the query before passing to AIGenerator.

        Failure means: the AI receives the bare user query without framing context.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        rag.query("raw user question")
        call_kwargs = mock_client.messages.create.call_args[1]
        user_message = call_kwargs["messages"][0]["content"]
        assert "Answer this question about course materials:" in user_message
        assert "raw user question" in user_message

    def test_query_passes_tool_definitions_to_ai_generator(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        query() must pass tool definitions so Claude can call search_course_content.

        Failure means: every question uses only Claude's training data, completely
        ignoring uploaded course content.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        rag.query("What is in lesson 1?")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t.get("name") for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_returns_empty_sources_when_no_tool_used(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        When no tool use occurs, sources must be an empty list (not None).

        Failure means: app.py's QueryResponse(sources=None) fails Pydantic
        validation, causing a 500 error on every non-content query.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        _, sources = rag.query("General question")
        assert sources == []

    def test_query_stores_exchange_in_session_history(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        After query(), the user question and assistant response must be stored
        in session history for multi-turn conversation context.

        Failure means: follow-up questions have no context.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Python is great!")

        session_id = rag.session_manager.create_session()
        rag.query("What is Python?", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is Python?" in history
        assert "Python is great!" in history

    def test_query_resets_sources_after_retrieval(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        Sources must be reset after each query so previous sources don't contaminate
        subsequent queries.

        Failure means: a search query's sources appear in the next query's response
        even when the next query uses no tools.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        rag.query("First query")
        assert rag.search_tool.last_sources == []

    def test_query_with_session_id_fetches_history(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        When session_id is provided and history exists, it must be included in the
        system prompt for context.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        session_id = rag.session_manager.create_session()
        # Add prior exchange to history
        rag.session_manager.add_exchange(session_id, "Prior question", "Prior answer")

        rag.query("Follow-up question", session_id=session_id)
        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "Prior question" in system or "Prior answer" in system

    def test_query_without_session_id_skips_history_lookup(
        self, rag_system_with_mocks, make_text_response
    ):
        """
        When no session_id is provided, generate_response() must be called with
        conversation_history=None.

        Failure means: a non-None value is passed for history, which may include
        stale data from another session or cause an error.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.return_value = make_text_response("Answer")

        # Spy on generate_response
        original = rag.ai_generator.generate_response
        call_history = []

        def spy_generate(*args, **kwargs):
            call_history.append(kwargs.get("conversation_history"))
            return original(*args, **kwargs)

        rag.ai_generator.generate_response = spy_generate
        rag.query("Question without session")

        assert len(call_history) == 1
        assert call_history[0] is None

    def test_query_propagates_api_errors_to_caller(
        self, rag_system_with_mocks
    ):
        """
        When AIGenerator raises (e.g., NotFoundError for invalid model name),
        query() must let the exception propagate rather than swallowing it.

        This mirrors the production bug:
          ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'
          → API 404 → NotFoundError
          → propagates from query() → caught in app.py → HTTPException(500)
          → frontend shows "Error: Query failed"

        Fix: change ANTHROPIC_MODEL to 'claude-sonnet-4-6' in config.py line 13.
        """
        rag, mock_client = rag_system_with_mocks
        mock_client.messages.create.side_effect = Exception(
            "Error code: 404 — {'type': 'error', 'error': "
            "{'type': 'not_found_error', 'message': 'model: claude-sonnet-4-20250514'}}"
        )

        with pytest.raises(Exception, match="not_found_error|404"):
            rag.query("What is Python?")


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
class TestRAGSystemQueryLiveServer:
    """
    Reproduces exactly what the running server does:
    real persistent ChromaDB with actual loaded course data + real Anthropic API.
    """

    @pytest.fixture
    def live_rag_system(self):
        """RAGSystem pointing at the real on-disk ChromaDB (same as the server)."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        for mod in ["rag_system", "vector_store", "ai_generator", "search_tools",
                    "session_manager", "document_processor", "config"]:
            sys.modules.pop(mod, None)

        # Use the real config (persistent chroma_db on disk, correct model)
        import os as _os
        _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

        from config import config
        from rag_system import RAGSystem

        config.ANTHROPIC_API_KEY = api_key
        return RAGSystem(config)

    def test_content_query_with_real_chroma_and_api(self, live_rag_system):
        """
        Sends a content-specific question through the REAL stack:
        persistent ChromaDB → CourseSearchTool → Anthropic API → response.

        This reproduces exactly what the running server does. Any exception
        raised here is the same one causing HTTPException(500) in production.
        """
        try:
            response, sources = live_rag_system.query(
                "What topics are covered in the courses?"
            )
            assert isinstance(response, str)
            assert len(response.strip()) > 0
        except Exception as e:
            pytest.fail(
                f"Real content query raised {type(e).__name__}: {e}\n"
                f"This is the exception causing HTTP 500 / 'query failed' in the browser."
            )

    def test_general_query_with_real_api(self, live_rag_system):
        """General question (no tool use expected) — baseline check."""
        try:
            response, _ = live_rag_system.query("What is machine learning?")
            assert isinstance(response, str) and len(response.strip()) > 0
        except Exception as e:
            pytest.fail(f"General query failed with {type(e).__name__}: {e}")


@pytest.mark.integration
class TestRAGSystemQueryIntegration:
    """
    Integration tests: real Anthropic API + in-memory ChromaDB.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    @pytest.fixture
    def real_rag_with_correct_model(self):
        """
        RAGSystem with real Anthropic API and in-memory ChromaDB.
        Uses the correct model name (claude-sonnet-4-6).
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import chromadb

        # Clear cached modules so fresh import picks up the patch
        for mod in ["rag_system", "vector_store", "ai_generator", "search_tools",
                    "session_manager", "document_processor", "config"]:
            sys.modules.pop(mod, None)

        eph = chromadb.EphemeralClient()
        with patch("chromadb.PersistentClient", return_value=eph):
            from config import Config
            from rag_system import RAGSystem

            cfg = Config()
            cfg.ANTHROPIC_API_KEY = api_key
            cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
            cfg.CHROMA_PATH = "/tmp/rag_integration_test"

            rag = RAGSystem(cfg)
        return rag

    def test_rag_query_fails_with_invalid_model_name(self):
        """
        REGRESSION ANCHOR: Documents the production bug.

        Using model='claude-sonnet-4-20250514' must raise NotFoundError(404).
        This test passes on the CURRENT BROKEN codebase (error is raised as expected)
        AND continues to pass after the config.py fix (since it hardcodes the bad model).

        Fix for production: change ANTHROPIC_MODEL to 'claude-sonnet-4-6' in config.py:13.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        import chromadb

        for mod in ["rag_system", "vector_store", "ai_generator", "search_tools",
                    "session_manager", "document_processor", "config"]:
            sys.modules.pop(mod, None)

        eph = chromadb.EphemeralClient()
        with patch("chromadb.PersistentClient", return_value=eph):
            from config import Config
            from rag_system import RAGSystem

            cfg = Config()
            cfg.ANTHROPIC_API_KEY = api_key
            cfg.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # THE BROKEN VALUE
            cfg.CHROMA_PATH = "/tmp/rag_bad_model_test"

            rag = RAGSystem(cfg)

        with pytest.raises((anthropic.NotFoundError, Exception)) as exc_info:
            rag.query("What is Python?")

        error_str = str(exc_info.value).lower()
        assert "404" in error_str or "not_found" in error_str or "model" in error_str, (
            f"Expected a 404/not_found error for invalid model name, got: {exc_info.value}"
        )

    def test_rag_query_succeeds_with_correct_model_name(
        self, real_rag_with_correct_model
    ):
        """
        With the correct model name ('claude-sonnet-4-6'), query() must:
        - Not raise any exception
        - Return a non-empty string response
        - Return a list for sources

        Failure after the fix means: something else is broken in the chain.
        """
        rag = real_rag_with_correct_model
        response, sources = rag.query("What is machine learning?")

        assert isinstance(response, str)
        assert len(response.strip()) > 0, "Response must be non-empty"
        assert isinstance(sources, list)

    def test_rag_query_session_stores_history(self, real_rag_with_correct_model):
        """
        After a real query, the session must contain the exchange.
        Confirms the full round-trip: API → response → session storage.
        """
        rag = real_rag_with_correct_model
        session_id = rag.session_manager.create_session()

        rag.query("What is deep learning?", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "deep learning" in history.lower()
