"""
conftest.py — shared fixtures and sys.path setup.

WHY sys.path manipulation is needed:
  backend/*.py files use bare imports (e.g. `from vector_store import VectorStore`).
  They have no package structure, so Python cannot find them unless backend/ is on
  sys.path. This conftest adds it before any test module imports happen.

HOW to run the suite:
  From the backend/ directory:
      uv run pytest tests/ -v
      uv run pytest tests/ -v -m "not integration"   # unit tests only
      uv run pytest tests/ -v -m integration          # real-API tests only

ENVIRONMENT:
  ANTHROPIC_API_KEY must be set for integration tests. Unit tests mock everything.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── sys.path setup ────────────────────────────────────────────────────────────
# __file__ is backend/tests/conftest.py → parent is backend/tests/ → parent is backend/
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ── pytest marker registration ────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that make real API/DB calls (requires ANTHROPIC_API_KEY)",
    )


# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store():
    """
    MagicMock standing in for VectorStore. Default: search() returns one result
    doc about "Introduction to Python, Lesson 1".

    Tests needing a different return value should override in the test body:
        mock_vector_store.search.return_value = SearchResults(...)
    """
    from vector_store import SearchResults

    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=["Python variables store data values. x = 5 creates an integer."],
        metadata=[{"course_title": "Introduction to Python", "lesson_number": 1}],
        distances=[0.05],
        error=None,
    )
    store.get_lesson_link.return_value = "https://example.com/python/lesson/1"
    return store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool wired to mock_vector_store."""
    from search_tools import CourseSearchTool

    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def mock_anthropic_client():
    """
    Patches anthropic.Anthropic at the module level and yields the mock client
    instance. Callers configure mock_anthropic_client.messages.create.return_value
    or .side_effect to control the API response.
    """
    with patch("anthropic.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def make_text_response():
    """
    Factory fixture: returns a callable that produces a mock Anthropic response
    simulating stop_reason='end_turn' with a single text content block.

    Usage:
        mock_anthropic_client.messages.create.return_value = make_text_response("Hello!")
    """

    def _factory(text: str):
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = text

        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [content_block]
        return response

    return _factory


@pytest.fixture
def make_tool_use_response():
    """
    Factory fixture: returns a callable that produces a mock Anthropic response
    simulating stop_reason='tool_use'. Uses MagicMock objects (not real SDK types)
    to avoid version-specific constructor requirements.

    Usage:
        first = make_tool_use_response("search_course_content", {"query": "..."})
        mock_client.messages.create.side_effect = [first, second]
    """

    def _factory(tool_name: str, tool_input: dict, tool_id: str = "tool_abc123"):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search for that."

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = tool_id
        tool_use_block.name = tool_name
        tool_use_block.input = tool_input

        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [text_block, tool_use_block]
        return response

    return _factory
