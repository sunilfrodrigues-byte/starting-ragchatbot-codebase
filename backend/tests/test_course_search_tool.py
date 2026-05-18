"""
test_course_search_tool.py — Unit tests for CourseSearchTool.execute().

CourseSearchTool is the bridge between the Anthropic tool-use API and ChromaDB.
When Claude invokes "search_course_content", ToolManager dispatches to this class.
If it misbehaves, the AI either gets wrong data or an error string instead of course
content, producing incorrect or empty answers.

All tests use mock_vector_store from conftest.py. No real ChromaDB or API calls.
"""

import pytest
from unittest.mock import MagicMock


class TestCourseSearchToolHappyPath:
    """Tests for successful search scenarios."""

    def test_execute_returns_formatted_string_on_success(self, course_search_tool):
        """
        execute() must return a non-empty string when the store finds results.

        Failure means: the tool returns None or raises, so the AI gets no context
        and produces hallucinated or empty answers.
        """
        result = course_search_tool.execute(query="what are Python variables?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_includes_document_content_in_output(self, course_search_tool):
        """
        The formatted output must contain the actual document text from the store.

        Failure means: _format_results() strips content, so the AI gets headers
        with no educational material.
        """
        result = course_search_tool.execute(query="variables")
        assert "Python variables store data" in result

    def test_execute_includes_course_title_header(self, course_search_tool):
        """
        Each result block must have a [CourseName] header for source attribution.

        Failure means: the AI cannot identify which course the result came from,
        breaking UI citation.
        """
        result = course_search_tool.execute(query="variables")
        assert "[Introduction to Python" in result

    def test_execute_includes_lesson_number_in_header(self, course_search_tool):
        """
        When metadata contains lesson_number, the header must include 'Lesson N'.

        Failure means: lesson context is lost, making correct lesson citation impossible.
        """
        result = course_search_tool.execute(query="variables")
        assert "Lesson 1" in result

    def test_execute_passes_query_to_store_search(self, course_search_tool, mock_vector_store):
        """
        execute() must forward the query parameter to VectorStore.search().

        Failure means: the vector search runs with the wrong or empty query,
        returning irrelevant results.
        """
        course_search_tool.execute(query="function definitions")
        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["query"] == "function definitions"

    def test_execute_passes_course_name_filter_to_store(self, course_search_tool, mock_vector_store):
        """
        When course_name is provided, it must be forwarded to VectorStore.search().

        Failure means: course filtering is silently dropped, returning results
        from all courses instead of the requested one.
        """
        course_search_tool.execute(query="variables", course_name="Python Basics")
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["course_name"] == "Python Basics"

    def test_execute_passes_lesson_number_filter_to_store(self, course_search_tool, mock_vector_store):
        """
        When lesson_number is provided, it must be forwarded to VectorStore.search().

        Failure means: lesson filtering is silently dropped, returning content from
        all lessons instead of the requested one.
        """
        course_search_tool.execute(query="variables", lesson_number=2)
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["lesson_number"] == 2

    def test_execute_populates_last_sources_on_success(self, course_search_tool):
        """
        After a successful search, last_sources must be a non-empty list.

        Failure means: RAGSystem.query() returns empty sources to the frontend,
        breaking citation UI even when results exist.
        """
        course_search_tool.execute(query="variables")
        assert len(course_search_tool.last_sources) > 0
        assert "label" in course_search_tool.last_sources[0]

    def test_execute_source_label_includes_course_and_lesson(self, course_search_tool):
        """
        Source label format must be 'CourseName - Lesson N' for UI display.

        Failure means: the source citation is malformatted, breaking frontend link display.
        """
        course_search_tool.execute(query="variables")
        label = course_search_tool.last_sources[0]["label"]
        assert "Introduction to Python" in label
        assert "Lesson 1" in label

    def test_execute_source_url_fetched_from_store(self, course_search_tool, mock_vector_store):
        """
        Source URL must come from VectorStore.get_lesson_link().

        Failure means: all source URLs are None, breaking clickable citations.
        """
        course_search_tool.execute(query="variables")
        url = course_search_tool.last_sources[0]["url"]
        assert url == "https://example.com/python/lesson/1"
        mock_vector_store.get_lesson_link.assert_called_with("Introduction to Python", 1)

    def test_execute_no_url_when_no_lesson_number(self, course_search_tool, mock_vector_store):
        """
        When metadata has no lesson_number, url must be None (no lesson link to fetch).
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["General course intro."],
            metadata=[{"course_title": "Introduction to Python"}],
            distances=[0.1],
            error=None,
        )
        course_search_tool.execute(query="intro")
        url = course_search_tool.last_sources[0]["url"]
        assert url is None
        mock_vector_store.get_lesson_link.assert_not_called()


class TestCourseSearchToolEmptyResults:
    """Tests for the 'no results found' scenario."""

    def test_execute_returns_no_content_found_message_when_empty(
        self, course_search_tool, mock_vector_store
    ):
        """
        When search returns no documents, execute() must return a clear 'no content'
        message so the AI can report it rather than hallucinate.

        Failure means: execute() returns empty string or None, causing the AI to
        receive no useful signal and potentially fabricate an answer.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        result = course_search_tool.execute(query="nonexistent topic")
        assert "No relevant content found" in result

    def test_empty_result_with_course_filter_mentions_course_name(
        self, course_search_tool, mock_vector_store
    ):
        """
        When empty, the message must mention which course was searched so the AI
        can give an informative "not found in X" answer.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        result = course_search_tool.execute(query="quantum physics", course_name="Python Basics")
        assert "Python Basics" in result

    def test_empty_result_with_lesson_filter_mentions_lesson_number(
        self, course_search_tool, mock_vector_store
    ):
        """
        When empty with lesson filter, the message must mention which lesson.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        result = course_search_tool.execute(query="advanced topic", lesson_number=5)
        assert "lesson 5" in result.lower()

    def test_empty_result_does_not_populate_last_sources(
        self, course_search_tool, mock_vector_store
    ):
        """
        When search returns no documents, last_sources must remain empty.

        Failure means: stale sources from a previous search contaminate the next
        empty-result query, showing wrong citations in the UI.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        course_search_tool.execute(query="nothing here")
        assert course_search_tool.last_sources == []


class TestCourseSearchToolErrorHandling:
    """Tests for error propagation from VectorStore."""

    def test_execute_returns_error_string_on_store_error(
        self, course_search_tool, mock_vector_store
    ):
        """
        When VectorStore.search() returns SearchResults with error set,
        execute() must return that error string.

        Failure means: execute() raises an exception, which crashes
        _handle_tool_execution() in ai_generator.py, causing a 500 error.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[],
            error="Search error: ChromaDB collection not found",
        )
        result = course_search_tool.execute(query="variables")
        assert isinstance(result, str)
        assert "Search error" in result

    def test_execute_never_raises_on_store_error(self, course_search_tool, mock_vector_store):
        """
        execute() must NEVER raise — it always returns a string.
        ToolManager.execute_tool() callers depend on this contract.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="DB failure"
        )
        try:
            result = course_search_tool.execute(query="test")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(
                f"execute() raised {type(e).__name__} instead of returning error string: {e}"
            )

    def test_execute_propagates_course_not_found_error(
        self, course_search_tool, mock_vector_store
    ):
        """
        When course name resolution fails (VectorStore returns error), execute()
        must forward that error string.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[],
            error="No course found matching 'Nonexistent Course'",
        )
        result = course_search_tool.execute(query="content", course_name="Nonexistent Course")
        assert "No course found" in result


class TestCourseSearchToolMultipleResults:
    """Tests for multiple result handling."""

    def test_multiple_results_separated_by_blank_lines(
        self, course_search_tool, mock_vector_store
    ):
        """
        Multiple result blocks must be separated by blank lines so the AI can
        distinguish distinct content blocks.

        Failure means: concatenated results make it hard to attribute content.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Variables store data.", "Functions are reusable."],
            metadata=[
                {"course_title": "Python", "lesson_number": 1},
                {"course_title": "Python", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.get_lesson_link.return_value = None
        result = course_search_tool.execute(query="python basics")
        assert "\n\n" in result

    def test_multiple_results_populate_multiple_sources(
        self, course_search_tool, mock_vector_store
    ):
        """
        Each result document must contribute one entry to last_sources.

        Failure means: only first or last result generates a citation,
        giving incomplete source attribution in the UI.
        """
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Doc A.", "Doc B."],
            metadata=[
                {"course_title": "Python", "lesson_number": 1},
                {"course_title": "Python", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.get_lesson_link.return_value = None
        course_search_tool.execute(query="python")
        assert len(course_search_tool.last_sources) == 2
