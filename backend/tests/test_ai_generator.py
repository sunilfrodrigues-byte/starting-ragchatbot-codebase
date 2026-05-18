"""
test_ai_generator.py — Unit tests for AIGenerator tool-calling paths.

Tests two execution paths in AIGenerator.generate_response():
  1. Direct path: Claude responds with stop_reason='end_turn' → return text
  2. Tool use path: stop_reason='tool_use' → _handle_tool_execution()
       → execute tools → second API call → return final text

All tests mock anthropic.Anthropic so no real API calls are made.
The mock_anthropic_client fixture from conftest.py patches 'anthropic.Anthropic'
because ai_generator.py does `import anthropic; anthropic.Anthropic(api_key=...)`.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestAIGeneratorInitialization:
    """Tests for AIGenerator.__init__."""

    def test_init_creates_anthropic_client_with_key(self):
        """
        AIGenerator.__init__ must call anthropic.Anthropic(api_key=...) with the key.

        Failure means: client created without auth, causing AuthenticationError.
        """
        with patch("anthropic.Anthropic") as mock_class:
            from ai_generator import AIGenerator

            AIGenerator(api_key="sk-test-key", model="claude-sonnet-4-6")
            mock_class.assert_called_once_with(api_key="sk-test-key")

    def test_init_stores_model_name(self):
        """The model name must be stored and used in subsequent API calls."""
        with patch("anthropic.Anthropic"):
            from ai_generator import AIGenerator

            gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
            assert gen.model == "claude-sonnet-4-6"

    def test_base_params_include_required_fields(self):
        """base_params must include model, max_tokens, and temperature."""
        with patch("anthropic.Anthropic"):
            from ai_generator import AIGenerator

            gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
            assert gen.base_params["model"] == "claude-sonnet-4-6"
            assert "max_tokens" in gen.base_params
            assert "temperature" in gen.base_params


class TestAIGeneratorDirectResponsePath:
    """Tests for the direct response path (no tool use)."""

    def test_returns_text_when_stop_reason_is_end_turn(
        self, mock_anthropic_client, make_text_response
    ):
        """
        When Claude returns stop_reason='end_turn', generate_response() must return
        the text from content[0].text.

        Failure means: direct responses return None or raise AttributeError,
        breaking all general knowledge questions.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response("Paris")
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response("What is the capital of France?")
        assert result == "Paris"

    def test_sends_query_as_user_role_message(
        self, mock_anthropic_client, make_text_response
    ):
        """
        The query must be sent as role='user' in the messages list.

        Failure means: the query is lost or misrouted, causing Claude to answer
        a blank question.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Test question?")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert "Test question?" in messages[0]["content"]

    def test_includes_system_prompt_in_api_call(
        self, mock_anthropic_client, make_text_response
    ):
        """
        Every API call must include a non-empty system prompt.

        Failure means: Claude has no instruction context, giving off-topic responses.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_appends_history_to_system_when_provided(
        self, mock_anthropic_client, make_text_response
    ):
        """
        Conversation history must appear in the system prompt when provided.

        Failure means: multi-turn conversations lose context, breaking follow-ups.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Follow-up?", conversation_history="User: Hello\nAssistant: Hi"
        )

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "Hello" in call_kwargs["system"]
        assert "Hi" in call_kwargs["system"]

    def test_omits_tools_when_none_provided(
        self, mock_anthropic_client, make_text_response
    ):
        """
        When tools=None, the API call must NOT include a 'tools' key.

        Failure means: Claude receives an empty tools list and may behave unexpectedly.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question", tools=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_includes_tools_when_provided(
        self, mock_anthropic_client, make_text_response
    ):
        """
        When tools list is provided, it must appear in the first API call.

        Failure means: Claude doesn't know about available tools and cannot
        perform course content searches.
        """
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        tool_defs = [{"name": "search_course_content", "description": "Search"}]
        gen.generate_response("Question", tools=tool_defs)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tool_defs

    def test_makes_exactly_one_api_call_on_direct_response(
        self, mock_anthropic_client, make_text_response
    ):
        """Only one API call should be made when no tool use occurs."""
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Answer"
        )
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question")
        assert mock_anthropic_client.messages.create.call_count == 1


class TestAIGeneratorToolUsePath:
    """
    Tests for the tool use path — the critical path that handles:
    first API call → tool_use → execute tool → second API call → final text.
    """

    def test_tool_use_triggers_two_api_calls(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        When first response has stop_reason='tool_use', a second API call must be made.

        Failure means: the tool use response is treated as a final answer, returning
        the TextBlock text ("Let me search...") or None instead of the synthesized answer.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        first = make_tool_use_response("search_course_content", {"query": "python"})
        second = make_text_response("Python is a programming language.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_tool_use_calls_tool_manager_with_correct_name(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        _handle_tool_execution must call tool_manager.execute_tool() with the tool
        name from the ToolUseBlock.

        Failure means: the tool dispatcher receives the wrong tool name and returns
        'Tool not found', giving the AI an error instead of search results.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response("search_course_content", {"query": "functions"})
        second = make_text_response("Functions are reusable code blocks.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Explain functions", tools=[{}], tool_manager=mock_tool_manager
        )

        call_args = mock_tool_manager.execute_tool.call_args
        assert call_args[0][0] == "search_course_content"

    def test_tool_use_unpacks_input_as_kwargs(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        Tool input dict from ToolUseBlock.input must be unpacked as **kwargs
        to tool_manager.execute_tool().

        Failure means: the tool receives no arguments (or a dict argument instead
        of keyword args), causing TypeError in CourseSearchTool.execute().
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response(
            "search_course_content",
            {"query": "OOP concepts", "course_name": "Python OOP", "lesson_number": 3},
        )
        second = make_text_response("OOP uses classes and objects.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Explain OOP", tools=[{}], tool_manager=mock_tool_manager)

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="OOP concepts",
            course_name="Python OOP",
            lesson_number=3,
        )

    def test_second_api_call_contains_tool_result_message(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        The second API call must include a user message with the tool_result,
        linking back to the tool_use_id so Claude can synthesize an answer.

        Failure means: Claude receives no tool results and cannot use search data.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Variables store data values."

        first = make_tool_use_response(
            "search_course_content", {"query": "variables"}, tool_id="tool_xyz789"
        )
        second = make_text_response("Python variables store values.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "What are variables?", tools=[{}], tool_manager=mock_tool_manager
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find the tool_result user message
        tool_result_msgs = [
            m
            for m in messages
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in m["content"]
            )
        ]
        assert (
            len(tool_result_msgs) == 1
        ), "Second API call must contain exactly one tool_result user message"
        tool_result = tool_result_msgs[0]["content"][0]
        assert tool_result["tool_use_id"] == "tool_xyz789"
        assert tool_result["content"] == "Variables store data values."

    def test_intermediate_call_includes_tools_when_round_available(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After round 1 tool use, the follow-up API call must include 'tools' so
        Claude can optionally make a second sequential tool call.

        Failure means: Claude cannot perform a second search even when one is needed,
        breaking multi-step queries.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response("search_course_content", {"query": "test"})
        second = make_text_response("Final answer")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

    def test_returns_final_response_text_after_tool_use(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After the full tool use cycle, generate_response() must return the text
        from the SECOND API call's response, not the tool result string.

        Failure means: the raw search result string is returned instead of Claude's
        synthesized answer.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Raw search results"

        first = make_tool_use_response("search_course_content", {"query": "test"})
        second = make_text_response("Synthesized answer from search results.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response(
            "Question", tools=[{}], tool_manager=mock_tool_manager
        )
        assert result == "Synthesized answer from search results."

    def test_no_tool_execution_when_tool_manager_is_none(
        self, mock_anthropic_client, make_tool_use_response
    ):
        """
        If stop_reason='tool_use' but tool_manager=None, generate_response must
        fall back gracefully rather than crash.

        This verifies the guard: `if response.stop_reason == 'tool_use' and tool_manager`.

        Failure means: AttributeError when accessing None.execute_tool().
        """
        first = make_tool_use_response("search_course_content", {"query": "test"})
        mock_anthropic_client.messages.create.return_value = first

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        try:
            result = gen.generate_response("Question", tools=[{}], tool_manager=None)
            assert isinstance(result, str)
        except AttributeError as e:
            pytest.fail(
                f"generate_response() crashed with AttributeError when tool_manager=None: {e}"
            )

    def test_final_call_after_two_rounds_excludes_tools(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After two tool rounds (at MAX_TOOL_ROUNDS limit), the final API call must
        NOT include 'tools', forcing Claude into synthesis mode.

        Failure means: Claude could attempt a third tool call, violating the round limit.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response("get_course_outline", {"course_name": "Python"})
        second = make_tool_use_response("search_course_content", {"query": "OOP"})
        third = make_text_response("Final synthesized answer.")
        mock_anthropic_client.messages.create.side_effect = [first, second, third]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    def test_second_call_includes_assistant_message_before_tool_result(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        The second API call's message list must include an assistant message
        (the initial tool_use response) before the tool_result user message.
        This gives Claude context about what it previously requested.

        Failure means: Claude sees the tool result without knowing it asked for it,
        causing confused or incorrect synthesis.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response("search_course_content", {"query": "test"})
        second = make_text_response("Final answer")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question", tools=[{}], tool_manager=mock_tool_manager)

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        roles = [m["role"] for m in messages]
        assert "assistant" in roles, "Second API call must include an assistant message"

        # Assistant message must come before tool_result user message
        assistant_idx = next(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        tool_result_idx = next(
            (
                i
                for i, m in enumerate(messages)
                if m["role"] == "user"
                and isinstance(m.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in m["content"]
                )
            ),
            None,
        )
        if tool_result_idx is not None:
            assert (
                assistant_idx < tool_result_idx
            ), "Assistant message must come before tool_result message in conversation"


class TestAIGeneratorSequentialToolUse:
    """
    Tests for the sequential tool-calling loop (up to MAX_TOOL_ROUNDS=2 rounds).

    Each round is one tool-use → execute → API call cycle. The loop terminates when:
    (a) MAX_TOOL_ROUNDS rounds completed, (b) Claude returns end_turn, or
    (c) execute_tool raises an exception.
    """

    def test_two_tool_rounds_makes_three_api_calls(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        When Claude uses a tool in both round 1 and round 2, exactly 3 API calls
        must be made and the final text is returned.

        Failure means: the loop exits after 1 round and the second search is skipped,
        breaking multi-step queries like "find a course on the same topic as lesson X".
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        first = make_tool_use_response("get_course_outline", {"course_name": "Python"})
        second = make_tool_use_response("search_course_content", {"query": "OOP"})
        third = make_text_response("Complete answer after two searches.")
        mock_anthropic_client.messages.create.side_effect = [first, second, third]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response(
            "Find a course on the same topic as lesson 1 of Python",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Complete answer after two searches."

    def test_two_tool_rounds_final_call_excludes_tools(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After 2 rounds (MAX_TOOL_ROUNDS reached), the 3rd API call must not include
        'tools', forcing Claude to synthesize without making further searches.

        Failure means: Claude could attempt a 3rd tool call, violating the round limit.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response("get_course_outline", {"course_name": "Python"})
        second = make_tool_use_response("search_course_content", {"query": "OOP"})
        third = make_text_response("Final answer.")
        mock_anthropic_client.messages.create.side_effect = [first, second, third]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question", tools=[{}], tool_manager=mock_tool_manager)

        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    def test_claude_stops_after_one_round_two_calls_total(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        When Claude uses one tool then returns end_turn (no second tool call needed),
        only 2 API calls are made and the text from the 2nd call is returned.

        Failure means: the loop runs an extra empty round, making 3 calls for a
        single-tool query.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        first = make_tool_use_response(
            "search_course_content", {"query": "python basics"}
        )
        second = make_text_response("Python is a high-level language.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response(
            "What is Python?", tools=[{}], tool_manager=mock_tool_manager
        )

        assert mock_anthropic_client.messages.create.call_count == 2
        assert result == "Python is a high-level language."

    def test_second_call_includes_tools_so_claude_can_do_round_two(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After round 1, the second API call must include 'tools' so Claude can
        optionally invoke round 2. Claude may choose not to — that is fine.

        Failure means: Claude is not offered tools for round 2, preventing it from
        doing the second search even when it would help.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        tool_defs = [{"name": "search_course_content"}]
        first = make_tool_use_response("search_course_content", {"query": "test"})
        second = make_text_response("Answer.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Question", tools=tool_defs, tool_manager=mock_tool_manager
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

    def test_tool_exception_terminates_loop_and_excludes_tools_from_final_call(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        When execute_tool raises an exception, the loop must terminate immediately
        and the next API call must not include 'tools', allowing Claude to respond
        gracefully with the error context.

        Failure means: crash propagates to the caller or Claude is offered tools
        again after an execution failure.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = RuntimeError("Database timeout")

        first = make_tool_use_response("search_course_content", {"query": "test"})
        second = make_text_response(
            "I encountered an error retrieving that information."
        )
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response(
            "Question", tools=[{}], tool_manager=mock_tool_manager
        )

        assert mock_anthropic_client.messages.create.call_count == 2
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert result == "I encountered an error retrieving that information."

    def test_message_history_accumulates_across_two_rounds(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        After 2 rounds, the final API call's message list must contain all 5 turns:
        user query → assistant (round 1) → tool result (round 1) →
        assistant (round 2) → tool result (round 2).

        Failure means: context from an earlier round is lost, causing Claude to
        synthesize without full search history.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        first = make_tool_use_response(
            "get_course_outline", {"course_name": "X"}, tool_id="t1"
        )
        second = make_tool_use_response(
            "search_course_content", {"query": "Y"}, tool_id="t2"
        )
        third = make_text_response("Full answer.")
        mock_anthropic_client.messages.create.side_effect = [first, second, third]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response(
            "Multi-step question", tools=[{}], tool_manager=mock_tool_manager
        )

        third_call_messages = mock_anthropic_client.messages.create.call_args_list[2][
            1
        ]["messages"]
        assert len(third_call_messages) == 5
        assert third_call_messages[0]["role"] == "user"
        assert third_call_messages[1]["role"] == "assistant"
        assert third_call_messages[2]["role"] == "user"
        assert any(
            isinstance(b, dict) and b.get("tool_use_id") == "t1"
            for b in third_call_messages[2]["content"]
        )
        assert third_call_messages[3]["role"] == "assistant"
        assert third_call_messages[4]["role"] == "user"
        assert any(
            isinstance(b, dict) and b.get("tool_use_id") == "t2"
            for b in third_call_messages[4]["content"]
        )

    def test_error_string_result_does_not_terminate_loop(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        When execute_tool returns an error string (not an exception), the loop must
        continue normally — tools are still offered in the next call.

        Only Python exceptions trigger early loop termination, not "no results" strings.
        Failure means: valid "no results" responses abort the loop prematurely,
        preventing Claude from attempting a follow-up search.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "No course found matching 'xyz'."

        tool_defs = [{"name": "search_course_content"}]
        first = make_tool_use_response("search_course_content", {"query": "xyz"})
        second = make_text_response("No course was found for that query.")
        mock_anthropic_client.messages.create.side_effect = [first, second]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        result = gen.generate_response(
            "Question", tools=tool_defs, tool_manager=mock_tool_manager
        )

        assert mock_anthropic_client.messages.create.call_count == 2
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert result == "No course was found for that query."

    def test_both_tool_calls_dispatched_with_correct_names_and_inputs(
        self, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        """
        In a two-round sequence, execute_tool must be called with the exact tool name
        and input kwargs from each round's ToolUseBlock.

        Failure means: wrong tool is dispatched or inputs are dropped, causing incorrect
        search results or TypeError in the tool executor.
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Results"

        first = make_tool_use_response(
            "get_course_outline", {"course_name": "Python OOP"}, tool_id="t1"
        )
        second = make_tool_use_response(
            "search_course_content",
            {"query": "OOP concepts", "lesson_number": 3},
            tool_id="t2",
        )
        third = make_text_response("Answer.")
        mock_anthropic_client.messages.create.side_effect = [first, second, third]

        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="sk-test", model="claude-sonnet-4-6")
        gen.generate_response("Question", tools=[{}], tool_manager=mock_tool_manager)

        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0][0] == "get_course_outline"
        assert calls[0][1] == {"course_name": "Python OOP"}
        assert calls[1][0][0] == "search_course_content"
        assert calls[1][1] == {"query": "OOP concepts", "lesson_number": 3}
