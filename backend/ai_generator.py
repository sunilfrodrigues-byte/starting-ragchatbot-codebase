import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use search_course_content **only** for questions about specific course content or detailed educational materials
- Use get_course_outline for questions about course structure, lesson lists, or course overview
- **Up to two sequential tool calls per query** — use a second call only when the first result is insufficient to answer the question
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Response Format:
- When returning a course outline, always include: course title, course link, and the number and title of every lesson
- Present lessons as a numbered list

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, messages, system_content, tools, tool_manager)

        return response.content[0].text

    def _run_tool_loop(self, current_response, messages: list, system_content: str,
                       tools, tool_manager) -> str:
        """
        Execute up to MAX_TOOL_ROUNDS of tool calling, accumulating conversation
        context between rounds. Terminates when Claude stops requesting tools,
        the round limit is reached, or a tool raises an exception.
        """
        calls_remaining = self.MAX_TOOL_ROUNDS

        while current_response.stop_reason == "tool_use" and calls_remaining > 0:
            messages.append({"role": "assistant", "content": current_response.content})

            tool_results = []
            execution_error = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as exc:
                        result = f"Tool execution error: {exc}"
                        execution_error = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            calls_remaining -= 1

            next_params = {**self.base_params, "messages": messages, "system": system_content}
            if tools and calls_remaining > 0 and not execution_error:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**next_params)

            if execution_error:
                break

        return current_response.content[0].text