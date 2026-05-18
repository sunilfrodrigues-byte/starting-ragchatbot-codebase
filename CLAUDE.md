# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the server** (from repo root):
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```
Or use the convenience script (Git Bash / Linux / Mac):
```bash
./run.sh
```

**Environment setup** — create a `.env` file in the repo root:
```
ANTHROPIC_API_KEY=your_key_here
```

The app serves at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials. FastAPI backend + vanilla JS frontend + ChromaDB vector store + Claude claude-sonnet-4-6.

### Query flow
1. Frontend (`frontend/script.js`) POSTs `{query, session_id}` to `POST /api/query`
2. `app.py` delegates to `RAGSystem.query()` in `rag_system.py`
3. `RAGSystem` fetches conversation history from `SessionManager`, then calls `AIGenerator.generate_response()`
4. `AIGenerator` makes a first Claude API call with the `search_course_content` tool available
5. If Claude invokes the tool, `ToolManager` dispatches to `CourseSearchTool`, which queries ChromaDB via `VectorStore.search()`
6. Tool results are sent back to Claude in a second API call; the synthesized answer is returned up the chain
7. Sources and session history are updated; `{answer, sources, session_id}` is returned to the frontend

### Key design decisions

**Two ChromaDB collections** (`vector_store.py`):
- `course_catalog` — one entry per course (title, instructor, link, lessons JSON). Used for fuzzy course name resolution when the tool is called with a `course_name`.
- `course_content` — all text chunks with `course_title` and `lesson_number` metadata. This is what gets semantically searched.

**Tool-based retrieval** — Claude decides whether to search at all. General knowledge questions are answered without hitting ChromaDB. Course-specific questions trigger the `search_course_content` tool (max one search per query).

**Session history** is stored in-memory in `SessionManager` (not persisted across restarts). Only the last 2 exchanges are injected into the system prompt as a plain text string, not as proper message history in the API call.

**Embedding model** — `all-MiniLM-L6-v2` (via `sentence-transformers`) runs locally. ChromaDB uses it for both indexing and query embedding.

### Document format
Course files in `docs/` must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<content...>

Lesson 2: <title>
...
```
`DocumentProcessor` parses these, chunks content at ~800 chars with 100-char sentence-based overlap, and tags each chunk with `course_title` and `lesson_number`. On startup, `app.py` loads all `.txt/.pdf/.docx` files from `../docs`, skipping courses already in the vector store.

### Config
All tunable parameters are in `backend/config.py`: model name, chunk size/overlap, max search results, max history turns, and ChromaDB path.
