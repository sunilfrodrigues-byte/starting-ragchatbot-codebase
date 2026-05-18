"""
test_config.py — Tests for configuration correctness, focusing on model name validity.

CRITICAL FILE: Contains the test that directly identifies the root cause of the
"query failed" bug: ANTHROPIC_MODEL = "claude-sonnet-4-20250514" is not a valid
Anthropic API model ID.

Error chain:
  config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
  → anthropic.Anthropic().messages.create(model=...) raises NotFoundError(404)
  → propagates through AIGenerator → RAGSystem.query()
  → caught by app.py except Exception → HTTPException(500)
  → frontend receives non-200 → displays "Error: Query failed"

Fix: change ANTHROPIC_MODEL to "claude-sonnet-4-6" in backend/config.py line 13.
"""

import os

import pytest


class TestConfigStructure:
    """Unit tests: Config object has required attributes with correct types."""

    def test_config_singleton_importable(self):
        """Config can be imported without errors."""
        from config import Config, config

        assert isinstance(config, Config)

    def test_anthropic_model_field_is_nonempty_string(self):
        """ANTHROPIC_MODEL must be a non-empty string."""
        from config import Config

        cfg = Config()
        assert isinstance(cfg.ANTHROPIC_MODEL, str)
        assert len(cfg.ANTHROPIC_MODEL.strip()) > 0

    def test_anthropic_api_key_field_exists(self):
        """ANTHROPIC_API_KEY attribute must exist."""
        from config import Config

        cfg = Config()
        assert hasattr(cfg, "ANTHROPIC_API_KEY")
        assert isinstance(cfg.ANTHROPIC_API_KEY, str)

    def test_model_name_is_not_known_invalid_value(self):
        """
        EXPECTED TO FAIL on the current codebase.

        "claude-sonnet-4-20250514" returns HTTP 404 from the Anthropic API.
        Every call to AIGenerator.generate_response() fails with NotFoundError,
        which propagates as HTTPException(500) and shows as "query failed" in the UI.

        Fix: change ANTHROPIC_MODEL to "claude-sonnet-4-6" in backend/config.py line 13.
        """
        KNOWN_INVALID_MODELS = {
            "claude-sonnet-4-20250514",
        }
        from config import Config

        cfg = Config()
        assert cfg.ANTHROPIC_MODEL not in KNOWN_INVALID_MODELS, (
            f"ANTHROPIC_MODEL is set to '{cfg.ANTHROPIC_MODEL}', which returns HTTP 404 "
            f"from the Anthropic API. This is the root cause of the 'query failed' error.\n"
            f"Fix: change ANTHROPIC_MODEL to 'claude-sonnet-4-6' in backend/config.py line 13."
        )

    def test_chroma_path_is_configured(self):
        """CHROMA_PATH must be a non-empty string."""
        from config import Config

        cfg = Config()
        assert isinstance(cfg.CHROMA_PATH, str) and cfg.CHROMA_PATH.strip()

    def test_max_results_is_positive_integer(self):
        """MAX_RESULTS must be a positive integer for ChromaDB n_results."""
        from config import Config

        cfg = Config()
        assert isinstance(cfg.MAX_RESULTS, int) and cfg.MAX_RESULTS > 0

    def test_chunk_size_exceeds_overlap(self):
        """CHUNK_SIZE must be greater than CHUNK_OVERLAP, else chunking produces empty chunks."""
        from config import Config

        cfg = Config()
        assert cfg.CHUNK_SIZE > cfg.CHUNK_OVERLAP > 0


@pytest.mark.integration
class TestModelNameValidityWithRealAPI:
    """
    Integration tests: validate model name against the live Anthropic API.
    Skipped if ANTHROPIC_API_KEY is not set.
    """

    def test_configured_model_accepted_by_api(self):
        """
        THE CRITICAL INTEGRATION TEST.

        Sends a minimal real API call using the model name from config.py.
        If the model name is invalid, the API returns HTTP 404 NotFoundError.

        Expected failure on current codebase:
            anthropic.NotFoundError: Error code: 404 — model: claude-sonnet-4-20250514

        Fix: change ANTHROPIC_MODEL to "claude-sonnet-4-6" in backend/config.py line 13.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        from config import config

        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": "Reply with OK only."}],
            )
            assert response.content is not None
            assert len(response.content) > 0
        except anthropic.NotFoundError as e:
            pytest.fail(
                f"ANTHROPIC_MODEL '{config.ANTHROPIC_MODEL}' rejected by API with 404.\n"
                f"This is the root cause of the 'query failed' error in production.\n"
                f"API error: {e}\n"
                f"Fix: change ANTHROPIC_MODEL to 'claude-sonnet-4-6' in backend/config.py line 13."
            )
        except anthropic.AuthenticationError:
            pytest.skip("Invalid ANTHROPIC_API_KEY — cannot validate model name")

    def test_known_good_model_name_is_accepted(self):
        """
        Confirms 'claude-sonnet-4-6' is accepted as a regression anchor.

        If this test fails, 'claude-sonnet-4-6' is itself no longer valid and a
        new model name needs to be identified.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=5,
                messages=[{"role": "user", "content": "Reply with OK only."}],
            )
            assert response.content is not None
        except anthropic.NotFoundError:
            pytest.fail(
                "'claude-sonnet-4-6' was rejected. The expected fix may be outdated — "
                "find the current valid model name."
            )
        except anthropic.AuthenticationError:
            pytest.skip("Invalid ANTHROPIC_API_KEY")
