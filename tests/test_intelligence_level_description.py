"""Tests for the shared intelligence-level OpenAPI documentation."""

from core_lib.tracing.observability_models import INTELLIGENCE_LEVEL_DESCRIPTION


def test_description_exposes_default_context_model_and_credit_boundaries():
    description = INTELLIGENCE_LEVEL_DESCRIPTION

    assert "Default: 4 (Standard Low)" in description
    assert "3.5k retrieved tokens | IQ 4 | 1x" in description
    assert "7k retrieved tokens | IQ 5 | 2x" in description
    assert "7k retrieved tokens | IQ 7 | 5x" in description
