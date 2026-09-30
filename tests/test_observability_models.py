import pytest
from pydantic import ValidationError

from core_lib.tracing.observability_models import FROM_FIELD_DESCRIPTION, FromMetadataSchema


def test_usage_origin_and_agent_are_preserved():
    model = FromMetadataSchema(usage_origin="mcp", usage_agent="rfx-agent")
    dumped = model.model_dump(exclude_none=True)
    assert dumped == {"usage_origin": "mcp", "usage_agent": "rfx-agent"}


def test_usage_origin_rejects_unknown_value():
    with pytest.raises(ValidationError):
        FromMetadataSchema(usage_origin="cli")


def test_description_documents_new_fields():
    assert "usage_origin" in FROM_FIELD_DESCRIPTION
    assert "usage_agent" in FROM_FIELD_DESCRIPTION
