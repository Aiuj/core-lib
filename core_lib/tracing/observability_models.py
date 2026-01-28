"""Observability metadata models and documentation.

This module defines the structure and documentation for observability metadata
passed via the `from_` parameter in API requests.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

# Type alias for from_ parameter
FromMetadata = Optional[Union[str, Dict[str, Any]]]

# Centralized description for 'from' parameter used across applications for traceability
FROM_FIELD_DESCRIPTION = """Source information for observability and tracing (dict or JSON string).

Accepts either a JSON string or a dictionary with the following structure:
- session_id: Session identifier
- app_name: Application name (e.g., "Saas Admin Platform")
- app_version: Application version (e.g., "0.1.0")
- model_name: AI model name (e.g., "gemini-2.5-flash")
- user_name: User's display name
- user_id: User's unique identifier (UUID)
- company_name: Organization name
- company_id: Organization unique identifier (UUID)

Example:
```json
{
    "session_id": "12",
    "app_name": "Saas Admin Platform",
    "app_version": "0.1.0",
    "model_name": "gemini-2.5-flash",
    "user_name": "Sarah Mitchell",
    "user_id": "10000000-0000-0000-0000-000000000101",
    "company_name": "TechVision Solutions",
    "company_id": "10000000-0000-0000-0000-000000000001"
}
```

All fields are optional."""


# Centralized description for 'intelligence_level' parameter used across applications
INTELLIGENCE_LEVEL_DESCRIPTION = """Intelligence level (0-10) controlling search depth, quality, and cost.

Higher levels use more sophisticated retrieval and analysis:

| Level | Name | Search Methods | QA Cache | Query Expansion | Context | Reranking | Search Quality |
|---|---|---|---|---|---|---|---|
| 0 | Minimal | Lexical only | No | None | 1 chunk | No | No |
| 1 | Mini | Lexical only | Direct | None | 2 chunks | No | No |
| 2 | Basic (Low) | Lexical + semantic QA | Direct + context | None | 3 chunks | No | No |
| 3 | Basic | Lexical + semantic QA | Direct + context | Keywords | 4 chunks | No | No |
| 4 | Standard (Low) | Hybrid + chunks | Direct + context | Keywords | 5 chunks | No | No |
| 5 | Standard (Default) | Hybrid + chunks | Direct + context | Hybrid | 8 chunks | No | No |
| 6 | High (Low) | Hybrid + chunks | Direct + context | Hybrid | 10 chunks | No | No |
| 7 | High | Hybrid + chunks | Direct + context | LLM | 12 chunks | Conditional | Yes |
| 8 | Max (Low) | Hybrid + chunks | Context only* | LLM | 15 chunks | Always | Yes |
| 9 | Max | Hybrid + chunks | Context only* | LLM | 20 chunks | Always | Yes |
| 10 | Ultimate | Hybrid + chunks | Context only* | LLM | 30 chunks | Always | Yes |

*Levels 8-10 don't use QA cache for direct answers, but QA pairs are still used as context.

Default: 5 (Standard). Higher levels = better quality but slower and more expensive.
Lower levels = faster and cheaper but may miss nuanced information."""


class FromMetadataSchema(BaseModel):
    """Schema defining the expected structure of the `from_` metadata.
    
    The `from_` parameter provides contextual information for observability,
    tracing, and audit trails. It can be passed as either:
    - A JSON string (automatically parsed by FastAPI)
    - A dictionary object
    
    All fields are optional, but providing complete information enables
    better tracing and analytics.
    
    Example JSON string format:
    ```json
    {
        "session_id": "12",
        "app_name": "Saas Admin Platform",
        "app_version": "0.1.0",
        "model_name": "gemini-2.5-flash",
        "user_name": "Sarah Mitchell",
        "user_id": "10000000-0000-0000-0000-000000000101",
        "company_name": "TechVision Solutions",
        "company_id": "10000000-0000-0000-0000-000000000001"
    }
    ```
    
    Example Python dict format:
    ```python
    {
        "session_id": "12",
        "app_name": "Saas Admin Platform",
        "app_version": "0.1.0",
        "model_name": "gemini-2.5-flash",
        "user_name": "Sarah Mitchell",
        "user_id": "10000000-0000-0000-0000-000000000101",
        "company_name": "TechVision Solutions",
        "company_id": "10000000-0000-0000-0000-000000000001"
    }
    ```
    
    Fields:
        session_id: Unique identifier for the current session/conversation
        app_name: Name of the calling application
        app_version: Version of the calling application
        model_name: AI model being used (e.g., "gemini-2.5-flash", "gpt-4")
        user_name: Human-readable name of the user making the request
        user_id: Unique identifier for the user (UUID format recommended)
        company_name: Human-readable name of the organization
        company_id: Unique identifier for the organization (UUID format)
    """
    
    # Session context
    session_id: Optional[str] = Field(
        None,
        description="Unique identifier for the current session/conversation"
    )
    
    # Application context
    app_name: Optional[str] = Field(
        None,
        description="Name of the calling application"
    )
    app_version: Optional[str] = Field(
        None,
        description="Version of the calling application"
    )
    
    # Model context
    model_name: Optional[str] = Field(
        None,
        description="AI model being used (e.g., 'gemini-2.5-flash', 'gpt-4')"
    )
    
    # User context
    user_name: Optional[str] = Field(
        None,
        description="Human-readable name of the user making the request"
    )
    user_id: Optional[str] = Field(
        None,
        description="Unique identifier for the user (UUID format recommended)"
    )
    
    # Organization context
    company_name: Optional[str] = Field(
        None,
        description="Human-readable name of the organization"
    )
    company_id: Optional[str] = Field(
        None,
        description="Unique identifier for the organization (UUID format)"
    )


# Re-export constants for backward compatibility
__all__ = ["FromMetadata", "FromMetadataSchema", "FROM_FIELD_DESCRIPTION", "INTELLIGENCE_LEVEL_DESCRIPTION"]
