"""Confidentiality level constants and utilities for access control.

This module defines a 5-tier confidentiality level system used across all
applications for data access control. Higher numeric values indicate
higher confidentiality requirements.

Confidentiality Levels:
- 10 (public): Publicly shareable information
- 30 (prospect): Information suitable for prospects/potential customers (DEFAULT)
- 50 (customer): Information for existing customers
- 70 (internal): Internal use only, not for external sharing
- 90 (confidential): Highly sensitive, restricted access

Access Control Semantics:
Users with a given confidentiality clearance level can access data at or below
their level. For example, a user with level 50 (customer) can access public (10),
prospect (30), and customer (50) data, but not internal (70) or confidential (90).

When multiple results have equal relevance, higher confidentiality data is
prioritized as it typically contains more detailed or sensitive information.
"""

from typing import Dict, Optional

# Confidentiality level definitions (string label -> numeric value)
CONFIDENTIALITY_LEVELS: Dict[str, int] = {
    "public": 10,
    "prospect": 30,
    "customer": 50,
    "internal": 70,
    "confidential": 90,
}

# Reverse mapping (numeric value -> string label)
CONFIDENTIALITY_LEVEL_NAMES: Dict[int, str] = {
    10: "public",
    30: "prospect",
    50: "customer",
    70: "internal",
    90: "confidential",
}

# Default confidentiality level (prospect - suitable for RFx management)
DEFAULT_CONFIDENTIALITY_LEVEL: int = 30

# Valid range for confidentiality levels
MIN_CONFIDENTIALITY_LEVEL: int = 0
MAX_CONFIDENTIALITY_LEVEL: int = 99

# OpenAPI/MCP documentation string
CONFIDENTIALITY_LEVEL_DESCRIPTION = (
    "Data access clearance level (0-99). Only retrieves data at or below this level. "
    "Higher confidentiality results are prioritized when relevance is equal. "
    "Levels: 10=public, 30=prospect (default), 50=customer, 70=internal, 90=confidential."
)


def validate_confidentiality_level(level: int) -> int:
    """Validate that a confidentiality level is within the valid range.
    
    Args:
        level: Confidentiality level to validate (0-99)
        
    Returns:
        The validated confidentiality level (same as input if valid)
        
    Raises:
        ValueError: If level is outside the valid range (0-99)
        
    Examples:
        >>> validate_confidentiality_level(50)
        50
        >>> validate_confidentiality_level(10)
        10
        >>> validate_confidentiality_level(100)
        Traceback (most recent call last):
        ...
        ValueError: Confidentiality level must be between 0 and 99, got 100
    """
    if not isinstance(level, int):
        raise ValueError(f"Confidentiality level must be an integer, got {type(level).__name__}")
    
    if level < MIN_CONFIDENTIALITY_LEVEL or level > MAX_CONFIDENTIALITY_LEVEL:
        raise ValueError(
            f"Confidentiality level must be between {MIN_CONFIDENTIALITY_LEVEL} "
            f"and {MAX_CONFIDENTIALITY_LEVEL}, got {level}"
        )
    
    return level


def get_confidentiality_level_name(level: int) -> Optional[str]:
    """Get the human-readable name for a confidentiality level.
    
    Args:
        level: Numeric confidentiality level
        
    Returns:
        String label if level is a standard tier, None otherwise
        
    Examples:
        >>> get_confidentiality_level_name(50)
        'customer'
        >>> get_confidentiality_level_name(90)
        'confidential'
        >>> get_confidentiality_level_name(55)
        None
    """
    return CONFIDENTIALITY_LEVEL_NAMES.get(level)


def get_confidentiality_level_value(name: str) -> Optional[int]:
    """Get the numeric value for a confidentiality level name.
    
    Args:
        name: String label (case-insensitive)
        
    Returns:
        Numeric level if name is valid, None otherwise
        
    Examples:
        >>> get_confidentiality_level_value("customer")
        50
        >>> get_confidentiality_level_value("CONFIDENTIAL")
        90
        >>> get_confidentiality_level_value("unknown")
        None
    """
    return CONFIDENTIALITY_LEVELS.get(name.lower())


__all__ = [
    "CONFIDENTIALITY_LEVELS",
    "CONFIDENTIALITY_LEVEL_NAMES",
    "DEFAULT_CONFIDENTIALITY_LEVEL",
    "MIN_CONFIDENTIALITY_LEVEL",
    "MAX_CONFIDENTIALITY_LEVEL",
    "CONFIDENTIALITY_LEVEL_DESCRIPTION",
    "validate_confidentiality_level",
    "get_confidentiality_level_name",
    "get_confidentiality_level_value",
]
