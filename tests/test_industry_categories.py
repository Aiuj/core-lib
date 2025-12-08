import pytest
from core_lib.config.industry_categories import (
    INDUSTRY_CATEGORIES,
    INDUSTRY_CATEGORIES_BY_KEY,
    INDUSTRY_CATEGORY_CHOICES
)

def test_industry_categories_structure():
    """Test that industry categories have the correct structure."""
    required_fields = {"key", "label", "description", "icon", "area"}
    
    assert len(INDUSTRY_CATEGORIES) > 0
    
    seen_keys = set()
    for category in INDUSTRY_CATEGORIES:
        # Check fields
        assert required_fields.issubset(category.keys())
        
        # Check types
        assert isinstance(category["key"], str)
        assert isinstance(category["label"], str)
        assert isinstance(category["description"], str)
        assert isinstance(category["icon"], str)
        assert isinstance(category["area"], str)
        
        # Check uniqueness
        key = category["key"]
        assert key not in seen_keys
        seen_keys.add(key)

def test_industry_categories_helpers():
    """Test that helper constants are correctly populated."""
    assert len(INDUSTRY_CATEGORIES_BY_KEY) == len(INDUSTRY_CATEGORIES)
    assert len(INDUSTRY_CATEGORY_CHOICES) == len(INDUSTRY_CATEGORIES)
    
    for key, category in INDUSTRY_CATEGORIES_BY_KEY.items():
        assert category["key"] == key
        
    for key, label in INDUSTRY_CATEGORY_CHOICES:
        assert key in INDUSTRY_CATEGORIES_BY_KEY
        assert INDUSTRY_CATEGORIES_BY_KEY[key]["label"] == label
