"""
Core exceptions for core-lib.

This module defines common exceptions that can be used across all projects
using core-lib, particularly for job processing and API interactions.
"""


class CoreLibError(Exception):
    """Base exception for all core-lib errors."""
    pass


class ConfigurationError(CoreLibError):
    """Exception raised when there's a configuration problem.
    
    This error indicates a setup/configuration issue that won't be resolved by retrying.
    Examples:
    - Missing Ollama models
    - Invalid API keys
    - Missing required environment variables
    - Unreachable services (DNS/connection failures)
    
    Jobs with this error should not be retried automatically as they require
    manual intervention to fix the underlying configuration.
    
    Attributes:
        message: Human-readable error message
        error_type: Category of error (e.g., "MISSING_MODEL", "INVALID_API_KEY")
        details: Optional dict with additional context
    """
    
    def __init__(
        self,
        message: str,
        error_type: str = None,
        details: dict = None
    ):
        super().__init__(message)
        self.error_type = error_type or "CONFIGURATION_ERROR"
        self.details = details or {}


class APIError(CoreLibError):
    """Base exception for API-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        status_code: int = None,
        details: dict = None
    ):
        super().__init__(message)
        self.error_code = error_code or "API_ERROR"
        self.status_code = status_code
        self.details = details or {}


class TimeoutError(APIError):
    """Exception raised when API requests timeout.
    
    This is a transient error that may be resolved by retrying.
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float = None,
        details: dict = None
    ):
        super().__init__(message, error_code="TIMEOUT_ERROR")
        self.timeout_seconds = timeout_seconds
        self.details = details or {}
