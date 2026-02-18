"""core_lib.utils - Utility functions and classes.

This package provides utility functionality for the core_lib library including:
- LanguageUtils: Language detection, text manipulation, and NLP utilities
- File utilities: Temporary file creation and cleanup helpers
- HealthChecker: Service health check framework for monitoring

Convenience imports for the `core_lib.utils` package.

Exports a small, stable surface so consumers can do::

    from core_lib.utils import LanguageUtils
    from core_lib.utils import create_tempfile, remove_tempfile
    from core_lib.utils import HealthChecker, HealthCheckResult, HealthStatus

The package intentionally re-exports only the primary utilities implemented
in the package to keep the public API small and predictable.
"""

from .language_utils import LanguageUtils
from .file_utils import create_tempfile, remove_tempfile
from .health_check import HealthChecker, HealthCheckResult, HealthStatus, create_lazy_check
from .datetime_utils import utc_now, parse_iso_datetime, to_iso_string

__all__ = [
    "AppSettings",
    "LanguageUtils",
    "create_tempfile",
    "remove_tempfile",
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "create_lazy_check",
    "utc_now",
    "parse_iso_datetime",
    "to_iso_string",
]
