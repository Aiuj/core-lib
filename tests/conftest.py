# Test configuration and fallbacks for core-lib
#
# Provides a lightweight fallback for async tests when pytest-asyncio
# plugin is not installed. This ensures repository tests still run in
# minimal environments while preserving compatibility with pytest-asyncio
# when it is available.

import inspect
import asyncio
import logging
from typing import Any
import pytest

logger = logging.getLogger("core_lib.tests")


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "asyncio: mark test as asynchronous (fallback provided if pytest-asyncio missing)"
    )


def pytest_pyfunc_call(pyfuncitem):  # type: ignore
    """Fallback async test runner.

    If pytest-asyncio is installed we do nothing and allow the plugin to manage execution.
    Otherwise we detect coroutine functions marked with @pytest.mark.asyncio and run them
    using asyncio.run().
    """
    # If pytest-asyncio plugin present, defer to it
    if any(name.startswith("pytest_asyncio") for name, _ in pyfuncitem.config.pluginmanager.list_name_plugin()):
        return None  # let plugin handle

    if "asyncio" in pyfuncitem.keywords and inspect.iscoroutinefunction(pyfuncitem.obj):
        # Get the function signature to determine which arguments it actually accepts
        sig = inspect.signature(pyfuncitem.obj)
        valid_params = set(sig.parameters.keys())
        
        # Only pass arguments that the function actually accepts
        filtered_kwargs = {
            k: v for k, v in pyfuncitem.funcargs.items() 
            if k in valid_params
        }
        asyncio.run(pyfuncitem.obj(**filtered_kwargs))
        return True  # indicate we handled invocation
    return None
