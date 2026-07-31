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

from freezegun import configure as configure_freezegun


# Langfuse exposes some Pydantic types through lazy module attributes.  Letting
# freezegun inspect those modules while entering a frozen-time context can
# trigger schema generation with freezegun's temporary datetime type.
configure_freezegun(extend_ignore_list=["langfuse"])


@pytest.fixture(autouse=True)
def clear_embedding_client_cache():
    """Keep environment-mutating tests independent of factory caching.

    The production factory deliberately caches clients by explicit arguments.
    Tests that change embedding environment variables must not inherit a client
    created by an earlier test with the same argument-based cache key.
    """
    from core_lib.embeddings import factory

    factory._embedding_client_cache.clear()
    yield
    factory._embedding_client_cache.clear()


@pytest.fixture(autouse=True)
def isolate_provider_health_from_external_cache(monkeypatch):
    """Keep unit tests independent from persisted Valkey/Redis health state."""
    from core_lib.llm.provider_health import ProviderHealthTracker

    monkeypatch.setattr(
        ProviderHealthTracker,
        "_get_cache",
        lambda tracker: tracker._cache_client,
    )


@pytest.fixture(autouse=True)
def isolate_embedding_results_from_external_cache(monkeypatch):
    """Prevent persisted Valkey/Redis embeddings from masking provider tests."""
    from core_lib.embeddings import base

    monkeypatch.setattr(base, "cache_get", lambda _key: None)
    monkeypatch.setattr(base, "cache_set", lambda *_args, **_kwargs: None)

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
