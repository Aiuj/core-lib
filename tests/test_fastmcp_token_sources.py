"""Tests for restricting where the FastMCP JWT middleware accepts a token.

The default accepts a token from the Authorization header, the MCP_JWT_TOKEN
environment variable, request metadata or a query parameter. That suits stdio
and private-network servers. A public, multi-tenant endpoint must accept the
header only: a process-wide env token would authenticate anonymous callers,
and a query-parameter token leaks into access and proxy logs.

These tests pin both behaviours, so neither the permissive default nor the
strict option can drift without someone noticing.
"""

import asyncio
import os
from unittest.mock import patch

import pytest

from core_lib.api_utils.fastmcp_auth import (
    STRICT_TOKEN_SOURCES,
    TOKEN_SOURCES,
    MCPAuthError,
    create_jwt_auth_middleware,
)
from core_lib.api_utils.jwt_auth import JWTAuthSettings, create_jwt_token

SECRET = "test-secret-for-token-source-checks"


def _settings():
    return JWTAuthSettings(jwt_secret=SECRET, require_auth=True)


def _token():
    return create_jwt_token(
        {"sub": "client-1", "company_id": "company-a", "scopes": ["kb.read"]},
        _settings(),
        token_type="access",
    )


async def _passthrough(context):
    return {"ok": True, "company_id": context.get("company_id")}


def _run(middleware, context):
    return asyncio.run(middleware(context, _passthrough))


class TestTokenSourceConstants:
    def test_strict_is_header_only(self):
        assert STRICT_TOKEN_SOURCES == ("header",)

    def test_strict_is_a_subset_of_all_sources(self):
        assert set(STRICT_TOKEN_SOURCES) <= set(TOKEN_SOURCES)


class TestDefaultBehaviourIsUnchanged:
    """The default must stay permissive; fleet servers rely on it."""

    def test_header_token_is_accepted(self):
        middleware = create_jwt_auth_middleware(_settings())
        context = {"headers": {"Authorization": f"Bearer {_token()}"}}

        assert _run(middleware, context)["company_id"] == "company-a"

    def test_env_token_is_accepted(self):
        middleware = create_jwt_auth_middleware(_settings())
        with patch.dict(os.environ, {"MCP_JWT_TOKEN": _token()}):
            assert _run(middleware, {})["ok"] is True

    def test_metadata_token_is_accepted(self):
        middleware = create_jwt_auth_middleware(_settings())
        context = {"metadata": {"token": _token()}}

        with patch.dict(os.environ, {}, clear=True):
            assert _run(middleware, context)["ok"] is True

    def test_query_token_is_accepted(self):
        middleware = create_jwt_auth_middleware(_settings())
        context = {"query_params": {"token": _token()}}

        with patch.dict(os.environ, {}, clear=True):
            assert _run(middleware, context)["ok"] is True


class TestStrictModeRejectsEverythingButTheHeader:
    def test_header_token_still_works(self):
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=STRICT_TOKEN_SOURCES
        )
        context = {"headers": {"Authorization": f"Bearer {_token()}"}}

        assert _run(middleware, context)["company_id"] == "company-a"

    def test_env_token_is_refused(self):
        """A process-wide token must not authenticate an anonymous caller."""
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=STRICT_TOKEN_SOURCES
        )

        with (
            patch.dict(os.environ, {"MCP_JWT_TOKEN": _token()}),
            pytest.raises(MCPAuthError),
        ):
            _run(middleware, {})

    def test_query_token_is_refused(self):
        """Query-parameter tokens leak into logs; never accept one."""
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=STRICT_TOKEN_SOURCES
        )
        context = {"query_params": {"token": _token()}}

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(MCPAuthError),
        ):
            _run(middleware, context)

    def test_metadata_token_is_refused(self):
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=STRICT_TOKEN_SOURCES
        )
        context = {"metadata": {"token": _token()}}

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(MCPAuthError),
        ):
            _run(middleware, context)

    def test_an_env_token_cannot_rescue_a_bad_header(self):
        """The header is tried and fails; the env must not be a fallback."""
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=STRICT_TOKEN_SOURCES
        )
        context = {"headers": {"Authorization": "Bearer not-a-real-token"}}

        with (
            patch.dict(os.environ, {"MCP_JWT_TOKEN": _token()}),
            pytest.raises(MCPAuthError),
        ):
            _run(middleware, context)


class TestExplicitSourceSelection:
    def test_a_named_subset_is_honoured(self):
        middleware = create_jwt_auth_middleware(
            _settings(), token_sources=("env",)
        )

        with patch.dict(os.environ, {"MCP_JWT_TOKEN": _token()}):
            assert _run(middleware, {})["ok"] is True

        context = {"headers": {"Authorization": f"Bearer {_token()}"}}
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(MCPAuthError),
        ):
            _run(middleware, context)

    def test_unknown_source_is_rejected_at_construction(self):
        """Fail when the middleware is built, not on the first request."""
        with pytest.raises(ValueError, match="Unknown token source"):
            create_jwt_auth_middleware(_settings(), token_sources=("headers",))

    def test_empty_source_list_is_rejected(self):
        """An empty tuple would lock everyone out; that is a typo, not intent."""
        with pytest.raises(ValueError, match="at least one source"):
            create_jwt_auth_middleware(_settings(), token_sources=())


class TestRequireAuthStillShortCircuits:
    def test_disabled_auth_bypasses_every_source_check(self):
        """require_auth=False is unchanged: it is the caller's decision."""
        settings = JWTAuthSettings(jwt_secret=SECRET, require_auth=False)
        middleware = create_jwt_auth_middleware(
            settings, token_sources=STRICT_TOKEN_SOURCES
        )

        with patch.dict(os.environ, {}, clear=True):
            assert _run(middleware, {})["ok"] is True
