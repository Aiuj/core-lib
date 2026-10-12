"""OpenRouter workload identity federation (token exchange).

Exchanges a JWT issued by your identity provider for a short-lived OpenRouter
access token, so no long-lived OpenRouter API key has to be stored.

Docs: https://openrouter.ai/docs/guides/overview/auth/workload-identity-federation

An instance is callable and returns a valid access token, re-exchanging
shortly before expiry. The OpenAI SDK accepts a callable ``api_key`` and
invokes it before each request, so it can be passed in place of a string.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

import httpx

DEFAULT_TOKEN_URL = "https://openrouter.ai/api/v1/oauth/token"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"

# Re-exchange this many seconds before the access token expires.
_REFRESH_MARGIN_S = 60

SubjectTokenSource = Union[str, Callable[[], str]]


class OpenRouterFederationError(RuntimeError):
    """Raised when the token exchange fails."""


class OpenRouterWorkloadIdentity:
    """Callable that yields a cached OpenRouter access token.

    Args:
        policy_id: Federation policy ID (Settings -> Workload identity).
        subject_token: The IdP JWT, or a callable returning a fresh one.
            Prefer a callable or ``subject_token_file`` for IdP tokens that
            rotate (e.g. Kubernetes projected service account tokens).
        subject_token_file: Path re-read on every exchange.
        token_url: Override the exchange endpoint.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        policy_id: str,
        subject_token: Optional[SubjectTokenSource] = None,
        subject_token_file: Optional[str] = None,
        token_url: str = DEFAULT_TOKEN_URL,
        timeout: float = 15.0,
    ) -> None:
        if not policy_id:
            raise ValueError("OpenRouter federation policy_id is required")
        if not subject_token and not subject_token_file:
            raise ValueError("Provide subject_token or subject_token_file")
        self.policy_id = policy_id
        self.token_url = token_url
        self.timeout = timeout
        self._subject_token = subject_token
        self._subject_token_file = subject_token_file
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at = 0.0

    @classmethod
    def from_env(
        cls, overrides: Optional[Dict[str, Any]] = None
    ) -> Optional["OpenRouterWorkloadIdentity"]:
        """Build from a config dict and/or environment; None if not configured.

        Env vars: ``OPENROUTER_FEDERATION_POLICY_ID``, ``OPENROUTER_IDP_TOKEN``,
        ``OPENROUTER_IDP_TOKEN_FILE``, ``OPENROUTER_TOKEN_URL``.
        """
        cfg = overrides or {}
        policy_id = cfg.get("policy_id") or os.getenv("OPENROUTER_FEDERATION_POLICY_ID")
        if not policy_id:
            return None
        token = cfg.get("subject_token") or os.getenv("OPENROUTER_IDP_TOKEN")
        token_file = cfg.get("subject_token_file") or os.getenv("OPENROUTER_IDP_TOKEN_FILE")
        if not token and not token_file:
            return None
        return cls(
            policy_id=str(policy_id),
            subject_token=token,
            subject_token_file=token_file,
            token_url=cfg.get("token_url")
            or os.getenv("OPENROUTER_TOKEN_URL")
            or DEFAULT_TOKEN_URL,
        )

    def _read_subject_token(self) -> str:
        if self._subject_token_file:
            with open(self._subject_token_file, "r", encoding="utf-8") as fh:
                token = fh.read().strip()
        elif callable(self._subject_token):
            token = self._subject_token()
        else:
            token = self._subject_token or ""
        if not token:
            raise OpenRouterFederationError("IdP subject token is empty")
        return token

    def _exchange(self) -> None:
        data = {
            "grant_type": GRANT_TYPE,
            "subject_token_type": SUBJECT_TOKEN_TYPE,
            "subject_token": self._read_subject_token(),
            "federation_policy_id": self.policy_id,
            "requested_token_type": REQUESTED_TOKEN_TYPE,
            "scope": "inference",
        }
        try:
            resp = httpx.post(self.token_url, data=data, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise OpenRouterFederationError(f"Token exchange request failed: {exc}") from exc
        if resp.status_code != 200:
            try:
                body = resp.json()
                detail = body.get("error_description") or body.get("error") or resp.text
            except ValueError:
                detail = resp.text
            raise OpenRouterFederationError(
                f"Token exchange failed (HTTP {resp.status_code}): {detail}"
            )
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise OpenRouterFederationError("Token exchange response had no access_token")
        self._access_token = token
        self._expires_at = time.monotonic() + float(payload.get("expires_in", 900))

    def get_token(self) -> str:
        with self._lock:
            if (
                not self._access_token
                or time.monotonic() >= self._expires_at - _REFRESH_MARGIN_S
            ):
                self._exchange()
            return self._access_token  # type: ignore[return-value]

    def __call__(self) -> str:
        return self.get_token()
