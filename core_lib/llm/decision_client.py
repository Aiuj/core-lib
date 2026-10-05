"""Native System One decisions through TypeSafe Jev or local Ollama.

These models answer typed questions, not chat completions. Both transports
use /v1/systemone; generation settings and tool schemas are never sent.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from core_lib.tracing.service_usage import log_llm_usage

logger = logging.getLogger(__name__)

DECISION_PROVIDERS = frozenset({"typesafe", "ollama-decision"})


@dataclass(frozen=True)
class Choice:
    instructions: Any
    criteria: Mapping[str, Any]
    type: str = "choice"


@dataclass(frozen=True)
class Noul:
    instructions: Any
    criteria: Mapping[str, Any] | None = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    instructions: Any
    criteria: list[Any]
    type: str = "score"


class DecisionError(RuntimeError):
    """Safe error that does not include credentials, state or response bodies."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and math.isfinite(value)
    )


class DecisionClient:
    """Synchronous, reusable HTTP client for Choice, Noul and Score.

    ``provider`` is ``typesafe`` or ``ollama-decision``. Returned dictionaries
    preserve native answers and usage, plus provider/requested_model/latency_ms.
    Callers decide how confidence affects routing; it grants no authorization.
    """

    def __init__(
        self,
        provider: str = "typesafe",
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30,
        max_retries: int = 2,
        max_retry_delay: float = 30,
        keep_alive: str | int | None = None,
        http_client: httpx.Client | None = None,
    ):
        if provider not in DECISION_PROVIDERS:
            raise ValueError("Decision provider must be typesafe or ollama-decision")
        if timeout <= 0 or max_retry_delay < 0 or max_retries < 0:
            raise ValueError("Invalid decision timeout or retry limits")
        self.provider = provider
        local = provider == "ollama-decision"
        self.model = model or ("nimble" if local else "jev-latest")
        if not self.model.strip():
            raise ValueError("Decision model must not be empty")
        default_url = (
            (
                os.getenv("OLLAMA_BASE_URL")
                or os.getenv("OLLAMA_HOST")
                or "http://localhost:11434"
            )
            if local
            else os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
        )
        self.base_url = (base_url or default_url).rstrip("/")
        # Accept either the server root or an API base ending in /v1.
        self.api_url = (
            self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        )
        self._api_key = api_key or (None if local else os.getenv("TYPESAFE_API_KEY"))
        if not local and not self._api_key:
            raise ValueError("TYPESAFE_API_KEY is required for TypeSafe decisions")
        if keep_alive is not None and not local:
            raise ValueError("keep_alive is supported only by local Ollama")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay
        self.keep_alive = keep_alive
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=timeout, follow_redirects=False
        )

    @classmethod
    def from_provider_config(cls, config, **overrides):
        """Consume the existing YAML registry's provider configuration."""
        if not config.is_decision_provider:
            raise ValueError("This configuration is not a decision provider")
        options = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.host,
            "timeout": (
                config.http_timeout_ms / 1000
                if config.http_timeout_ms
                else config.extra.get("timeout", 30)
            ),
            "max_retries": config.extra.get("max_retries", 2),
            "max_retry_delay": config.extra.get("max_retry_delay", 30),
            "keep_alive": config.extra.get("keep_alive"),
        }
        options.update(overrides)
        return cls(config.provider, **options)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _questions(self, questions: Mapping[str, Any]) -> dict:
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty map")
        result = {}
        local = self.provider == "ollama-decision"
        for name, value in questions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Question IDs must be nonempty strings")
            if isinstance(value, (Choice, Noul, Score)):
                value = {k: v for k, v in asdict(value).items() if v is not None}
            if not isinstance(value, Mapping):
                raise ValueError("Each question must be a typed question or map")
            value = dict(value)
            kind = value.get("type")
            instructions = value.get("instructions")
            if kind not in {"choice", "noul", "score"}:
                raise ValueError("Unsupported decision question type")
            if (
                not isinstance(instructions, (str, dict, list))
                or not instructions
                or (isinstance(instructions, str) and not instructions.strip())
            ):
                raise ValueError(
                    "Question instructions must be nonempty text, object or array"
                )
            if set(value) - {"type", "instructions", "criteria"}:
                raise ValueError("Unknown decision question fields")
            criteria = value.get("criteria")
            if kind == "choice":
                limit = 26 if local else 255
                if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= limit:
                    raise ValueError(f"Choice requires 2–{limit} options")
                if any(not isinstance(key, str) or not key.strip() for key in criteria):
                    raise ValueError("Choice labels must be nonempty strings")
            elif kind == "score":
                limit = 26 if local else 10
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= limit:
                    raise ValueError(f"Score requires 2–{limit} ordered levels")
            elif criteria is not None and (
                not isinstance(criteria, Mapping) or set(criteria) != {"true", "false"}
            ):
                raise ValueError("Noul criteria must describe true and false")
            result[name] = value
        return result

    @staticmethod
    def _validate_response(data: Any, questions: Mapping) -> None:
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("model"), str)
            or not data["model"].strip()
        ):
            raise DecisionError("Decision API returned an invalid model")
        answers = data.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise DecisionError("Decision API returned mismatched answers")
        for name, question in questions.items():
            answer = answers[name]
            kind = question["type"]
            if not isinstance(answer, dict) or answer.get("type") != kind:
                raise DecisionError("Decision API returned an invalid answer type")
            if kind == "noul":
                if not _probability(answer.get("noul")):
                    raise DecisionError(
                        "Decision API returned an invalid yes/no probability"
                    )
                continue
            probabilities = answer.get("probabilities")
            labels = (
                set(question["criteria"])
                if kind == "choice"
                else {str(i) for i in range(len(question["criteria"]))}
            )
            if (
                not isinstance(probabilities, dict)
                or set(probabilities) != labels
                or not all(_probability(p) for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.002)
                or not _probability(answer.get("confidence"))
            ):
                raise DecisionError(
                    "Decision API returned an invalid probability distribution"
                )
            if kind == "choice":
                if (
                    not isinstance(answer.get("choice"), str)
                    or answer["choice"] not in labels
                ):
                    raise DecisionError("Decision API selected an unknown option")
                if probabilities[answer["choice"]] < max(probabilities.values()):
                    raise DecisionError(
                        "Decision API selected an option inconsistent with its probabilities"
                    )
            else:
                score = answer.get("score")
                if (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not 0 <= score <= len(labels) - 1
                    or not math.isfinite(score)
                ):
                    raise DecisionError("Decision API returned an invalid score")
        usage = data.get("usage")
        if not isinstance(usage, dict) or any(
            not isinstance(usage.get(key), int)
            or isinstance(usage[key], bool)
            or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise DecisionError("Decision API returned invalid token usage")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return max(
                        0, (retry_at - datetime.now(timezone.utc)).total_seconds()
                    )
                except (ValueError, TypeError):
                    pass
        return min(self.max_retry_delay, 0.5 * 2**attempt + random.uniform(0, 0.1))

    def decide(self, *, state: str | dict | list, questions: Mapping[str, Any]) -> dict:
        """Evaluate all questions in one call. Never reinterpret state as chat."""
        if (
            not isinstance(state, (str, dict, list))
            or not state
            or (isinstance(state, str) and not state.strip())
        ):
            raise ValueError("state must be nonempty text, object or array")
        questions = self._questions(questions)
        payload = {"model": self.model, "state": state, "questions": questions}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if self.provider == "ollama-decision" and len(body) > 64 * 1024:
            raise ValueError("Ollama text decision request exceeds 64 KiB")
        started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post(
                    self.api_url + "/systemone",
                    content=body,
                    headers=self._headers(),
                    timeout=self.timeout,
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                # A timeout can represent a completed, billable request. Do not replay it.
                raise DecisionError(
                    f"{self.provider} decision request failed in transport"
                ) from None
            if (
                response.status_code in {429, 502, 503, 504, 529}
                and attempt < self.max_retries
            ):
                delay = self._retry_delay(response, attempt)
                if delay <= self.max_retry_delay:
                    time.sleep(delay)
                    continue
            if not response.is_success:
                raise DecisionError(
                    f"{self.provider} decision API returned HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError:
                raise DecisionError("Decision API returned invalid JSON") from None
            self._validate_response(data, questions)
            latency_ms = (time.monotonic() - started) * 1000
            try:
                log_llm_usage(
                    provider=self.provider,
                    model=data["model"],
                    input_tokens=data["usage"]["input_tokens"],
                    output_tokens=data["usage"]["output_tokens"],
                    latency_ms=latency_ms,
                    structured=True,
                    response_format="decision",
                    purpose="decision",
                    host=self._safe_host(),
                    metadata={
                        "requested_model": self.model,
                        "question_count": len(questions),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log decision usage: {e}")
            return {
                **data,
                "provider": self.provider,
                "requested_model": self.model,
                "latency_ms": latency_ms,
            }
        raise AssertionError("Unreachable retry state")

    system_one = decide

    def _safe_host(self) -> str:
        """Endpoint for telemetry with any userinfo/query/fragment stripped."""
        try:
            parts = urlsplit(self.base_url)
            host = parts.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            if parts.port:
                host = f"{host}:{parts.port}"
            return f"{parts.scheme}://{host}{parts.path}" if host else ""
        except ValueError:
            return ""

    def list_models(self) -> list[dict]:
        """List available models without consuming decision tokens."""
        local = self.provider == "ollama-decision"
        root = self.base_url.removesuffix("/v1")
        url = root + "/api/tags" if local else self.api_url + "/models"
        try:
            response = self._http.get(
                url,
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            raise DecisionError(
                f"{self.provider} model listing failed in transport"
            ) from None
        if not response.is_success:
            raise DecisionError(
                f"{self.provider} model listing returned HTTP {response.status_code}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError:
            raise DecisionError(
                "Decision model listing returned invalid JSON"
            ) from None
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("models"), list)
            or any(
                not isinstance(item, dict) or not isinstance(item.get("name"), str)
                for item in data["models"]
            )
        ):
            raise DecisionError("Decision model listing returned invalid models")
        return data["models"]

    def close(self):
        if self._owns_client:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_decision_client(
    provider: str | None = None, *, registry=None, **options
) -> DecisionClient:
    """Create explicitly, or select the first configured decision YAML entry.

    Passing a registry plus provider selects that provider within the registry.
    There is no automatic cloud/local failover: comparisons keep the requested
    provider, and local state is never silently sent to a cloud API.
    """
    if provider is not None and registry is None:
        return DecisionClient(provider, **options)
    if registry is None:
        from .provider_registry import ProviderRegistry

        registry = ProviderRegistry.from_env()
    candidates = [
        p
        for p in registry.decision_providers
        if provider is None or p.provider == provider
    ]
    if not candidates:
        raise ValueError("No configured decision provider matches the request")
    return DecisionClient.from_provider_config(candidates[0], **options)
