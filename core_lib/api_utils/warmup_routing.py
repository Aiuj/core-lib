"""Shared non-blocking warmup routing for fallback provider chains."""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, List, TypeVar

from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

T = TypeVar("T")


class WarmupFallbackRouter:
    """Track providers warming after WoL and restore their priority afterward.

    Embedding, reranker, and LLM fallback clients use this same state machine:

    * an active warmup provider is skipped immediately;
    * expected warmup failures do not demote provider health;
    * a provider whose warmup just expired is tried before the current fallback;
    * successful recovery clears its warmup tracking state.
    """

    def __init__(self) -> None:
        self._warming_keys: set[Hashable] = set()

    @staticmethod
    def _read_provider_warmup(provider: object) -> bool:
        is_in_warmup = getattr(provider, "is_in_warmup", None)
        if not callable(is_in_warmup):
            return False
        try:
            return bool(is_in_warmup() is True)
        except Exception as exc:
            logger.debug("Could not read provider warmup state: %s", exc)
            return False

    def is_warming(self, key: Hashable, provider: object) -> bool:
        """Return current warmup state and remember active providers."""
        active = self._read_provider_warmup(provider)
        if active:
            self._warming_keys.add(key)
        return active

    def prioritize_recovered(
        self,
        items: Iterable[T],
        *,
        key: Callable[[T], Hashable],
        provider: Callable[[T], object],
    ) -> List[T]:
        """Put providers whose warmup expired before the normal route order."""
        normal = list(items)
        recovered = [
            item
            for item in normal
            if key(item) in self._warming_keys
            and not self._read_provider_warmup(provider(item))
        ]
        seen = {key(item) for item in recovered}
        return recovered + [item for item in normal if key(item) not in seen]

    def mark_success(self, key: Hashable) -> None:
        """Clear warmup tracking after a recovered provider succeeds."""
        self._warming_keys.discard(key)

    def any_warming(self, keyed_providers: Iterable[tuple[Hashable, object]]) -> bool:
        """Return whether any supplied provider is currently warming."""
        return any(self.is_warming(key, provider) for key, provider in keyed_providers)
