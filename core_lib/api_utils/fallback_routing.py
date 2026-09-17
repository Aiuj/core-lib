"""Shared routing state for ordered fallback provider chains.

The concrete embedding and reranker clients deliberately retain their own
request, result, cache, and health-policy code.  This mixin owns the common
provider-selection behavior so their Wake-on-LAN routing cannot drift.
"""

from __future__ import annotations

from typing import Iterable, List

from .warmup_routing import WarmupFallbackRouter


class FallbackProviderRouting:
    """Manage provider order and non-blocking WoL warmup state.

    Hosts using this mixin must set ``self.providers`` before calling
    :meth:`_init_fallback_routing`.
    """

    def _init_fallback_routing(self) -> None:
        self.current_provider_index = 0
        self._warmup_router = WarmupFallbackRouter()

    def _ordered_provider_indices(self, candidates: Iterable[int]) -> List[int]:
        """Deduplicate candidates and prioritize providers just out of warmup."""
        seen: set[int] = set()
        normal: List[int] = []
        for index in candidates:
            if index in seen or not 0 <= index < len(self.providers):
                continue
            seen.add(index)
            normal.append(index)
        return self._warmup_router.prioritize_recovered(
            normal,
            key=lambda index: index,
            provider=lambda index: self.providers[index],
        )

    def _is_provider_warming(self, provider_index: int) -> bool:
        """Return whether a provider is in the non-blocking WoL window."""
        return self._warmup_router.is_warming(
            provider_index, self.providers[provider_index]
        )

    def _record_provider_success(self, provider_index: int) -> None:
        """Select a successful provider and clear its recovered-warmup state."""
        self.current_provider_index = provider_index
        self._warmup_router.mark_success(provider_index)

    def _selected_provider(self):
        """Return the provider that handled the most recent successful request."""
        return self.providers[self.current_provider_index]
