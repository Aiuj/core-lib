"""Contract tests for shared LLM, embedding, and reranker warmup routing."""

from core_lib.api_utils import FallbackProviderRouting, WarmupFallbackRouter


class _Provider:
    def __init__(self, warming: bool = False):
        self.warming = warming

    def is_in_warmup(self) -> bool:
        return self.warming


class _FallbackRoute(FallbackProviderRouting):
    def __init__(self, providers) -> None:
        self.providers = providers
        self._init_fallback_routing()


def test_shared_router_skips_and_restores_wol_primary() -> None:
    router = WarmupFallbackRouter()
    primary = _Provider(warming=True)
    fallback = _Provider(warming=False)
    providers = {0: primary, 1: fallback}

    assert router.is_warming(0, primary) is True
    assert router.prioritize_recovered(
        [1, 0], key=lambda idx: idx, provider=lambda idx: providers[idx]
    ) == [1, 0]

    primary.warming = False
    assert router.prioritize_recovered(
        [1, 0], key=lambda idx: idx, provider=lambda idx: providers[idx]
    ) == [0, 1]

    router.mark_success(0)
    assert router.prioritize_recovered(
        [1, 0], key=lambda idx: idx, provider=lambda idx: providers[idx]
    ) == [1, 0]


def test_shared_router_treats_missing_or_broken_warmup_as_inactive() -> None:
    router = WarmupFallbackRouter()

    class _NoWarmup:
        pass

    class _BrokenWarmup:
        def is_in_warmup(self):
            raise RuntimeError("probe failed")

    assert router.is_warming("none", _NoWarmup()) is False
    assert router.is_warming("broken", _BrokenWarmup()) is False


def test_shared_fallback_routing_deduplicates_and_restores_primary() -> None:
    primary = _Provider(warming=True)
    fallback = _Provider()
    route = _FallbackRoute([primary, fallback])

    assert route._is_provider_warming(0) is True
    assert route._ordered_provider_indices([1, 0, 1, 3]) == [1, 0]

    primary.warming = False
    assert route._ordered_provider_indices([1, 0, 1]) == [0, 1]

    route._record_provider_success(0)
    assert route.current_provider_index == 0
    assert route._selected_provider() is primary
