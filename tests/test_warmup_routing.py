"""Contract tests for shared LLM, embedding, and reranker warmup routing."""

from core_lib.api_utils import WarmupFallbackRouter


class _Provider:
    def __init__(self, warming: bool = False):
        self.warming = warming

    def is_in_warmup(self) -> bool:
        return self.warming


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
