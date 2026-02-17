"""core_lib.llm.retry

Generic retry and circuit breaker utilities for API clients.

This module provides decorators and configuration to add resilient retry
behavior to functions that perform network requests. It supports several
common backoff strategies and can be configured to retry on specific
classes of errors (e.g., rate limits, server errors, network issues).

The primary entry point is the `retry_handler` decorator, which wraps a
synchronous or asynchronous function in retry logic according to a
provided `RetryConfig`.

Example:
    from core_lib.llm.retry import RetryConfig, retry_handler

    retry_config = RetryConfig(max_retries=3)

    @retry_handler(retry_config)
    def my_flaky_api_call(param: str) -> dict:
        # ... logic that might raise an exception ...
        return {"data": "success"}

    # The call will be retried automatically on failure
    result = my_flaky_api_call("test")
"""

import asyncio
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Type, Coroutine, Dict, Tuple, Optional
from dataclasses import dataclass, field

from core_lib import get_module_logger

logger = get_module_logger()


class RetryStrategy(Enum):
    """Retry strategy options for API failures."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry logic.

    Attributes:
        max_retries: Maximum number of retry attempts before giving up.
        base_delay: The initial delay in seconds for the first retry.
        max_delay: The maximum delay in seconds between retries.
        strategy: The backoff strategy to use (e.g., exponential).
        retry_on_exceptions: A tuple of exception types that should trigger a
            retry. This allows fine-grained control over what is retryable.
        jitter_factor: A factor to add random jitter to delays to prevent
            thundering herd problems. Delay is multiplied by a random number
            in [1, 1 + jitter_factor]. Defaults to 0.5.
    """
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    jitter_factor: float = 0.5


def retry_handler(config: RetryConfig) -> Callable:
    """A decorator that adds retry logic to a function or coroutine.

    It inspects the decorated function to determine if it is synchronous or
    asynchronous and applies the appropriate sleep implementation (`time.sleep`
    or `asyncio.sleep`).

    Args:
        config: A `RetryConfig` instance defining the retry behavior.

    Returns:
        A wrapper function that will retry the decorated function on failure.
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retry_on_exceptions as e:
                    last_exception = e
                    if attempt >= config.max_retries:
                        logger.error(
                            f"Final attempt failed for {func.__name__}. No more retries. "
                            f"{type(e).__name__}: {e}",
                            extra={"attempt": attempt, "max_retries": config.max_retries}
                        )
                        break  # Exit loop to re-raise

                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}. Retrying in {delay:.2f}s. "
                        f"{type(e).__name__}: {e}",
                        extra={"attempt": attempt, "delay": delay}
                    )
                    time.sleep(delay)
            raise last_exception  # type: ignore

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retry_on_exceptions as e:
                    last_exception = e
                    if attempt >= config.max_retries:
                        logger.error(
                            f"Final attempt failed for {func.__name__}. No more retries. "
                            f"{type(e).__name__}: {e}",
                            extra={"attempt": attempt, "max_retries": config.max_retries}
                        )
                        break

                    delay = _calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}. Retrying in {delay:.2f}s. "
                        f"{type(e).__name__}: {e}",
                        extra={"attempt": attempt, "delay": delay}
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate the sleep delay for the current retry attempt."""
    import random

    delay: float
    if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.base_delay * (2 ** attempt)
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.base_delay * (attempt + 1)
    else:  # FIXED_DELAY
        delay = config.base_delay

    # Apply jitter: delay * (1 + random * jitter_factor)
    jitter = delay * config.jitter_factor * random.random()
    return min(config.max_delay, delay + jitter)
