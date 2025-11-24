"""Health check utilities for service monitoring.

This module provides a flexible health check framework for verifying the availability
and readiness of various services (databases, search engines, LLMs, caches, etc.)
before application startup or for monitoring endpoints.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        return result


class HealthChecker:
    """Service health checker with support for multiple check types.
    
    This class allows registering multiple health checks and running them
    either sequentially or in parallel. It's designed to be flexible and
    work with any service that can be checked via a callable.
    
    Example:
        >>> checker = HealthChecker()
        >>> 
        >>> # Register individual checks
        >>> checker.add_check("database", check_database_connection)
        >>> checker.add_check("cache", check_redis_connection)
        >>> 
        >>> # Run all checks
        >>> results = checker.run_all_checks()
        >>> if checker.is_healthy(results):
        >>>     print("All services healthy")
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """Initialize health checker.
        
        Args:
            logger: Optional logger instance for health check messages
        """
        self._checks: List[Tuple[str, Callable[[], HealthCheckResult]]] = []
        self._logger = logger
    
    def add_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a health check.
        
        Args:
            name: Name of the service being checked
            check_func: Callable that returns a HealthCheckResult
        """
        self._checks.append((name, check_func))
    
    def add_simple_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        healthy_message: str = "Service is healthy",
        unhealthy_message: str = "Service is unhealthy",
    ) -> None:
        """Register a simple boolean health check.
        
        Args:
            name: Name of the service being checked
            check_func: Callable that returns True if healthy, False otherwise
            healthy_message: Message to use when check passes
            unhealthy_message: Message to use when check fails
        """
        def wrapper() -> HealthCheckResult:
            try:
                is_healthy = check_func()
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                    message=healthy_message if is_healthy else unhealthy_message,
                )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                )
        
        self.add_check(name, wrapper)
    
    def add_dict_check(
        self,
        name: str,
        check_func: Callable[[], Dict[str, Any]],
        healthy_key: str = "healthy",
    ) -> None:
        """Register a health check that returns a dictionary.
        
        Args:
            name: Name of the service being checked
            check_func: Callable that returns a dict with health status
            healthy_key: Key in the dict that indicates health status
        """
        def wrapper() -> HealthCheckResult:
            try:
                result = check_func()
                is_healthy = result.get(healthy_key, False)
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                    details=result,
                )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                )
        
        self.add_check(name, wrapper)
    
    def add_callable_check(
        self,
        name: str,
        service: Any,
        method_name: str = "health_check",
    ) -> None:
        """Register a health check using a method on a service object.
        
        Args:
            name: Name of the service being checked
            service: Service object with a health check method
            method_name: Name of the health check method (default: "health_check")
        """
        def wrapper() -> HealthCheckResult:
            try:
                if not hasattr(service, method_name):
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Service does not have {method_name} method",
                    )
                
                result = getattr(service, method_name)()
                
                # Handle different return types
                if isinstance(result, HealthCheckResult):
                    return result
                elif isinstance(result, dict):
                    is_healthy = result.get("healthy", result.get("overall_healthy", False))
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                        details=result,
                    )
                elif isinstance(result, bool):
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    )
                else:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Unexpected result type: {type(result)}",
                    )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                )
        
        self.add_check(name, wrapper)
    
    def run_all_checks(self, fail_fast: bool = False) -> List[HealthCheckResult]:
        """Run all registered health checks.
        
        Args:
            fail_fast: If True, stop at first failure
            
        Returns:
            List of health check results
        """
        results: List[HealthCheckResult] = []
        
        for name, check_func in self._checks:
            try:
                if self._logger:
                    self._logger.debug(f"Running health check: {name}")
                
                result = check_func()
                results.append(result)
                
                if self._logger:
                    if result.is_healthy:
                        self._logger.info(f"Health check passed: {name}")
                    else:
                        self._logger.warning(
                            f"Health check failed: {name} - {result.message or 'No message'}"
                        )
                
                if fail_fast and not result.is_healthy:
                    break
                    
            except Exception as e:
                if self._logger:
                    self._logger.exception(f"Health check raised exception: {name}")
                
                results.append(
                    HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Exception: {str(e)}",
                    )
                )
                
                if fail_fast:
                    break
        
        return results
    
    def is_healthy(self, results: Optional[List[HealthCheckResult]] = None) -> bool:
        """Check if all health checks passed.
        
        Args:
            results: Optional list of results to check. If None, runs all checks.
            
        Returns:
            True if all checks are healthy
        """
        if results is None:
            results = self.run_all_checks()
        
        return all(result.is_healthy for result in results)
    
    def get_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Get a summary of health check results.
        
        Args:
            results: List of health check results
            
        Returns:
            Dictionary with summary information
        """
        total = len(results)
        healthy = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        unhealthy = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        degraded = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
        unknown = sum(1 for r in results if r.status == HealthStatus.UNKNOWN)
        
        return {
            "overall_healthy": all(r.is_healthy for r in results),
            "total_checks": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "degraded": degraded,
            "unknown": unknown,
            "checks": [r.to_dict() for r in results],
        }


def create_lazy_check(
    name: str,
    factory: Callable[[], Any],
    method_name: str = "health_check",
) -> Callable[[], HealthCheckResult]:
    """Create a health check that lazily initializes the service.
    
    This is useful when you want to register a health check for a service
    that may not be initialized yet.
    
    Args:
        name: Name of the service
        factory: Callable that creates/returns the service instance
        method_name: Name of the health check method on the service
        
    Returns:
        Callable that performs the health check
    """
    def check() -> HealthCheckResult:
        try:
            service = factory()
            if service is None:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Service factory returned None",
                )
            
            if not hasattr(service, method_name):
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Service does not have {method_name} method",
                )
            
            result = getattr(service, method_name)()
            
            # Handle different return types
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, dict):
                is_healthy = result.get("healthy", result.get("overall_healthy", False))
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                    details=result,
                )
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                )
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
            )
    
    return check
