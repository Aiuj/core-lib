"""Shared Infinity API client with multi-server failover support.

Provides a unified HTTP client for Infinity embedding and reranking servers
with automatic failover between multiple URLs for high availability.
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None

from core_lib.tracing.logger import get_module_logger
from .wake_on_lan import WakeOnLanStrategy

logger = get_module_logger()


class InfinityAPIError(Exception):
    """Base exception for Infinity API errors."""
    pass


class InfinityAPIClient:
    """HTTP client for Infinity API with automatic multi-server failover.
    
    Supports comma-separated URLs for high availability. Automatically retries
    failed requests on backup servers.
    
    Example:
        ```python
        client = InfinityAPIClient(
            base_urls="http://server1:7997,http://server2:7997",
            timeout=30
        )
        
        response = client.post("/embeddings", json={"model": "...", "input": [...]})
        ```
    """
    
    def __init__(
        self,
        base_urls: str | List[str],
        timeout: int = 30,
        token: Optional[str] = None,
        max_retries_per_url: int = 1,
        wake_on_lan: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Infinity API client with failover support.
        
        Args:
            base_urls: Single URL or comma-separated URLs or list of URLs
            timeout: Request timeout in seconds
            token: Optional authentication token
            max_retries_per_url: How many times to retry each URL before moving to next
            wake_on_lan: Optional Wake-on-LAN config for sleeping hosts
        """
        
        # Parse URLs
        if isinstance(base_urls, str):
            # Handle comma-separated URLs
            self.base_urls = [url.strip().rstrip('/') for url in base_urls.split(',')]
        elif isinstance(base_urls, list):
            self.base_urls = [url.strip().rstrip('/') for url in base_urls]
        else:
            raise ValueError(f"base_urls must be string or list, got {type(base_urls)}")
        
        if not self.base_urls:
            raise ValueError("At least one base URL must be provided")
        
        # Validate URLs
        for url in self.base_urls:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")
        
        self.timeout = timeout
        self.token = token
        self.max_retries_per_url = max_retries_per_url
        self.wake_on_lan = WakeOnLanStrategy(wake_on_lan)
        
        # Track current working URL to prefer it on next request
        self.current_url_index = 0
        self.url_failures: Dict[int, int] = {i: 0 for i in range(len(self.base_urls))}
        
        logger.debug(
            f"Initialized InfinityAPIClient with {len(self.base_urls)} URLs: "
            f"{', '.join(self.base_urls)}"
        )

    def _is_connection_or_timeout_error(self, error: Exception) -> bool:
        """Return True when error indicates host may be sleeping/unreachable."""
        if requests is None:
            return False
        return isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    
    def _build_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Make POST request with automatic failover.
        
        Args:
            endpoint: API endpoint (e.g., '/embeddings', '/rerank')
            json: Request body as dict
            headers: Additional headers
            timeout: Override default timeout
            
        Returns:
            Tuple of (response_data, url_used)
            
        Raises:
            InfinityAPIError: If all URLs fail
        """
        endpoint = endpoint.lstrip('/')
        timeout = timeout or self.timeout
        headers = self._build_headers(headers)
        
        errors = []
        urls_tried = []
        
        # Try each URL, starting with the last successful one
        for attempt in range(len(self.base_urls)):
            url_index = (self.current_url_index + attempt) % len(self.base_urls)
            base_url = self.base_urls[url_index]
            full_url = f"{base_url}/{endpoint}"
            urls_tried.append(full_url)
            
            for retry in range(self.max_retries_per_url):
                try:
                    start_time = time.time()
                    effective_timeout = self.wake_on_lan.maybe_get_initial_timeout(
                        base_url,
                        timeout,
                    )
                    response = requests.post(
                        full_url,
                        json=json,
                        headers=headers,
                        timeout=effective_timeout
                    )
                    
                    # Raise for HTTP errors
                    response.raise_for_status()
                    
                    # Success - update current URL index and reset failure count
                    self.current_url_index = url_index
                    self.url_failures[url_index] = 0
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"Infinity API request succeeded: {endpoint} @ {base_url} "
                        f"({elapsed_ms:.0f}ms)"
                    )
                    
                    return response.json(), base_url
                
                except requests.exceptions.Timeout as e:
                    wake_result = self.wake_on_lan.maybe_wake(base_url, e)
                    if wake_result.succeeded:
                        retry_timeout = wake_result.retry_timeout_seconds or timeout
                        logger.info(
                            f"Retrying Infinity request after WoL wake: {full_url} "
                            f"with timeout={retry_timeout}s"
                        )
                        try:
                            response = requests.post(
                                full_url,
                                json=json,
                                headers=headers,
                                timeout=retry_timeout,
                            )
                            response.raise_for_status()
                            self.current_url_index = url_index
                            self.url_failures[url_index] = 0
                            return response.json(), base_url
                        except Exception as retry_error:
                            error_msg = f"Post-wake retry failed: {full_url} - {retry_error}"
                            errors.append(error_msg)
                            self.url_failures[url_index] += 1
                            logger.warning(error_msg)

                    error_msg = f"Timeout after {effective_timeout}s: {full_url}"
                    errors.append(error_msg)
                    self.url_failures[url_index] += 1
                    logger.warning(
                        f"{error_msg} (attempt {retry + 1}/{self.max_retries_per_url})"
                    )
                    
                except requests.exceptions.ConnectionError as e:
                    wake_result = self.wake_on_lan.maybe_wake(base_url, e)
                    if wake_result.succeeded:
                        retry_timeout = wake_result.retry_timeout_seconds or timeout
                        logger.info(
                            f"Retrying Infinity request after WoL wake: {full_url} "
                            f"with timeout={retry_timeout}s"
                        )
                        try:
                            response = requests.post(
                                full_url,
                                json=json,
                                headers=headers,
                                timeout=retry_timeout,
                            )
                            response.raise_for_status()
                            self.current_url_index = url_index
                            self.url_failures[url_index] = 0
                            return response.json(), base_url
                        except Exception as retry_error:
                            error_msg = f"Post-wake retry failed: {full_url} - {retry_error}"
                            errors.append(error_msg)
                            self.url_failures[url_index] += 1
                            logger.warning(error_msg)

                    error_msg = f"Connection failed: {full_url} - {str(e)}"
                    errors.append(error_msg)
                    self.url_failures[url_index] += 1
                    logger.warning(
                        f"{error_msg} (attempt {retry + 1}/{self.max_retries_per_url})"
                    )
                    break  # No point retrying connection errors
                    
                except requests.exceptions.HTTPError as e:
                    # For HTTP errors, try to extract detailed message
                    error_detail = None
                    try:
                        error_data = response.json()
                        if 'error' in error_data and 'message' in error_data['error']:
                            error_detail = error_data['error']['message']
                    except:
                        pass
                    
                    error_msg = (
                        f"HTTP {response.status_code}: {full_url}"
                        + (f" - {error_detail}" if error_detail else "")
                    )
                    errors.append(error_msg)
                    self.url_failures[url_index] += 1
                    logger.warning(
                        f"{error_msg} (attempt {retry + 1}/{self.max_retries_per_url})"
                    )
                    
                    # Don't retry 4xx errors (except 429 rate limit)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
                    
                except Exception as e:
                    error_msg = f"Unexpected error: {full_url} - {str(e)}"
                    errors.append(error_msg)
                    self.url_failures[url_index] += 1
                    logger.warning(
                        f"{error_msg} (attempt {retry + 1}/{self.max_retries_per_url})"
                    )
        
        # All URLs failed
        error_summary = (
            f"All Infinity servers failed for {endpoint}. "
            f"Tried {len(urls_tried)} URL(s): {', '.join(urls_tried)}. "
            f"Errors: {'; '.join(errors[:3])}"  # Limit to first 3 errors
        )
        logger.error(error_summary)
        raise InfinityAPIError(error_summary)
    
    def get(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Make GET request with automatic failover.
        
        Args:
            endpoint: API endpoint (e.g., '/health', '/models')
            headers: Additional headers
            timeout: Override default timeout
            
        Returns:
            Tuple of (response_data, url_used)
            
        Raises:
            InfinityAPIError: If all URLs fail
        """
        endpoint = endpoint.lstrip('/')
        timeout = timeout or self.timeout
        headers = self._build_headers(headers)
        
        errors = []
        
        # Try each URL
        for url_index, base_url in enumerate(self.base_urls):
            full_url = f"{base_url}/{endpoint}"
            
            try:
                response = requests.get(
                    full_url,
                    headers=headers,
                    timeout=timeout
                )
                
                response.raise_for_status()
                
                # Success
                return response.json(), base_url
                
            except Exception as e:
                error_msg = f"{full_url}: {str(e)}"
                errors.append(error_msg)
                logger.debug(f"GET request failed: {error_msg}")
        
        # All URLs failed
        error_summary = f"All servers failed for GET {endpoint}: {'; '.join(errors)}"
        logger.warning(error_summary)
        raise InfinityAPIError(error_summary)
    
    def health_check(self) -> bool:
        """Check if at least one Infinity server is healthy.
        
        Returns:
            True if any server responds successfully
        """
        for base_url in self.base_urls:
            try:
                # Try health endpoint
                response = requests.get(
                    f"{base_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    logger.debug(f"Health check passed: {base_url}")
                    return True
            except Exception as e:
                logger.debug(f"Health check failed for {base_url}: {e}")
                continue
        
        logger.warning("Health check failed for all Infinity servers")
        return False
    
    def get_url_status(self) -> List[Dict[str, Any]]:
        """Get status information for all configured URLs.
        
        Returns:
            List of dicts with URL status info
        """
        status = []
        for i, url in enumerate(self.base_urls):
            status.append({
                'url': url,
                'index': i,
                'is_current': i == self.current_url_index,
                'failures': self.url_failures.get(i, 0),
            })
        return status
