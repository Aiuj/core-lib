"""
Base API Client with Time-Based Authentication

Provides a reusable base class for HTTP API clients with built-in support
for time-based HMAC authentication, legacy API key authentication, and no auth.
"""

import httpx
from typing import Optional, Dict, Any, Tuple, BinaryIO, Union
from pathlib import Path
from .time_based_auth import generate_time_key

try:
    from core_lib import get_module_logger
    logger = get_module_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class APIClient:
    """Base API client with time-based authentication support.
    
    This class provides a foundation for building HTTP API clients with
    built-in authentication handling. It supports:
    - Time-based HMAC authentication (recommended)
    - Legacy static API key authentication
    - No authentication
    
    Authentication priority:
    1. Time-based auth (if auth_enabled=True and auth_private_key is set)
    2. Legacy API key (if api_key is set)
    3. No authentication
    
    Example:
        class MyAPIClient(APIClient):
            def get_data(self, item_id: str):
                headers = self._prepare_headers()
                response = httpx.get(
                    f"{self.base_url}/items/{item_id}",
                    headers=headers
                )
                return response.json()
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        auth_enabled: bool = False,
        auth_private_key: Optional[str] = None,
        auth_header_name: str = "x-auth-key",
        api_key_header_name: str = "x-api-key",
        timeout: float = 30.0,
        verify_ssl: bool = True,
        service_name: Optional[str] = None,
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API (e.g., http://localhost:9095)
            api_key: Optional static API key for legacy authentication
            auth_enabled: Enable time-based HMAC authentication
            auth_private_key: Private key for time-based auth (required if auth_enabled=True)
            auth_header_name: HTTP header name for time-based auth (default: x-auth-key)
            api_key_header_name: HTTP header name for legacy API key (default: x-api-key)
            timeout: Default request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (set False for self-signed certs)
            service_name: Human-readable name for this service (for logging)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.auth_enabled = auth_enabled
        self.auth_private_key = auth_private_key
        self.auth_header_name = auth_header_name
        self.api_key_header_name = api_key_header_name
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.service_name = service_name or self._derive_service_name(base_url)
        
        # Determine authentication method for logging
        auth_method = self._get_auth_method()
        logger.info(f"Initialized API client for {self.service_name} ({base_url}) with auth: {auth_method}")
    
    def _derive_service_name(self, base_url: str) -> str:
        """Derive a service name from the URL if not provided."""
        # Extract host:port from URL for use as fallback service name
        try:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname or "unknown"
        except Exception:
            return "unknown"
    
    def _get_auth_method(self) -> str:
        """Get the current authentication method as a string."""
        if self.auth_enabled and self.auth_private_key:
            return "time-based HMAC"
        elif self.api_key:
            return "static API key"
        else:
            return "none"
    
    def _prepare_headers(
        self,
        additional_headers: Optional[Dict[str, str]] = None,
        skip_auth: bool = False
    ) -> Dict[str, str]:
        """
        Prepare HTTP headers with authentication.
        
        Args:
            additional_headers: Optional additional headers to include
            skip_auth: If True, skip adding authentication headers
            
        Returns:
            Dictionary of HTTP headers ready for use
            
        Raises:
            Exception: If time-based auth key generation fails
        """
        headers = {"Content-Type": "application/json"}
        
        # Add additional headers if provided
        if additional_headers:
            headers.update(additional_headers)
        
        # Skip authentication if requested
        if skip_auth:
            return headers
        
        # Add authentication header based on configuration
        if self.auth_enabled and self.auth_private_key:
            # Use time-based HMAC authentication
            try:
                auth_key = generate_time_key(self.auth_private_key)
                headers[self.auth_header_name] = auth_key
                logger.debug(f"Generated time-based auth key for API request")
            except Exception as e:
                logger.error(f"Failed to generate time-based auth key: {e}")
                raise
        elif self.api_key:
            # Fall back to legacy API key authentication
            headers[self.api_key_header_name] = self.api_key
            logger.debug(f"Using legacy API key authentication")
        
        return headers
    
    def _handle_response_error(
        self,
        error: Exception,
        operation: str = "API request"
    ) -> Dict[str, Any]:
        """
        Handle common HTTP errors and convert to standardized response.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation for logging
            
        Returns:
            Standardized error response dictionary
        """
        if isinstance(error, httpx.HTTPStatusError):
            error_code = f"HTTP_{error.response.status_code}"
            error_description = f"HTTP {error.response.status_code}: {error.response.text}"
            status_code = error.response.status_code
            
            logger.error(
                f"HTTP error during {operation} - "
                f"Code: {error_code}, Status: {status_code}, "
                f"Response: {error.response.text}"
            )
            
            return {
                "success": False,
                "error_code": error_code,
                "error_description": error_description,
                "status_code": status_code
            }
        
        elif isinstance(error, httpx.TimeoutException):
            error_code = "TIMEOUT_ERROR"
            error_description = f"Request timeout after {self.timeout}s: {str(error)}"
            
            logger.error(f"Timeout error during {operation} - {error_description}")
            
            return {
                "success": False,
                "error_code": error_code,
                "error_description": error_description
            }
        
        elif isinstance(error, httpx.RequestError):
            error_code = "REQUEST_ERROR"
            error_description = f"Network/connection error: {str(error)}"
            
            logger.error(f"Request error during {operation} - {error_description}")
            
            return {
                "success": False,
                "error_code": error_code,
                "error_description": error_description
            }
        
        else:
            error_code = "UNEXPECTED_ERROR"
            error_description = f"Unexpected error: {type(error).__name__} - {str(error)}"
            
            logger.error(f"Unexpected error during {operation} - {error_description}", exc_info=True)
            
            return {
                "success": False,
                "error_code": error_code,
                "error_description": error_description
            }
    
    def _create_client(self, timeout: Optional[float] = None) -> httpx.Client:
        """
        Create an httpx.Client with default settings.
        
        Args:
            timeout: Optional timeout override, uses self.timeout if not provided
            
        Returns:
            Configured httpx.Client instance
        """
        return httpx.Client(
            timeout=timeout or self.timeout,
            verify=self.verify_ssl
        )
    
    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from endpoint.
        
        Args:
            endpoint: API endpoint (with or without leading slash)
        
        Returns:
            Full URL
        """
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"
    
    def _extract_error_message(self, response: httpx.Response) -> str:
        """
        Extract error message from response.
        
        Args:
            response: Response object
        
        Returns:
            Error message string
        """
        try:
            error_data = response.json()
            # Try common error message fields
            if isinstance(error_data, dict):
                return (error_data.get('message') or 
                       error_data.get('error') or 
                       error_data.get('detail') or 
                       str(error_data))
            return str(error_data)
        except Exception:
            return response.text or f"HTTP {response.status_code}"
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        expect_json: bool = True
    ) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            headers: Optional additional headers
            timeout: Optional timeout override
            expect_json: Whether to parse response as JSON (default: True)
        
        Returns:
            Tuple of (success: bool, data: Any or None, error_message: str or None)
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        logger.info(f"API Request: GET {url}")
        if params:
            logger.debug(f"Query params: {params}")
        
        try:
            with self._create_client(timeout) as client:
                response = client.get(url, params=params, headers=request_headers)
                response.raise_for_status()
                
                logger.debug(f"API Response: {response.status_code} from {response.url}")
                
                if expect_json:
                    return True, response.json(), None
                else:
                    return True, response.content, None
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"[{self.service_name}] HTTP {e.response.status_code}: {self._extract_error_message(e.response)}"
            logger.error(f"API error: {error_msg}")
            return False, None, error_msg
        except httpx.TimeoutException as e:
            error_msg = f"[{self.service_name}] Request timed out after {timeout or self.timeout}s: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except httpx.RequestError as e:
            error_msg = f"[{self.service_name}] Connection error to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"[{self.service_name}] Unexpected error during request to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def post(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        expect_json: bool = True
    ) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            json_data: Optional JSON payload
            data: Optional form data
            files: Optional files to upload (dict of {field_name: file_data})
            headers: Optional additional headers
            timeout: Optional timeout override
            expect_json: Whether to parse response as JSON (default: True)
        
        Returns:
            Tuple of (success: bool, data: Any or None, error_message: str or None)
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        # Remove Content-Type header if uploading files (httpx will set it)
        if files:
            request_headers.pop('Content-Type', None)
        
        logger.info(f"API Request: POST {url}")
        if params:
            logger.debug(f"Query params: {params}")
        if json_data:
            logger.debug(f"JSON payload: {json_data}")
        
        try:
            with self._create_client(timeout) as client:
                response = client.post(
                    url,
                    params=params,
                    json=json_data,
                    data=data,
                    files=files,
                    headers=request_headers
                )
                response.raise_for_status()
                
                logger.debug(f"API Response: {response.status_code} from {response.url}")
                
                if expect_json:
                    return True, response.json(), None
                else:
                    return True, response.content, None
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"[{self.service_name}] HTTP {e.response.status_code}: {self._extract_error_message(e.response)}"
            logger.error(f"API error: {error_msg}")
            return False, None, error_msg
        except httpx.TimeoutException as e:
            error_msg = f"[{self.service_name}] Request timed out after {timeout or self.timeout}s: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except httpx.RequestError as e:
            error_msg = f"[{self.service_name}] Connection error to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"[{self.service_name}] Unexpected error during request to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def put(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        expect_json: bool = True
    ) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            json_data: Optional JSON payload
            headers: Optional additional headers
            timeout: Optional timeout override
            expect_json: Whether to parse response as JSON (default: True)
        
        Returns:
            Tuple of (success: bool, data: Any or None, error_message: str or None)
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        logger.info(f"API Request: PUT {url}")
        if params:
            logger.debug(f"Query params: {params}")
        if json_data:
            logger.debug(f"JSON payload: {json_data}")
        
        try:
            with self._create_client(timeout) as client:
                response = client.put(
                    url,
                    params=params,
                    json=json_data,
                    headers=request_headers
                )
                response.raise_for_status()
                
                logger.debug(f"API Response: {response.status_code} from {response.url}")
                
                if expect_json:
                    return True, response.json(), None
                else:
                    return True, response.content, None
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"[{self.service_name}] HTTP {e.response.status_code}: {self._extract_error_message(e.response)}"
            logger.error(f"API error: {error_msg}")
            return False, None, error_msg
        except httpx.TimeoutException as e:
            error_msg = f"[{self.service_name}] Request timed out after {timeout or self.timeout}s: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except httpx.RequestError as e:
            error_msg = f"[{self.service_name}] Connection error to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"[{self.service_name}] Unexpected error during request to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        expect_json: bool = True
    ) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            json_data: Optional JSON payload
            headers: Optional additional headers
            timeout: Optional timeout override
            expect_json: Whether to parse response as JSON (default: True)
        
        Returns:
            Tuple of (success: bool, data: Any or None, error_message: str or None)
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        logger.info(f"API Request: DELETE {url}")
        if params:
            logger.debug(f"Query params: {params}")
        if json_data:
            logger.debug(f"JSON payload: {json_data}")
        
        try:
            with self._create_client(timeout) as client:
                response = client.delete(
                    url,
                    params=params,
                    json=json_data,
                    headers=request_headers
                )
                response.raise_for_status()
                
                logger.debug(f"API Response: {response.status_code} from {response.url}")
                
                if expect_json:
                    return True, response.json(), None
                else:
                    return True, response.content, None
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"[{self.service_name}] HTTP {e.response.status_code}: {self._extract_error_message(e.response)}"
            logger.error(f"API error: {error_msg}")
            return False, None, error_msg
        except httpx.TimeoutException as e:
            error_msg = f"[{self.service_name}] Request timed out after {timeout or self.timeout}s: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except httpx.RequestError as e:
            error_msg = f"[{self.service_name}] Connection error to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"[{self.service_name}] Unexpected error during request to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def download_file(
        self,
        endpoint: str,
        output_path: Union[str, Path],
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        chunk_size: int = 8192
    ) -> Tuple[bool, Optional[str]]:
        """
        Download a file from the API and save it to disk.
        
        Args:
            endpoint: API endpoint
            output_path: Path where to save the downloaded file
            params: Optional query parameters
            headers: Optional additional headers
            timeout: Optional timeout override
            chunk_size: Size of chunks to read when streaming (default: 8192 bytes)
        
        Returns:
            Tuple of (success: bool, error_message: str or None)
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        logger.info(f"API Request: GET {url} (downloading to {output_path})")
        if params:
            logger.debug(f"Query params: {params}")
        
        try:
            with self._create_client(timeout) as client:
                with client.stream("GET", url, params=params, headers=request_headers) as response:
                    response.raise_for_status()
                    
                    # Save the file
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                
                logger.info(f"Successfully downloaded file to {output_path}")
                return True, None
                
        except httpx.HTTPStatusError as e:
            error_msg = f"[{self.service_name}] HTTP {e.response.status_code}: {self._extract_error_message(e.response)}"
            logger.error(f"API error: {error_msg}")
            return False, error_msg
        except httpx.TimeoutException as e:
            error_msg = f"[{self.service_name}] Request timed out after {timeout or self.timeout}s: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except httpx.RequestError as e:
            error_msg = f"[{self.service_name}] Connection error to {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"[{self.service_name}] Error downloading file from {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def close(self):
        """Close any resources. For compatibility with context manager pattern."""
        # httpx.Client is already closed by context manager in each method
        # This method exists for compatibility with requests.Session pattern
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
