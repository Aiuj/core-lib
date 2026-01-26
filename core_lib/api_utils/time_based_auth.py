"""Time-based HMAC authentication for secure inter-application communication.

This module provides a simple yet secure authentication mechanism that generates
time-based keys valid for a 3-hour window (previous hour, current hour, next hour)
to prevent disruption during hour transitions.

The authentication uses HMAC-SHA256 with a private key to generate public keys
that are only valid for specific time windows.

Example:
    Server side:
        ```python
        from core_lib.api_utils import verify_time_key
        from core_lib.config import AuthSettings
        
        settings = AuthSettings.from_env()
        
        # In your FastAPI/MCP endpoint
        if not verify_time_key(client_key, settings.auth_private_key):
            raise HTTPException(status_code=401, detail="Invalid authentication key")
        ```
    
    Client side:
        ```python
        from core_lib.api_utils import generate_time_key
        from core_lib.config import AuthSettings
        
        settings = AuthSettings.from_env()
        auth_key = generate_time_key(settings.auth_private_key)
        
        # Add to request headers
        headers = {"x-auth-key": auth_key}
        ```
"""

import hmac
import hashlib
from datetime import datetime, timezone
from typing import Iterable, Optional, Union


class TimeBasedAuthError(Exception):
    """Raised when time-based authentication fails."""
    pass


def _get_time_window_keys(dt: Optional[datetime] = None) -> list[str]:
    """Get hour-based time window identifiers for the given datetime.
    
    Returns identifiers for previous hour, current hour, and next hour
    to ensure valid keys during hour transitions.
    
    Args:
        dt: Datetime to get windows for. If None, uses current UTC time.
        
    Returns:
        List of 3 time window strings in format "YYYY-MM-DD-HH"
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # Get current hour timestamp
    current_hour = dt.replace(minute=0, second=0, microsecond=0)
    
    # Generate time windows for previous, current, and next hour
    windows = []
    for hour_offset in [-1, 0, 1]:
        from datetime import timedelta
        window_time = current_hour + timedelta(hours=hour_offset)
        # Format as YYYY-MM-DD-HH for hour-based windows
        window_key = window_time.strftime("%Y-%m-%d-%H")
        windows.append(window_key)
    
    return windows


def generate_time_key(
    private_key: str,
    dt: Optional[datetime] = None,
    encoding: str = "utf-8"
) -> str:
    """Generate a time-based authentication key using HMAC.
    
    Creates a public authentication key valid for the current hour using
    HMAC-SHA256 with the provided private key. The key will be accepted
    during the previous, current, and next hour windows.
    
    Args:
        private_key: Secret private key shared between client and server
        dt: Datetime to generate key for. If None, uses current UTC time.
        encoding: String encoding to use (default: utf-8)
        
    Returns:
        Hex-encoded HMAC authentication key
        
    Raises:
        TimeBasedAuthError: If private_key is empty or invalid
        
    Example:
        ```python
        private_key = "my-secret-key-12345"
        auth_key = generate_time_key(private_key)
        # Use auth_key in request headers
        ```
    """
    if not private_key or not private_key.strip():
        raise TimeBasedAuthError("Private key cannot be empty")
    
    # Get current time window
    windows = _get_time_window_keys(dt)
    current_window = windows[1]  # Use current hour (middle window)
    
    # Generate HMAC using private key and time window
    key_bytes = private_key.encode(encoding)
    message_bytes = current_window.encode(encoding)
    
    hmac_obj = hmac.new(key_bytes, message_bytes, hashlib.sha256)
    return hmac_obj.hexdigest()


def verify_time_key(
    provided_key: str,
    private_key: Union[str, Iterable[str]],
    dt: Optional[datetime] = None,
    encoding: str = "utf-8"
) -> bool:
    """Verify a time-based authentication key.
    
    Checks if the provided key matches any of the valid time windows
    (previous hour, current hour, next hour) to handle hour transitions
    gracefully. Supports multiple private keys for key rotation.
    
    Args:
        provided_key: The authentication key to verify
        private_key: Secret private key(s) - can be a single string or iterable of strings
        dt: Datetime to verify against. If None, uses current UTC time.
        encoding: String encoding to use (default: utf-8)
        
    Returns:
        True if the key is valid for any of the time windows with any private key
        
    Raises:
        TimeBasedAuthError: If no private keys are provided
        
    Example:
        ```python
        # Single key
        private_key = "my-secret-key-12345"
        if verify_time_key(client_key, private_key):
            proceed_with_request()
        
        # Multiple keys (for rotation)
        private_keys = ["current-key", "old-key-still-valid"]
        if verify_time_key(client_key, private_keys):
            proceed_with_request()
        ```
    """
    # Handle single key or iterable of keys
    if isinstance(private_key, str):
        private_keys = [private_key]
    else:
        private_keys = list(private_key)
    
    if not private_keys:
        raise TimeBasedAuthError("At least one private key must be provided")
    
    # Filter out empty keys
    private_keys = [k for k in private_keys if k and k.strip()]
    if not private_keys:
        raise TimeBasedAuthError("Private key cannot be empty")
    
    if not provided_key or not provided_key.strip():
        return False
    
    # Get all valid time windows (previous, current, next hour)
    windows = _get_time_window_keys(dt)
    
    # Check if provided key matches any combination of private key and time window
    for pk in private_keys:
        key_bytes = pk.encode(encoding)
        
        for window in windows:
            message_bytes = window.encode(encoding)
            hmac_obj = hmac.new(key_bytes, message_bytes, hashlib.sha256)
            expected_key = hmac_obj.hexdigest()
            
            # Use constant-time comparison to prevent timing attacks
            if hmac.compare_digest(provided_key, expected_key):
                return True
    
    return False
