# Settings Management System

The Core Library provides a comprehensive, extensible settings management system that handles configuration for any application. The system supports environment variables, .env files, type conversion, validation, and easy extension for custom settings.

## Overview

The settings system is built around several key components:

- **`StandardSettings`**: Main settings class with all standard services (LLM, cache, database, etc.) and easy extension via `extend_from_env()`
- **Service-specific modules**: Dedicated classes for each service type (LLM, embeddings, cache, tracing, database, MCP server)
- **`BaseSettings`**: Abstract base class for advanced custom settings (when you need full control)
- **`SettingsManager`**: Manages multiple settings instances with validation
- **Utilities**: Helper classes for .env loading and environment variable parsing

> **💡 Quick Start**: For most applications, use `StandardSettings.extend_from_env()` to add custom configuration. This gives you all standard services plus your custom fields with minimal code.

### Modular Architecture

The settings system uses a modular architecture with dedicated files for each service type:

```
core_lib/config/
├── base_settings.py       # Core framework and utilities
├── app_settings.py        # Core application settings  
├── llm_settings.py        # LLM provider configuration
├── embeddings_settings.py # Embeddings provider configuration
├── cache_settings.py      # Cache provider configuration
├── tracing_settings.py    # Tracing provider configuration
├── database_settings.py   # Database configuration
├── mcp_settings.py        # MCP server configuration
├── fastapi_settings.py    # FastAPI HTTP server configuration
└── standard_settings.py   # Main unified settings class
```

This modular design provides:
- **Clear separation of concerns** - each service has its own module
- **Easy maintenance** - changes to one service don't affect others
- **Extensibility** - new services can be added as separate modules
- **Testability** - each module can be tested independently

## Quick Start

### Basic Usage

```python
from core_lib.config import StandardSettings

# Load all settings from environment variables and .env files
settings = StandardSettings.from_env()

print(f"App: {settings.app_name} v{settings.version}")
print(f"Environment: {settings.environment}")

# Access service configurations if enabled
if settings.llm:
    print(f"LLM Provider: {settings.llm.provider}")
    print(f"LLM Model: {settings.llm.model}")

if settings.cache:
    print(f"Cache: {settings.cache.provider}://{settings.cache.host}:{settings.cache.port}")

if settings.database:
    print(f"Database: {settings.database.username}@{settings.database.host}:{settings.database.port}/{settings.database.database}")
```

### Using .env Files

Create a `.env` file in your project root:

```bash
# Core app settings
APP_NAME=my-awesome-app
ENVIRONMENT=production
LOG_LEVEL=INFO

# LLM configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# Cache configuration
ENABLE_CACHE=true
REDIS_HOST=redis.example.com
REDIS_PASSWORD=secret123
REDIS_PREFIX=myapp:

# Tracing configuration
ENABLE_TRACING=true
LANGFUSE_PUBLIC_KEY=pk_your_public_key
LANGFUSE_SECRET_KEY=sk_your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# MCP server configuration
MCP_SERVER_NAME=my-app-server
MCP_SERVER_PORT=8204
MCP_SERVER_HOST=0.0.0.0
MCP_TRANSPORT=streamable-http
```

The settings system will automatically discover and load `.env` files from:
1. Current working directory
2. Project root (directory containing `pyproject.toml`)
3. User home directory

### Service Auto-Detection

The system automatically detects which services to enable based on environment variables:

```python
# Will automatically enable LLM if any LLM-related env vars are present
# Will automatically enable cache if Redis/Valkey env vars are present
# Will automatically enable tracing if Langfuse env vars are present
settings = StandardSettings.from_env()

# Or explicitly control which services to enable
settings = StandardSettings.from_env(
    enable_llm=True,
    enable_embeddings=False,
    enable_cache=True,
    enable_tracing=False
)

# Null-safe accessors for convenience
# Access sub-config attributes without checking None
print(settings.mcp_server_safe.transport)     # str or None
print(settings.fastapi_server_safe.url)       # str or None
print(settings.llm_safe.model)                # str or None
```

### Adding Custom Configuration

For application-specific settings, extend StandardSettings using the simplified `extend_from_env()` pattern:

```python
class MyAppSettings:
    @classmethod
    def from_env(cls, **kwargs):
        return StandardSettings.extend_from_env(
            custom_config={
                "api_key": {"env_vars": ["API_KEY"], "required": True},
                "debug_mode": {"env_vars": ["DEBUG"], "default": False, "env_type": bool},
                "max_workers": {"env_vars": ["MAX_WORKERS"], "default": 4, "env_type": int},
            },
            **kwargs
        )

# Get all standard services + custom configuration
settings = MyAppSettings.from_env()
print(f"API Key: {settings.api_key}")
print(f"LLM enabled: {settings.enable_llm}")
```

## Individual Settings Classes

### LLM Settings

```python
from core_lib.config import LLMSettings

# Auto-detect provider from environment
llm_settings = LLMSettings.from_env()

# Or specify explicitly
llm_settings = LLMSettings.from_env(
    provider="openai",
    model="gpt-4",
    temperature=0.3
)

# Get configuration for existing LLM clients
llm_config = settings.get_llm_config()  # Returns OpenAIConfig/GeminiConfig/OllamaConfig
```

### Embeddings Settings

```python
from core_lib.config import EmbeddingsSettings

embeddings_settings = EmbeddingsSettings.from_env(
    provider="openai",
    model="text-embedding-3-large",
    embedding_dimension=1536
)

# Get configuration (returns EmbeddingsSettings, also available as EmbeddingsConfig for compatibility)
embeddings_config = settings.get_embeddings_config()
```

### Cache Settings  

```python
from core_lib.config import CacheSettings

cache_settings = CacheSettings.from_env(
    provider="redis",  # or "valkey"
    host="localhost",
    port=6379,
    password="secret"
)

# Get configuration for existing cache client
redis_config = settings.get_redis_config()  # Returns RedisConfig
```

### Database Settings

```python
from core_lib.config import DatabaseSettings

database_settings = DatabaseSettings.from_env(
    host="localhost",
    port=5432,
    database="myapp",
    username="user",
    password="password",
    sslmode="require"
)

# Generate connection strings for different drivers
sync_conn = database_settings.get_sync_connection_string()      # postgresql+psycopg2://
async_conn = database_settings.get_async_connection_string()    # postgresql+asyncpg://
basic_conn = database_settings.get_connection_string()          # postgresql://

# Use with StandardSettings
settings = StandardSettings.from_env()
if settings.database:
    db_config = settings.get_database_config()  # Returns DatabaseSettings
```

### MCP Server Settings

```python
from core_lib.config import MCPServerSettings

# Configure MCP (Model Context Protocol) server
mcp_settings = MCPServerSettings.from_env(
    server_name="my-app-server",
    version="1.0.0",
    host="0.0.0.0",
    port=8204,
    transport="streamable-http",
    timeout=30
)

# Get server information for MCP protocol
server_info = mcp_settings.get_server_info()  # {"name": "...", "version": "..."}

# Get connection configuration for MCP clients
connection_config = mcp_settings.get_connection_config()

# Use with StandardSettings
settings = StandardSettings.from_env()
if settings.mcp_server:
    mcp_config = settings.get_mcp_server_config()  # Returns MCPServerSettings
```

### FastAPI Server Settings

```python
from core_lib.config import FastAPIServerSettings

# Configure optional FastAPI HTTP server
fastapi_settings = FastAPIServerSettings.from_env(
    host="0.0.0.0",
    port=8096,
    reload=False,
    api_auth_enabled=False,
    api_key_header_name="x-api-key",
)

# Use with StandardSettings
settings = StandardSettings.from_env()
if settings.fastapi_server:
    api_cfg = settings.get_fastapi_server_config()  # Returns FastAPIServerSettings
    print(api_cfg.url)
```

### Tracing Settings

```python
from core_lib.config import TracingSettings

tracing_settings = TracingSettings.from_env(
    service_name="my-service",
    langfuse_public_key="pk_...",
    langfuse_secret_key="sk_..."
)
```

## Creating Custom Settings

The easiest way to add custom configuration to your application is to extend `StandardSettings` using the `extend_from_env()` method. This approach gives you all the standard services (LLM, cache, database, etc.) plus your custom fields with minimal code.

### Recommended Approach: Extending StandardSettings

```python
from core_lib.config import StandardSettings
from typing import Optional

class MyAppSettings:
    """Custom settings that extend StandardSettings with app-specific configuration."""
    
    @classmethod
    def from_env(cls, **kwargs):
        """Load settings from environment with custom field mappings."""
        return StandardSettings.extend_from_env(
            custom_config={
                "api_key": {
                    "env_vars": ["API_KEY", "MY_API_KEY"], 
                    "required": True
                },
                "api_timeout": {
                    "env_vars": ["API_TIMEOUT", "TIMEOUT"], 
                    "default": 30, 
                    "env_type": int
                },
                "debug_mode": {
                    "env_vars": ["DEBUG_MODE", "DEBUG"], 
                    "default": False, 
                    "env_type": bool
                },
                "max_workers": {
                    "env_vars": ["MAX_WORKERS", "WORKERS"], 
                    "default": 4, 
                    "env_type": int
                },
                "feature_flags": {
                    "env_vars": ["FEATURE_FLAGS"], 
                    "default": [], 
                    "env_type": list
                },
                "allowed_hosts": {
                    "env_vars": ["ALLOWED_HOSTS"], 
                    "default": ["localhost"], 
                    "env_type": list
                },
            },
            **kwargs
        )

# Usage
settings = MyAppSettings.from_env()

# Access standard services
if settings.enable_llm:
    llm_config = settings.get_llm_config()
    print(f"LLM Provider: {llm_config.provider}")

if settings.enable_cache:
    redis_config = settings.get_redis_config()
    print(f"Cache Host: {redis_config.host}")

# Access custom fields
print(f"API Key: {settings.api_key[:8]}...")
print(f"Debug Mode: {settings.debug_mode}")
print(f"Max Workers: {settings.max_workers}")
print(f"Feature Flags: {settings.feature_flags}")
```

### Custom Configuration Options

Each custom field supports these configuration options:

- **`env_vars`**: List of environment variable names to check (in order of preference)
- **`default`**: Default value if no environment variable is found
- **`env_type`**: Type conversion (str, int, float, bool, list)
- **`required`**: Whether the field is required (raises error if missing)

### Environment Variables for Custom Settings

```bash
# Required API key
API_KEY=your-secret-key

# Optional configuration with defaults
API_TIMEOUT=60
DEBUG_MODE=true
MAX_WORKERS=8
FEATURE_FLAGS=feature_a,feature_b,feature_c
ALLOWED_HOSTS=localhost,127.0.0.1,example.com

# Standard services still work
OPENAI_API_KEY=sk-your-key
REDIS_HOST=localhost
```

### Benefits of This Approach

✅ **Get all standard services automatically** (LLM, cache, database, tracing, MCP server)  
✅ **Minimal boilerplate code** - just define your custom fields  
✅ **Full type safety and validation**  
✅ **Environment variable parsing with fallbacks**  
✅ **Compatible with existing StandardSettings APIs**  
✅ **Easy to test and maintain**

### Custom Initialization and Validation

For scenarios where you need custom initialization logic, validation, or post-processing, the cleanest approach is to inherit directly from `StandardSettings`:

```python
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from core_lib.config import StandardSettings, SettingsError, EnvParser

@dataclass(frozen=True)
class MyAppSettings(StandardSettings):
    """Custom settings that extend StandardSettings with app-specific configuration."""
    
    # Custom fields
    api_key: Optional[str] = None
    api_timeout: int = 30
    debug_mode: bool = False
    max_workers: int = 4
    feature_flags: List[str] = field(default_factory=list)
    temp_dir: str = "/tmp/myapp"
    output_format: str = "json"
    
    # Computed fields (will be set after initialization)
    computed_output_dir: Optional[str] = field(default=None)
    
    @classmethod
    def from_env(cls, load_dotenv=True, dotenv_paths=None, **overrides):
        """Load settings from environment with custom fields and logic."""
        # Load dotenv files
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # First get standard settings using parent class logic
        standard_settings = super().from_env(load_dotenv=False, **overrides)
        
        # Parse our custom environment variables
        custom_fields = {
            "api_key": EnvParser.get_env("API_KEY", "MY_API_KEY", required=True),
            "api_timeout": EnvParser.get_env("API_TIMEOUT", "TIMEOUT", default=30, env_type=int),
            "debug_mode": EnvParser.get_env("DEBUG_MODE", "DEBUG", default=False, env_type=bool),
            "max_workers": EnvParser.get_env("MAX_WORKERS", "WORKERS", default=4, env_type=int),
            "feature_flags": EnvParser.get_env("FEATURE_FLAGS", default=[], env_type=list),
            "temp_dir": EnvParser.get_env("TEMP_DIR", "TMP_DIR", default="/tmp/myapp"),
            "output_format": EnvParser.get_env("OUTPUT_FORMAT", default="json"),
        }
        
        # Create the instance with both standard and custom fields
        instance_data = {
            # Standard fields
            **standard_settings.as_dict(),
            # Custom fields
            **custom_fields,
            # Overrides
            **overrides
        }
        
        # Create instance
        instance = cls(**instance_data)
        
        # Perform custom initialization and return modified instance
        return instance._post_init_setup()
    
    def _post_init_setup(self):
        """Custom initialization logic after loading settings."""
        updates = {}
        
        # Ensure temp directory exists and get absolute path
        temp_path = Path(self.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        updates['temp_dir'] = str(temp_path.absolute())
        
        # Set up output directory based on environment
        if self.environment == 'production':
            output_dir = Path('/var/log/myapp')
        else:
            output_dir = Path('./logs')
        output_dir.mkdir(parents=True, exist_ok=True)
        updates['computed_output_dir'] = str(output_dir)
        
        # Adjust worker count based on available CPUs
        import os
        cpu_count = os.cpu_count() or 1
        adjusted_workers = min(self.max_workers, cpu_count * 2)
        updates['max_workers'] = adjusted_workers
        
        # Update log level if debug mode
        if self.debug_mode:
            updates['log_level'] = 'DEBUG'
        
        # Validate settings
        self._validate_custom_settings()
        
        # Create new instance with updates (since dataclass is frozen)
        if updates:
            # Get current values and apply updates
            current_dict = {
                field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()
                if field.init
            }
            current_dict.update(updates)
            return type(self)(**current_dict)
        
        return self
    
    def _validate_custom_settings(self):
        """Custom validation logic."""
        # Validate API configuration
        if self.api_timeout <= 0:
            raise SettingsError("API timeout must be positive")
        
        # Validate worker configuration
        if self.max_workers <= 0:
            raise SettingsError("Max workers must be positive")
        
        # Validate feature flags
        allowed_flags = ['feature_a', 'feature_b', 'feature_c', 'beta_features']
        invalid_flags = [flag for flag in self.feature_flags if flag not in allowed_flags]
        if invalid_flags:
            raise SettingsError(f"Invalid feature flags: {invalid_flags}")
        
        # Validate output format
        valid_formats = ['json', 'yaml', 'xml']
        if self.output_format not in valid_formats:
            raise SettingsError(f"Output format must be one of: {valid_formats}")

# Usage - Clean and simple!
settings = MyAppSettings.from_env()

# All standard services work perfectly
if settings.enable_llm:
    llm_config = settings.get_llm_config()
    print(f"LLM Provider: {llm_config.provider}")

if settings.enable_cache:
    redis_config = settings.get_redis_config()
    print(f"Redis Host: {redis_config.host}")

# Custom fields with post-initialization
print(f"API Key: {settings.api_key[:8]}...")
print(f"Temp directory: {settings.temp_dir}")  # Absolute path, guaranteed to exist
print(f"Output directory: {settings.computed_output_dir}")  # Created based on environment
print(f"Adjusted workers: {settings.max_workers}")  # Adjusted for CPU count
print(f"Debug mode: {settings.debug_mode}")
print(f"Feature flags: {settings.feature_flags}")
```

### Simplified Version (No Post-Processing)

If you don't need complex initialization logic, you can make it even simpler:

```python
from dataclasses import dataclass
from typing import Optional, List
from core_lib.config import StandardSettings, EnvParser

@dataclass(frozen=True)
class SimpleAppSettings(StandardSettings):
    """Simple custom settings without complex post-processing."""
    
    api_key: Optional[str] = None
    api_timeout: int = 30
    debug_mode: bool = False
    max_workers: int = 4
    
    @classmethod
    def from_env(cls, load_dotenv=True, dotenv_paths=None, **overrides):
        """Load settings from environment."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Get standard settings
        standard_settings = super().from_env(load_dotenv=False, **overrides)
        
        # Parse custom fields
        custom_fields = {
            "api_key": EnvParser.get_env("API_KEY", required=True),
            "api_timeout": EnvParser.get_env("API_TIMEOUT", default=30, env_type=int),
            "debug_mode": EnvParser.get_env("DEBUG_MODE", default=False, env_type=bool),
            "max_workers": EnvParser.get_env("MAX_WORKERS", default=4, env_type=int),
        }
        
        # Combine and create instance
        return cls(**{**standard_settings.as_dict(), **custom_fields, **overrides})

# Usage
settings = SimpleAppSettings.from_env()
print(f"API Key: {settings.api_key}")
print(f"LLM Enabled: {settings.enable_llm}")
```

### Testing Your Custom Settings

When writing unit tests, you want to ensure tests are isolated and don't depend on external `.env` files or environment variables. Here are several strategies:

#### Strategy 1: Disable .env Loading in Tests

```python
import unittest
import os
from unittest.mock import patch
from pathlib import Path

class TestMyAppSettings(unittest.TestCase):
    
    def setUp(self):
        """Clear environment before each test."""
        # Store original environment
        self.original_env = dict(os.environ)
        
        # Clear relevant environment variables
        test_vars = [
            'API_KEY', 'API_TIMEOUT', 'DEBUG_MODE', 'MAX_WORKERS',
            'OPENAI_API_KEY', 'REDIS_HOST', 'TEMP_DIR', 'OUTPUT_FORMAT'
        ]
        for var in test_vars:
            os.environ.pop(var, None)
    
    def tearDown(self):
        """Restore original environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_settings_with_defaults_no_dotenv(self):
        """Test settings using only defaults, no .env files."""
        # Set only required variables
        os.environ['API_KEY'] = 'test-key'
        
        # Disable .env loading and provide overrides
        settings = MyAppSettings.from_env(
            load_dotenv=False,  # Disable .env file loading
            # Provide defaults for testing
            temp_dir='/tmp/test',
            enable_llm=False,   # Disable services for faster tests
            enable_cache=False,
            enable_tracing=False
        )
        
        self.assertEqual(settings.api_key, 'test-key')
        self.assertEqual(settings.api_timeout, 30)  # Default value
        self.assertFalse(settings.debug_mode)       # Default value
        self.assertEqual(settings.max_workers, 4)   # Default value
        self.assertFalse(settings.enable_llm)       # Disabled for test
    
    def test_settings_with_mocked_env(self):
        """Test settings with specific environment variables."""
        test_env = {
            'API_KEY': 'test-api-key',
            'API_TIMEOUT': '60',
            'DEBUG_MODE': 'true',
            'MAX_WORKERS': '8',
            'TEMP_DIR': '/tmp/test_app'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            settings = MyAppSettings.from_env(
                load_dotenv=False,  # Don't load .env files
                enable_llm=False,   # Keep tests fast
                enable_cache=False
            )
            
            self.assertEqual(settings.api_key, 'test-api-key')
            self.assertEqual(settings.api_timeout, 60)
            self.assertTrue(settings.debug_mode)
            self.assertEqual(settings.max_workers, 8)
    
    def test_settings_direct_instantiation(self):
        """Test settings by direct instantiation (bypassing environment)."""
        settings = MyAppSettings(
            # Standard fields
            app_name='test-app',
            version='1.0.0',
            environment='test',
            log_level='DEBUG',
            project_root=None,
            # Service configurations (None = disabled)
            llm=None,
            embeddings=None,
            cache=None,
            tracing=None,
            database=None,
            mcp_server=None,
            # Service flags
            enable_llm=False,
            enable_embeddings=False,
            enable_cache=False,
            enable_tracing=False,
            enable_database=False,
            enable_mcp_server=False,
            # Custom fields
            api_key='test-key',
            api_timeout=30,
            debug_mode=False,
            max_workers=4,
            feature_flags=['test_feature'],
            temp_dir='/tmp/test',
            output_format='json',
            computed_output_dir=None
        )
        
        self.assertEqual(settings.api_key, 'test-key')
        self.assertEqual(settings.app_name, 'test-app')
        self.assertFalse(settings.enable_llm)
```

#### Strategy 2: Test-Specific Settings Class

```python
from dataclasses import dataclass
from typing import Optional
from core_lib.config import StandardSettings, EnvParser

@dataclass(frozen=True)
class TestAppSettings(StandardSettings):
    """Test-specific settings that don't require external dependencies."""
    
    api_key: Optional[str] = "test-default-key"
    api_timeout: int = 30
    debug_mode: bool = False
    max_workers: int = 2  # Lower for tests
    
    @classmethod
    def from_env(cls, load_dotenv=False, **overrides):  # Default to no .env
        """Test-friendly settings loader."""
        # Set test-friendly defaults
        test_defaults = {
            'enable_llm': False,
            'enable_cache': False,
            'enable_tracing': False,
            'enable_database': False,
            'enable_mcp_server': False,
            'environment': 'test',
            'log_level': 'DEBUG',
            'temp_dir': '/tmp/test_app',
            **overrides
        }
        
        # Only load .env if explicitly requested
        return super().from_env(load_dotenv=load_dotenv, **test_defaults)

# In your tests
def test_app_functionality():
    settings = TestAppSettings.from_env()  # No .env loading, test defaults
    assert settings.api_key == "test-default-key"
    assert settings.environment == "test"
    assert not settings.enable_llm  # Services disabled
```

#### Strategy 3: Environment Variable Mocking with Pytest

```python
import pytest
import os
from unittest.mock import patch

@pytest.fixture
def clean_env():
    """Fixture that provides a clean environment for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield

@pytest.fixture
def test_env():
    """Fixture that provides a controlled test environment."""
    test_vars = {
        'API_KEY': 'test-key',
        'API_TIMEOUT': '30',
        'DEBUG_MODE': 'false',
        'MAX_WORKERS': '2',
        'ENVIRONMENT': 'test'
    }
    with patch.dict(os.environ, test_vars, clear=True):
        yield

def test_settings_with_clean_env(clean_env):
    """Test with completely clean environment."""
    # This will use only defaults and overrides
    settings = MyAppSettings.from_env(
        load_dotenv=False,
        api_key='test-override',  # Direct override
        enable_llm=False
    )
    assert settings.api_key == 'test-override'

def test_settings_with_test_env(test_env):
    """Test with controlled environment variables."""
    settings = MyAppSettings.from_env(
        load_dotenv=False,  # Don't load .env files
        enable_llm=False    # Keep tests fast
    )
    assert settings.api_key == 'test-key'
    assert settings.environment == 'test'
```

#### Strategy 4: Configuration Factory for Tests

```python
class SettingsFactory:
    """Factory for creating test settings configurations."""
    
    @staticmethod
    def create_test_settings(**overrides):
        """Create settings optimized for testing."""
        defaults = {
            # App defaults
            'app_name': 'test-app',
            'version': '1.0.0-test',
            'environment': 'test',
            'log_level': 'DEBUG',
            
            # Disable external services
            'enable_llm': False,
            'enable_cache': False,
            'enable_tracing': False,
            'enable_database': False,
            'enable_mcp_server': False,
            
            # Test-friendly values
            'api_key': 'test-key',
            'api_timeout': 10,  # Fast timeouts
            'debug_mode': True,
            'max_workers': 1,   # Single worker for deterministic tests
            'temp_dir': '/tmp/test',
            'output_format': 'json',
            
            # Override any custom values
            **overrides
        }
        
        return MyAppSettings.from_env(load_dotenv=False, **defaults)
    
    @staticmethod
    def create_integration_test_settings(**overrides):
        """Create settings for integration tests (with some services enabled)."""
        return SettingsFactory.create_test_settings(
            enable_cache=True,  # Enable cache for integration tests
            **overrides
        )

# Usage in tests
def test_business_logic():
    settings = SettingsFactory.create_test_settings()
    # Your test logic here
    
def test_with_cache_integration():
    settings = SettingsFactory.create_integration_test_settings(
        redis_host='localhost:6379'
    )
    # Integration test logic here
```

### Best Practices for Testing Settings

✅ **Always disable .env loading** in unit tests (`load_dotenv=False`)  
✅ **Use environment mocking** for controlled test environments  
✅ **Disable external services** by default (LLM, cache, database)  
✅ **Use direct instantiation** when you need full control  
✅ **Create test-specific settings classes** for complex scenarios  
✅ **Use fixtures** to ensure test isolation  
✅ **Mock external dependencies** rather than using real services

### Pytest Integration

For pytest users, here's a complete testing setup:

```python
# conftest.py
import pytest
import os
from unittest.mock import patch

@pytest.fixture
def clean_env():
    """Provide a clean environment for tests."""
    with patch.dict(os.environ, {}, clear=True):
        yield

@pytest.fixture  
def test_env():
    """Provide a controlled test environment."""
    env_vars = {
        'API_KEY': 'pytest-test-key',
        'ENVIRONMENT': 'test',
        'LOG_LEVEL': 'DEBUG'
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield

@pytest.fixture
def test_settings():
    """Provide test-optimized settings."""
    return MyAppSettings.from_env(
        load_dotenv=False,
        enable_llm=False,
        enable_cache=False,
        api_key='fixture-test-key',
        temp_dir='/tmp/pytest'
    )

# test_settings.py
import pytest
from my_app.settings import MyAppSettings

def test_settings_defaults(clean_env):
    """Test settings with default values only."""
    settings = MyAppSettings.from_env(
        load_dotenv=False,
        api_key='test-key',  # Required field
        enable_llm=False
    )
    assert settings.api_timeout == 30
    assert not settings.debug_mode

def test_settings_with_env(test_env):
    """Test settings with environment variables."""
    settings = MyAppSettings.from_env(load_dotenv=False)
    assert settings.api_key == 'pytest-test-key'
    assert settings.environment == 'test'

def test_business_logic(test_settings):
    """Test business logic with pre-configured settings."""
    # Use the test_settings fixture
    assert test_settings.api_key == 'fixture-test-key'
    # Your business logic tests here

@pytest.mark.parametrize('api_timeout,expected', [
    (10, 10),
    (60, 60),
    (120, 120)
])
def test_api_timeout_values(clean_env, api_timeout, expected):
    """Test different API timeout values."""
    settings = MyAppSettings.from_env(
        load_dotenv=False,
        api_key='test',
        api_timeout=api_timeout,
        enable_llm=False
    )
    assert settings.api_timeout == expected
```

### Environment Variables

```bash
# Custom settings
API_KEY=your-secret-key
API_TIMEOUT=60
DEBUG_MODE=true
MAX_WORKERS=8
FEATURE_FLAGS=feature_a,feature_b,feature_c
TEMP_DIR=/tmp/myapp
OUTPUT_FORMAT=yaml

# Standard services still work
OPENAI_API_KEY=sk-your-key
REDIS_HOST=localhost
MCP_SERVER_PORT=8204
```

### Key Benefits of Direct Inheritance

✅ **Clean class hierarchy** - `MyAppSettings` IS-A `StandardSettings`  
✅ **No data copying** - Direct inheritance, no object-to-object copying  
✅ **Full type safety** - Proper dataclass with defined fields  
✅ **IDE support** - Full autocomplete and type checking  
✅ **Custom validation** - Override methods as needed  
✅ **Familiar pattern** - Standard OOP inheritance

### Alternative: Advanced Custom Settings (BaseSettings)

For more complex scenarios where you need full control over validation logic or want completely custom settings classes, you can still inherit from `BaseSettings`:

```python
from dataclasses import dataclass
from typing import Optional
from core_lib.config import BaseSettings, SettingsError

@dataclass(frozen=True)
class CustomAPISettings(BaseSettings):
    """Advanced custom settings with complex validation."""
    
    base_url: str = "https://api.example.com"
    timeout: int = 30
    max_retries: int = 3
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 1000
    
    @classmethod
    def from_env(cls, load_dotenv=True, dotenv_paths=None, **overrides):
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        settings_dict = {
            "base_url": cls.get_env("API_BASE_URL", default="https://api.example.com"),
            "timeout": cls.get_env("API_TIMEOUT", default=30, env_type=int),
            "max_retries": cls.get_env("API_MAX_RETRIES", default=3, env_type=int),
            "api_key": cls.get_env("API_KEY", required=True),
            "rate_limit_per_minute": cls.get_env("API_RATE_LIMIT", default=1000, env_type=int),
        }
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self):
        """Custom validation logic."""
        if self.timeout <= 0:
            raise SettingsError("Timeout must be positive")
        if self.max_retries < 0:
            raise SettingsError("Max retries cannot be negative")
        if not self.base_url.startswith(('http://', 'https://')):
            raise SettingsError("Base URL must be a valid HTTP/HTTPS URL")
```

> **💡 Recommendation**: Use `StandardSettings.extend_from_env()` for most use cases. Only use `BaseSettings` directly when you need complex custom validation or completely separate settings classes.

## Environment Variable Parsing

The system provides powerful environment variable parsing with type conversion:

```python
from core_lib.config import EnvParser

# Basic parsing with defaults
api_key = EnvParser.get_env("API_KEY", "OPENAI_API_KEY", default="")
timeout = EnvParser.get_env("TIMEOUT", default=30, env_type=int)
enabled = EnvParser.get_env("FEATURE_ENABLED", default=False, env_type=bool)

# Required variables (raises EnvironmentVariableError if missing)
db_password = EnvParser.get_env("DB_PASSWORD", required=True)

# List parsing (comma-separated values)
allowed_hosts = EnvParser.get_env("ALLOWED_HOSTS", default=[], env_type=list)
# ALLOWED_HOSTS=localhost,127.0.0.1,example.com -> ["localhost", "127.0.0.1", "example.com"]

# Multiple fallback names
redis_host = EnvParser.get_env("REDIS_HOST", "CACHE_HOST", "DB_HOST", default="localhost")
```

## Settings Manager

Use the settings manager to register and manage multiple configuration instances:

```python
from core_lib.config import settings_manager, StandardSettings

# Register settings
app_settings = StandardSettings.from_env()
settings_manager.register("app", app_settings)

# Add custom settings
db_settings = DatabaseSettings.from_env()
settings_manager.register("database", db_settings)

# Access settings
app_config = settings_manager.get_required("app")
db_config = settings_manager.get("database")

# Validate all settings
errors = settings_manager.validate_all()
if errors:
    print("Configuration errors found:")
    for name, error_list in errors.items():
        print(f"  {name}: {error_list}")

# Export all settings
all_settings = settings_manager.as_dict()
```

## Integration with Existing Libraries

### Using with LLM Clients

```python
from core_lib.config import StandardSettings
from core_lib.llm import LLMClient

# Load settings and create LLM client
settings = StandardSettings.from_env()
if settings.llm:
    llm_config = settings.get_llm_config()
    llm_client = LLMClient(llm_config)
    
    response = llm_client.chat("Hello, world!")
    print(response["content"])
```

### Using with Cache

```python
from core_lib.config import StandardSettings
from core_lib.cache import RedisCache

settings = StandardSettings.from_env()
if settings.cache:
    redis_config = settings.get_redis_config()
    cache = RedisCache(redis_config)
    
    cache.set("key", "value")
    value = cache.get("key")
```

### Using with Embeddings

```python
from core_lib.config import StandardSettings
from core_lib.embeddings import create_embeddings_provider

settings = StandardSettings.from_env()
if settings.embeddings:
    embeddings_config = settings.get_embeddings_config()
    embeddings_provider = create_embeddings_provider(embeddings_config)
    
    vectors = embeddings_provider.embed_texts(["Hello world"])
```

## Advanced Features

### Custom .env File Paths

```python
from pathlib import Path

# Load from specific paths
settings = StandardSettings.from_env(
    load_dotenv=True,
    dotenv_paths=[
        Path("/etc/myapp/.env"),      # System config
        Path.home() / ".myapp.env",    # User config  
        Path.cwd() / ".env.local"      # Local overrides
    ]
)
```

### Settings Merging and Overrides

```python
# Base settings from environment
base_settings = StandardSettings.from_env()

# Create variant with overrides
test_settings = base_settings.merge(
    environment="test",
    log_level="DEBUG",
    enable_cache=False
)

# Direct overrides during creation
settings = StandardSettings.from_env(
    app_name="custom-app",
    enable_tracing=False,
    # Override LLM model specifically
    llm=LLMSettings.from_env(model="gpt-3.5-turbo")
)
```

### Validation and Error Handling

```python
try:
    settings = StandardSettings.from_env()
    if not settings.is_valid:
        print("Settings validation failed:")
        for error in settings.validation_errors:
            print(f"  - {error}")
except EnvironmentVariableError as e:
    print(f"Missing required environment variable: {e}")
except SettingsError as e:
    print(f"Settings configuration error: {e}")
```

## Best Practices

1. **Use .env files for development**, environment variables for production
2. **Validate settings early** in your application startup
3. **Use the settings manager** for complex applications with multiple configuration sources
4. **Create custom settings classes** for domain-specific configuration
5. **Leverage auto-detection** to simplify configuration management
6. **Use type hints** and validation to catch configuration errors early
7. **Document your environment variables** for your team and deployment processes

## Environment Variables Reference

### Core Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `"app"` | Application name |
| `APP_VERSION` | `"0.1.0"` | Application version (fallback if not in pyproject.toml) |
| `ENVIRONMENT` | `"dev"` | Environment name (dev/prod/staging/etc.) |
| `LOG_LEVEL` | `"DEBUG"` (dev) / `"INFO"` (prod) | Logging level |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | Auto-detected | LLM provider (openai/azure/gemini/ollama) |
| `LLM_MODEL` | Provider-specific | Model name |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_MAX_TOKENS` | `None` | Maximum tokens to generate |
| `LLM_THINKING_ENABLED` | `false` | Enable thinking mode |

**OpenAI/Azure:**
- `OPENAI_API_KEY` - API key
- `OPENAI_BASE_URL` - Custom base URL
- `OPENAI_ORGANIZATION` - Organization ID  
- `OPENAI_PROJECT` - Project ID
- `AZURE_OPENAI_ENDPOINT` - Azure endpoint
- `AZURE_OPENAI_API_VERSION` - API version

**Gemini:**
- `GEMINI_API_KEY` / `GOOGLE_GENAI_API_KEY` - API key

**Ollama:**
- `OLLAMA_HOST` / `OLLAMA_BASE_URL` - Ollama server URL
- `OLLAMA_TIMEOUT` - Request timeout

### Embeddings Configuration

**See [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md) for comprehensive embedding configuration including high availability and token authentication.**

#### Unified Configuration (Single Provider)

```bash
EMBEDDING_PROVIDER=infinity              # Provider: openai|google_genai|ollama|infinity|local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Model name
EMBEDDING_BASE_URL=http://localhost:7997 # Server URL (works for all providers)
EMBEDDING_TIMEOUT=30                     # Request timeout in seconds
EMBEDDING_DIMENSION=384                  # Optional: embedding dimension
EMBEDDING_TASK_TYPE=SEMANTIC_SIMILARITY  # Optional: task type
EMBEDDING_CACHE_DURATION_SECONDS=7200    # Cache duration (default: 2 hours, 0=disabled)
```

#### Provider-Specific Configuration (Multi-Provider Setups)

```bash
# Common defaults
EMBEDDING_BASE_URL=http://default-server:7997
EMBEDDING_TIMEOUT=30

# Provider-specific overrides (take precedence)
INFINITY_BASE_URL=http://infinity-server:7997  # Comma-separated for HA
INFINITY_TOKEN=token123                         # Optional authentication
INFINITY_TIMEOUT=30
OLLAMA_URL=http://ollama-server:11434           # Comma-separated for HA
OLLAMA_TIMEOUT=60
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key
GOOGLE_GENAI_API_KEY=your-key
```

**Priority Chain:**
- Infinity: `INFINITY_BASE_URL` > `EMBEDDING_BASE_URL` > default
- Ollama: `OLLAMA_URL` > `EMBEDDING_BASE_URL` > default
- OpenAI: `OPENAI_BASE_URL` > `EMBEDDING_BASE_URL` > default

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `"infinity"` | Embeddings provider |
| `EMBEDDING_MODEL` | Provider-specific | Model name |
| `EMBEDDING_DIMENSION` | Model default | Embedding dimension |
| `EMBEDDING_TASK_TYPE` | `None` | Task type for Google GenAI |
| `EMBEDDING_BASE_URL` | Provider default | Generic server URL (all providers) |
| `INFINITY_BASE_URL` | `http://localhost:7997` | Infinity-specific URL(s) |
| `INFINITY_TOKEN` | `None` | Infinity authentication token(s) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama-specific URL(s) |
| `OPENAI_BASE_URL` | OpenAI API | OpenAI/Azure endpoint |

### Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_PROVIDER` | `"redis"` | Cache provider (redis/valkey) |
| `REDIS_HOST` / `VALKEY_HOST` | `"localhost"` | Cache server host |
| `REDIS_PORT` / `VALKEY_PORT` | `6379` | Cache server port |
| `REDIS_DB` / `VALKEY_DB` | `0` | Database number |
| `REDIS_PREFIX` / `VALKEY_PREFIX` | `"cache:"` | Key prefix |
| `REDIS_CACHE_TTL` / `VALKEY_CACHE_TTL` | `3600` | Default TTL in seconds |
| `REDIS_PASSWORD` / `VALKEY_PASSWORD` | `None` | Password |
| `REDIS_TIMEOUT` / `VALKEY_TIMEOUT` | `4` | Connection timeout |

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` / `DATABASE_HOST` | `"localhost"` | PostgreSQL server host |
| `POSTGRES_PORT` / `DATABASE_PORT` | `5432` | PostgreSQL server port |
| `POSTGRES_DB` / `DATABASE_NAME` | `"app"` | Database name |
| `POSTGRES_USER` / `DATABASE_USER` | `"app_user"` | Database username |
| `POSTGRES_PASSWORD` / `DATABASE_PASSWORD` | `"app_password"` | Database password |
| `POSTGRES_SSLMODE` / `DATABASE_SSLMODE` | `"disable"` | SSL mode (disable/allow/prefer/require/verify-ca/verify-full) |
| `POSTGRES_POOL_SIZE` / `DATABASE_POOL_SIZE` | `10` | Connection pool size |
| `POSTGRES_MAX_OVERFLOW` / `DATABASE_MAX_OVERFLOW` | `20` | Max pool overflow |
| `POSTGRES_POOL_TIMEOUT` / `DATABASE_POOL_TIMEOUT` | `30` | Pool timeout in seconds |

### Tracing Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_TRACING_ENABLED` | `true` | Enable Langfuse tracing (also accepts legacy `TRACING_ENABLED`) |
| `LANGFUSE_PUBLIC_KEY` | Required | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Required | Langfuse secret key |
| `LANGFUSE_HOST` | `"http://localhost:3000"` | Langfuse host URL |

### MCP Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_NAME` / `APP_NAME` | `"app-server"` | MCP server name |
| `MCP_SERVER_VERSION` / `APP_VERSION` | `"0.1.0"` | MCP server version |
| `MCP_SERVER_HOST` | `"0.0.0.0"` | Host address to bind server |
| `MCP_SERVER_PORT` | `8204` | Port number for server |
| `MCP_SERVER_URL` | Auto-generated | Full URL for server |
| `MCP_SERVER_TIMEOUT` | `30` | Request timeout in seconds |
| `MCP_TRANSPORT` | `"streamable-http"` | Transport type (streamable-http/stdio/websocket) |

### FastAPI HTTP Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FASTAPI_SERVER` | Auto-detected | Force enable/disable FastAPI server |
| `FASTAPI_HOST` | `"0.0.0.0"` | Host address to bind server |
| `FASTAPI_PORT` | `8096` | Port number for server |
| `FASTAPI_RELOAD` | `false` | Enable Hot Reload (development) |
| `FASTAPI_URL` | Auto-generated | Full URL for server |
| `API_AUTH_ENABLED` | `false` | Require API key authentication |
| `API_KEY_HEADER_NAME` | `"x-api-key"` | Header name to read API key from |
| `API_KEYS` | `[]` | Comma-separated list of valid keys |

### Service Enablement

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LLM` | Auto-detected | Force enable/disable LLM |
| `ENABLE_EMBEDDINGS` | Auto-detected | Force enable/disable embeddings |
| `ENABLE_CACHE` | Auto-detected | Force enable/disable cache |
| `ENABLE_TRACING` | Auto-detected | Force enable/disable tracing |
| `ENABLE_DATABASE` | Auto-detected | Force enable/disable database |
| `ENABLE_MCP_SERVER` | Auto-detected | Force enable/disable MCP server |

### Ollama-Specific Configuration

```bash
# Model and basic settings
OLLAMA_MODEL=llama3.2                    # Model name to use

# For LLM (chat/generation)
OLLAMA_BASE_URL=http://localhost:11434   # Ollama server URL for LLM
OLLAMA_TEMPERATURE=0.7                   # Sampling temperature (0.0-2.0)
OLLAMA_MAX_TOKENS=                       # Maximum tokens to generate (optional)
OLLAMA_THINKING_ENABLED=false            # Enable step-by-step thinking mode

# For Embeddings (use unified config or provider-specific)
EMBEDDING_BASE_URL=http://localhost:11434  # Recommended: unified config
# OR
OLLAMA_URL=http://localhost:11434          # Alternative: provider-specific

# Timeout settings
EMBEDDING_TIMEOUT=30                     # Unified timeout for embeddings
OLLAMA_TIMEOUT=60                        # Timeout for LLM operations

# Advanced Ollama settings
OLLAMA_NUM_CTX=                          # Context window size (optional)
OLLAMA_NUM_PREDICT=                      # Max tokens to predict (optional)
OLLAMA_REPEAT_PENALTY=                   # Repetition penalty (optional)
OLLAMA_TOP_K=                            # Top-K sampling (optional)
OLLAMA_TOP_P=                            # Top-P sampling (optional)
```

### Gemini-Specific Configuration

```bash
# Required settings
GEMINI_API_KEY=your-google-api-key       # Google API key (REQUIRED)
# OR
GOOGLE_GENAI_API_KEY=your-key            # Alternative key name

# Model and basic settings  
GEMINI_MODEL=gemini-1.5-flash            # Model name to use
GEMINI_TEMPERATURE=0.7                   # Sampling temperature (0.0-2.0)
GEMINI_MAX_TOKENS=                       # Maximum tokens to generate (optional)
GEMINI_THINKING_ENABLED=false            # Enable step-by-step thinking mode

# Advanced Gemini settings
GEMINI_BASE_URL=https://generativelanguage.googleapis.com  # API base URL
```

### Logging Configuration

#### Basic Logging

```bash
LOG_LEVEL=INFO                        # Root log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_ENABLED=true                 # Enable file logging
LOG_FILE_PATH=logs/app.log            # Log file path
LOG_FILE_MAX_BYTES=1048576            # Max file size before rotation (1MB)
LOG_FILE_BACKUP_COUNT=3               # Number of backup files
```

#### OTLP (OpenTelemetry) Logging

**Quick Setup (Auto-Enable):**

```bash
# Minimal config - OTLP auto-enables with these settings
ENABLE_LOGGER=true                                   # Enable logging features
OTLP_ENDPOINT=http://localhost:4318/v1/logs          # OTLP collector endpoint

# Optional: customize service identification (auto-detected)
APP_NAME=my-app                                      # Auto-used for OTLP_SERVICE_NAME
# OTLP_SERVICE_VERSION auto-detected from pyproject.toml
```

**Full Configuration:**

```bash
OTLP_ENABLED=true                                    # Explicit enable (optional if auto-enabled)
OTLP_ENDPOINT=http://localhost:4318/v1/logs          # OTLP collector endpoint
OTLP_LOG_LEVEL=INFO                                  # Independent OTLP log level (optional)
OTLP_HEADERS='{"Authorization": "Bearer token"}'     # Auth headers (JSON)
OTLP_TIMEOUT=10                                      # Request timeout (seconds)
OTLP_INSECURE=false                                  # Skip SSL verification
OTLP_SERVICE_NAME=my-app                             # Service name (default: APP_NAME or "core-lib")
OTLP_SERVICE_VERSION=1.0.0                           # Service version (default: from pyproject.toml)
```

**Auto-Enable Logic:**  
OTLP automatically enables when:
1. `ENABLE_LOGGER=true` (or `LOG_FILE_ENABLED=true`) AND
2. `OTLP_ENDPOINT` is defined AND
3. `OTLP_ENABLED` is not explicitly set to `false`

**Independent Log Levels:**

```bash
# Example: DEBUG on console, only INFO+ to OTLP
LOG_LEVEL=DEBUG
OTLP_LOG_LEVEL=INFO

# Example: Reduce OTLP costs - only errors
LOG_LEVEL=INFO
OTLP_LOG_LEVEL=ERROR
```

#### OVH Logs Data Platform

```bash
OVH_LDP_ENABLED=true                  # Enable OVH LDP
OVH_LDP_TOKEN=your-token              # Authentication token
OVH_LDP_ENDPOINT=gra1.logs.ovh.com    # OVH endpoint
OVH_LDP_PORT=12202                    # GELF TCP port
OVH_LDP_PROTOCOL=gelf_tcp             # Protocol: gelf_tcp, gelf_udp, syslog_tcp, syslog_udp
OVH_LDP_USE_TLS=true                  # Use TLS encryption
```

## Example .env Files

### Development Environment

```bash
# App settings
APP_NAME=my-app
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# LLM (local Ollama)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.7
OLLAMA_THINKING_ENABLED=true

# Embeddings (local Infinity)
EMBEDDING_PROVIDER=infinity
EMBEDDING_BASE_URL=http://localhost:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Cache (local Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Staging Environment

```bash
# App settings
APP_NAME=my-app
ENVIRONMENT=staging
LOG_LEVEL=INFO

# LLM (cloud Gemini)
GEMINI_API_KEY=your-staging-api-key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.3

# Embeddings (HA Infinity cluster)
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://staging-h1:7997,http://staging-h2:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Cache (HA Redis cluster)
REDIS_HOST=redis-staging.example.com
REDIS_PASSWORD=staging-password
REDIS_PREFIX=staging:
```

### Production Environment

```bash
# App settings
APP_NAME=my-app
ENVIRONMENT=production
LOG_LEVEL=INFO

# LLM (OpenAI)
OPENAI_API_KEY=sk-prod-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3

# Embeddings (HA Infinity with auth)
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://prod-h1:7997,http://prod-h2:7997,http://prod-h3:7997
INFINITY_TOKEN=token1,token2,token3
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
EMBEDDING_CACHE_DURATION_SECONDS=7200

# Cache (HA Redis with auth)
REDIS_HOST=redis-prod.example.com
REDIS_PASSWORD=prod-secret-password
REDIS_PREFIX=prod:
REDIS_CACHE_TTL=3600

# Database
POSTGRES_HOST=db-prod.example.com
POSTGRES_PORT=5432
POSTGRES_DB=production_db
POSTGRES_USER=prod_user
POSTGRES_PASSWORD=prod-db-password
POSTGRES_SSLMODE=require

# Tracing
ENABLE_TRACING=true
LANGFUSE_PUBLIC_KEY=pk_prod_key
LANGFUSE_SECRET_KEY=sk_prod_key
LANGFUSE_HOST=https://cloud.langfuse.com

# MCP Server
MCP_SERVER_NAME=prod-app-server
MCP_SERVER_PORT=8204
MCP_SERVER_HOST=0.0.0.0
MCP_TRANSPORT=streamable-http

# OTLP Logging
ENABLE_LOGGER=true
OTLP_ENDPOINT=https://otlp.example.com/v1/logs
OTLP_LOG_LEVEL=WARNING
OTLP_SERVICE_NAME=my-app-production
```

## Running Scripts

Use `uv run` to execute scripts with the proper environment:

```bash
# Run Python scripts
uv run python your_script.py

# Run examples
uv run python examples/example_embeddings_usage.py
uv run python examples/example_llm_usage.py

# Run tests
uv run pytest -q

# Run with specific .env file
uv run --env-file .env.production python your_script.py
```
