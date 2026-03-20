import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core_lib.tracing.tracing import (
    TracingProvider,
    LangfuseTracingProvider,
    TracingManager,
    setup_tracing
)


class ConcreteTracingProvider(TracingProvider):
    """Concrete implementation for testing abstract base class."""
    
    def __init__(self):
        self.metadata_calls = []
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        self.metadata_calls.append(metadata)


class TestTracingProvider(unittest.TestCase):
    """Test the abstract TracingProvider base class."""
    
    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        # Should not be able to instantiate abstract class directly
        with self.assertRaises(TypeError):
            TracingProvider()
    
    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""
        provider = ConcreteTracingProvider()
        
        # Test add_metadata
        metadata = {"user_id": "123", "session": "abc"}
        provider.add_metadata(metadata)
        self.assertEqual(len(provider.metadata_calls), 1)
        self.assertEqual(provider.metadata_calls[0], metadata)


class TestLangfuseTracingProvider(unittest.TestCase):
    """Test the LangfuseTracingProvider implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_langfuse_client = Mock()
        self.provider = LangfuseTracingProvider(self.mock_langfuse_client)
    
    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider._client, self.mock_langfuse_client)
    
    def test_add_metadata(self):
        """Test add_metadata method."""
        metadata = {"user_id": "user123", "environment": "test"}
        self.provider.add_metadata(metadata)
        
        self.mock_langfuse_client.update_current_span.assert_called_once_with(metadata=metadata)
    
    def test_add_metadata_multiple_calls(self):
        """Test multiple calls to add_metadata."""
        metadata1 = {"key1": "value1"}
        metadata2 = {"key2": "value2"}
        
        self.provider.add_metadata(metadata1)
        self.provider.add_metadata(metadata2)
        
        expected_calls = [
            call(metadata=metadata1),
            call(metadata=metadata2)
        ]
        self.mock_langfuse_client.update_current_span.assert_has_calls(expected_calls)


class TestTracingManager(unittest.TestCase):
    """Test the TracingManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear environment variables to ensure clean test state
        self.original_env = {}
        for key in ["APP_NAME", "APP_VERSION", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_initialization_with_service_name(self):
        """Test manager initialization with service name."""
        manager = TracingManager("test-service")
        self.assertEqual(manager.settings.service_name, "test-service")
        self.assertEqual(manager.settings.service_version, "0.1.0")
        self.assertIsNone(manager._provider)
        self.assertFalse(manager._initialized)
    
    def test_initialization_without_service_name(self):
        """Test manager initialization without service name."""
        manager = TracingManager()
        self.assertEqual(manager.settings.service_name, "unknown")
        self.assertEqual(manager.settings.service_version, "0.1.0")
    
    def test_initialization_with_env_variables(self):
        """Test manager initialization with environment variables."""
        os.environ["APP_NAME"] = "env-service"
        os.environ["APP_VERSION"] = "2.0.0"
        
        try:
            manager = TracingManager()
            self.assertEqual(manager.settings.service_name, "env-service")
            self.assertEqual(manager.settings.service_version, "2.0.0")
        finally:
            os.environ.pop("APP_NAME", None)
            os.environ.pop("APP_VERSION", None)
    
    def test_service_name_priority(self):
        """Test that explicit service name takes priority over environment variable."""
        os.environ["APP_NAME"] = "env-service"
        
        try:
            manager = TracingManager("explicit-service")
            self.assertEqual(manager.settings.service_name, "explicit-service")
        finally:
            os.environ.pop("APP_NAME", None)
    
    @patch('core_lib.tracing.tracing.trace')
    @patch('core_lib.tracing.tracing.get_client')
    def test_setup_already_initialized_tracer(self, mock_get_client, mock_trace):
        """Test setup when tracer is already initialized."""
        # Mock that TracerProvider is already set
        mock_trace.get_tracer_provider.return_value = Mock()  # Not ProxyTracerProvider
        mock_trace.ProxyTracerProvider.return_value = Mock()
        
        mock_langfuse_client = Mock()
        mock_get_client.return_value = mock_langfuse_client
        
        manager = TracingManager("test-service")
        provider = manager.setup()
        
        # Should return LangfuseTracingProvider
        self.assertIsInstance(provider, LangfuseTracingProvider)
        self.assertEqual(provider._client, mock_langfuse_client)
        self.assertTrue(manager._initialized)
        mock_get_client.assert_called_once()
    
    @patch('core_lib.tracing.tracing.trace')
    @patch('core_lib.tracing.tracing.TracerProvider')
    @patch('core_lib.tracing.tracing.Resource')
    @patch('core_lib.tracing.tracing.Langfuse')
    @patch('core_lib.tracing.tracing.resolve_machine_identity')
    def test_setup_fresh_initialization(self, mock_resolve_machine_identity, mock_langfuse, mock_resource, mock_tracer_provider, mock_trace):
        """Test setup with fresh initialization."""
        # Mock that TracerProvider is not set (ProxyTracerProvider)
        proxy_provider = Mock()
        mock_trace.get_tracer_provider.return_value = proxy_provider
        mock_trace.ProxyTracerProvider.return_value = proxy_provider
        
        # Set up mocks
        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance
        
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        
        mock_langfuse_client = Mock()
        mock_langfuse.return_value = mock_langfuse_client
        mock_resolve_machine_identity.return_value = ("server-01", "host-01")
        
        # Set environment variables
        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_public_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret_key"
        os.environ["LANGFUSE_HOST"] = "http://test.langfuse.com"
        
        manager = TracingManager("test-service")
        provider = manager.setup()
        
        # Verify Resource creation
        mock_resource.create.assert_called_once_with({
            "service.name": "test-service",
            "service.version": "0.1.0",
            "service.instance.id": "server-01",
            "host.name": "host-01",
        })
        
        # Verify TracerProvider setup
        mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider_instance)
        
        # Verify Langfuse setup
        mock_langfuse.assert_called_once_with(
            x_langfuse_sdk_name="Langfuse Python SDK",
            x_langfuse_sdk_version="1.0.0",
            x_langfuse_public_key="test_public_key",
            username="test_secret_key",
            password="",
            base_url="http://test.langfuse.com",
        )
        
        # Verify return value
        self.assertIsInstance(provider, LangfuseTracingProvider)
        self.assertEqual(provider._client, mock_langfuse_client)
        self.assertTrue(manager._initialized)
    
    @patch('core_lib.tracing.tracing.trace')
    @patch('core_lib.tracing.tracing.TracerProvider')
    @patch('core_lib.tracing.tracing.Resource')
    @patch('core_lib.tracing.tracing.Langfuse')
    def test_setup_default_langfuse_host(self, mock_langfuse, mock_resource, mock_tracer_provider, mock_trace):
        """Test setup with default Langfuse host."""
        # Mock that TracerProvider is not set
        proxy_provider = Mock()
        mock_trace.get_tracer_provider.return_value = proxy_provider
        mock_trace.ProxyTracerProvider.return_value = proxy_provider
        
        # Set up mocks
        mock_resource.create.return_value = Mock()
        mock_tracer_provider.return_value = Mock()
        mock_langfuse.return_value = Mock()
        
        manager = TracingManager("test-service")
        manager.setup()
        
        # Verify Langfuse called with default host
        mock_langfuse.assert_called_once()
        call_kwargs = mock_langfuse.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "http://localhost:3000")
    
    def test_setup_idempotent(self):
        """Test that setup is idempotent."""
        manager = TracingManager("test-service")
        
        with patch.object(manager, '_initialized', True), \
             patch.object(manager, '_provider', Mock()) as mock_provider:
            
            result = manager.setup()
            self.assertEqual(result, mock_provider)
    
    def test_get_provider_before_setup(self):
        """Test get_provider before setup."""
        manager = TracingManager("test-service")
        self.assertIsNone(manager.get_provider())
    
    def test_get_provider_after_setup(self):
        """Test get_provider after setup."""
        manager = TracingManager("test-service")
        mock_provider = Mock()
        manager._provider = mock_provider
        
        self.assertEqual(manager.get_provider(), mock_provider)
    
    def test_add_metadata_with_provider(self):
        """Test add_metadata when provider exists."""
        manager = TracingManager("test-service")
        mock_provider = Mock()
        manager._provider = mock_provider
        metadata = {"name": "test", "version": "1.0"}
        manager.add_metadata(metadata)
        mock_provider.add_metadata.assert_called_once_with(metadata)

    def test_add_metadata_without_provider(self):
        """Test add_metadata when provider doesn't exist."""
        manager = TracingManager("test-service")
        # Should not raise exception
        manager.add_metadata({"name": "test"})

    def test_initialization_with_settings(self):
        """Test TracingManager initialization with TracingSettings."""
        from core_lib.config.tracing_settings import TracingSettings
        
        settings = TracingSettings(
            enabled=True,
            service_name="settings-service",
            service_version="2.0.0",
            langfuse_public_key="settings-public-key",
            langfuse_secret_key="settings-secret-key",
            langfuse_host="https://settings-host.com"
        )
        
        manager = TracingManager(service_name="override-service", settings=settings)
        
        # Settings should take precedence, but service_name parameter should override settings.service_name
        self.assertEqual(manager.settings.service_name, "override-service")
        self.assertEqual(manager.settings.service_version, "2.0.0")
        self.assertEqual(manager.settings.langfuse_public_key, "settings-public-key")
        self.assertEqual(manager.settings.langfuse_secret_key, "settings-secret-key")
        self.assertEqual(manager.settings.langfuse_host, "https://settings-host.com")
        self.assertTrue(manager.settings.enabled)

    def test_initialization_with_settings_disabled(self):
        """Test TracingManager initialization with disabled tracing."""
        from core_lib.config.tracing_settings import TracingSettings
        
        settings = TracingSettings(
            enabled=False,
            service_name="disabled-service",
            langfuse_public_key="disabled-key",
            langfuse_secret_key="disabled-secret"
        )
        
        manager = TracingManager(settings=settings)
        self.assertFalse(manager.settings.enabled)
        
        # When disabled, setup should return NoOpTracingProvider
        provider = manager.setup()
        from core_lib.tracing.tracing import NoOpTracingProvider
        self.assertIsInstance(provider, NoOpTracingProvider)


class TestSetupTracing(unittest.TestCase):
    """Test the setup_tracing function."""
    
    @patch('core_lib.tracing.tracing.TracingManager')
    def test_setup_tracing_with_name(self, mock_manager_class):
        """Test setup_tracing function with service name."""
        mock_manager = Mock()
        mock_provider = Mock()
        mock_manager.setup.return_value = mock_provider
        mock_manager_class.return_value = mock_manager
        
        result = setup_tracing("test-service")
        
        mock_manager_class.assert_called_once_with(service_name="test-service", settings=None)
        mock_manager.setup.assert_called_once()
        self.assertEqual(result, mock_provider)
    
    @patch('core_lib.tracing.tracing.TracingManager')
    def test_setup_tracing_without_name(self, mock_manager_class):
        """Test setup_tracing function without service name."""
        mock_manager = Mock()
        mock_provider = Mock()
        mock_manager.setup.return_value = mock_provider
        mock_manager_class.return_value = mock_manager
        
        result = setup_tracing()
        
        mock_manager_class.assert_called_once_with(service_name=None, settings=None)
        mock_manager.setup.assert_called_once()
        self.assertEqual(result, mock_provider)
    
    @patch('core_lib.tracing.tracing.TracingManager')
    def test_setup_tracing_with_settings(self, mock_manager_class):
        """Test setup_tracing function with TracingSettings."""
        from core_lib.config.tracing_settings import TracingSettings
        
        mock_manager = Mock()
        mock_provider = Mock()
        mock_manager.setup.return_value = mock_provider
        mock_manager_class.return_value = mock_manager
        
        # Create mock settings
        settings = TracingSettings(
            enabled=True,
            service_name="test-service",
            service_version="1.0.0",
            langfuse_public_key="test-public-key",
            langfuse_secret_key="test-secret-key",
            langfuse_host="https://test-host.com"
        )
        
        result = setup_tracing(name="override-service", settings=settings)
        
        mock_manager_class.assert_called_once_with(service_name="override-service", settings=settings)
        mock_manager.setup.assert_called_once()
        self.assertEqual(result, mock_provider)


class TestTracingIntegration(unittest.TestCase):
    """Integration tests for tracing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear environment variables
        self.original_env = {}
        for key in ["APP_NAME", "APP_VERSION", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Clean up after tests."""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    @patch('core_lib.tracing.tracing.trace')
    @patch('core_lib.tracing.tracing.Langfuse')
    def test_end_to_end_workflow(self, mock_langfuse, mock_trace):
        """Test complete workflow from setup to usage."""
        # Setup mocks
        proxy_provider = Mock()
        mock_trace.get_tracer_provider.return_value = proxy_provider
        mock_trace.ProxyTracerProvider.return_value = proxy_provider
        
        mock_langfuse_client = Mock()
        mock_langfuse.return_value = mock_langfuse_client
        
        # Set environment
        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"
        
        # Create manager and setup
        manager = TracingManager("integration-test")
        provider = manager.setup()
        
        # Use the provider
        provider.add_metadata({"user": "test_user", "action": "test_action"})
        
        # Use manager methods
        manager.add_metadata({"component": "tracing"})
        
        # Verify interactions - all calls in order
        expected_all_calls = [
            call(metadata={"user": "test_user", "action": "test_action"}),
            call(metadata={"component": "tracing"})
        ]
        mock_langfuse_client.update_current_span.assert_has_calls(expected_all_calls)
