import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core_lib.cache.cache_manager import set_cache, get_cache, cache_get, cache_set
from core_lib.cache.redis_cache import RedisCache
from core_lib.cache.redis_config import RedisConfig

class TestRedisCache(unittest.TestCase):
    def setUp(self):
        # Reset the global cache instance before each test
        import core_lib.cache.cache_manager
        core_lib.cache.cache_manager._cache_instance = None

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_initialization_with_config(self, mock_redis):
        config = RedisConfig(host='localhost', port=6379, db=0, prefix='test:')
        cache = RedisCache('test_app', config=config)
        self.assertEqual(cache.name, 'test_app')
        self.assertEqual(cache.config.host, 'localhost')
        self.assertEqual(cache.config.port, 6379)

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_connect_success(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        cache = RedisCache('test_app')
        cache.connect()
        self.assertTrue(cache.connected)
        self.assertIsNotNone(cache.client)
        mock_client.ping.assert_called_once()

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_connect_failure(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.side_effect = Exception("Connection failed")
        cache = RedisCache('test_app')
        cache.connect()
        self.assertFalse(cache.connected)
        self.assertFalse(cache.client)

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_set_and_get(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = '"output"'
        mock_client.setex.return_value = True
        cache = RedisCache('test_app')
        cache.connect()
        cache.set({'a': 1}, 'output')
        result = cache.get({'a': 1})
        self.assertEqual(result, 'output')
        mock_client.setex.assert_called_once()

    def test_cache_bulk_get_and_set_use_single_operations(self):
        cache = RedisCache('test_app')
        cache.client = MagicMock()
        cache.client.mget.return_value = ['"first"', None, '"third"']

        assert cache.get_many(["a", "b", "c"]) == ["first", None, "third"]
        cache.client.mget.assert_called_once()

        pipeline = cache.client.pipeline.return_value
        cache.set_many([("a", [0.1]), ("b", [0.2])], ttl=600)

        cache.client.pipeline.assert_called_once_with(transaction=False)
        assert pipeline.setex.call_count == 2
        pipeline.sadd.assert_called_once()
        pipeline.execute.assert_called_once()

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_miss(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        cache = RedisCache('test_app')
        cache.connect()
        result = cache.get({'key': 'not_found'})
        self.assertIsNone(result)

    @patch('core_lib.cache.redis_cache.redis.Redis')
    def test_cache_with_custom_ttl(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        cache = RedisCache('test_app')
        cache.connect()
        cache.set({'key': 'test'}, 'value', ttl=600)
        args, kwargs = mock_client.setex.call_args
        self.assertEqual(args[1], 600)  # TTL should be 600

    def test_cache_operations_when_client_false(self):
        cache = RedisCache('test_app')
        cache.client = False
        result = cache.get({'key': 'test'})
        self.assertIsNone(result)
        cache.set({'key': 'test'}, 'value')

    @patch('core_lib.cache.cache_manager.create_cache')
    def test_get_cache_singleton(self, mock_create_cache):
        mock_instance = MagicMock()
        mock_instance.client = MagicMock()
        mock_create_cache.return_value = mock_instance
        # Use set_cache to initialize singleton
        from core_lib.cache.cache_manager import set_cache, get_cache
        set_cache()
        cache1 = get_cache()
        cache2 = get_cache()
        self.assertIs(cache1, cache2)

    @patch('core_lib.cache.cache_manager.create_cache')
    def test_get_cache_connection_failure(self, mock_create_cache):
        mock_instance = MagicMock()
        mock_instance.client = False
        mock_create_cache.return_value = mock_instance
        from core_lib.cache.cache_manager import set_cache, get_cache
        set_cache()
        cache = get_cache()
        self.assertFalse(cache)

    @patch('core_lib.cache.cache_manager.get_cache')
    def test_cache_get_helper(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = 'cached_value'
        mock_get_cache.return_value = mock_cache
        result = cache_get({'key': 'test'})
        self.assertEqual(result, 'cached_value')
        mock_cache.get.assert_called_once_with({'key': 'test'})

    @patch('core_lib.cache.cache_manager.get_cache')
    def test_cache_get_helper_when_cache_false(self, mock_get_cache):
        mock_get_cache.return_value = False
        result = cache_get({'key': 'test'})
        self.assertIsNone(result)

    @patch('core_lib.cache.cache_manager.get_cache')
    def test_cache_set_helper(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        cache_set({'key': 'test'}, 'value', ttl=300)
        mock_cache.set.assert_called_once_with({'key': 'test'}, 'value', 300)

    @patch('core_lib.cache.cache_manager.get_cache')
    def test_cache_set_helper_when_cache_false(self, mock_get_cache):
        mock_get_cache.return_value = False
        cache_set({'key': 'test'}, 'value')

    def test_make_key_consistency(self):
        cache = RedisCache('test_app')
        key1 = cache._make_key({'a': 1, 'b': 2})
        key2 = cache._make_key({'b': 2, 'a': 1})  # Different order
        self.assertEqual(key1, key2)  # Should be the same due to sort_keys=True
        self.assertTrue(key1.startswith(cache.config.prefix + 'test_app:'))

    @patch('core_lib.cache.redis_config.os.getenv')
    def test_redis_config_from_env(self, mock_getenv):
        mock_getenv.side_effect = lambda key, default: {
            'REDIS_HOST': 'redis-server',
            'REDIS_PORT': '6380',
            'REDIS_DB': '1',
            'REDIS_PREFIX': 'myapp:',
            'REDIS_CACHE_TTL': '7200',
            'REDIS_PASSWORD': 'secret',
            'REDIS_TIMEOUT': '5'
        }.get(key, default)
        config = RedisConfig.from_env()
        self.assertEqual(config.host, 'redis-server')
        self.assertEqual(config.port, 6380)
        self.assertEqual(config.db, 1)
        self.assertEqual(config.prefix, 'myapp:')

if __name__ == '__main__':
    unittest.main()
