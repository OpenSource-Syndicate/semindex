"""
Cache System for semindex
Provides caching for embeddings, file hashes, search results, and other expensive operations
"""
import hashlib
import pickle
import time
import os
import threading
from typing import Any, Dict, Optional, List
from collections import OrderedDict


class LRUCache:
    """Simple LRU cache with TTL support for expensive operations"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and hasn't expired"""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set a value in the cache with current timestamp"""
        with self._lock:
            # Remove oldest entries if we're at max size
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest (first item)
            
            self.cache[key] = (value, time.time())
            # Move to end (most recently used)
            self.cache.move_to_end(key)
    
    def clear(self):
        """Clear all entries from the cache"""
        with self._lock:
            self.cache.clear()


class CacheManager:
    """Centralized caching manager for semindex operations"""
    
    def __init__(self, max_cache_size: int = 10000):
        self.embedding_cache = LRUCache(max_size=max_cache_size * 2, ttl=7200)  # 2 hours
        self.file_hash_cache = LRUCache(max_size=max_cache_size * 5, ttl=86400)  # 24 hours
        self.search_cache = LRUCache(max_size=max_cache_size // 10, ttl=1800)  # 30 minutes
        self.symbol_parse_cache = LRUCache(max_size=max_cache_size, ttl=3600)  # 1 hour
        self.context_cache = LRUCache(max_size=max_cache_size, ttl=1800)  # 30 minutes
        self._lock = threading.Lock()
    
    def get_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(key)
    
    def cache_embedding(self, text: str, embedding: Any):
        """Cache embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache.set(key, embedding)
    
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get cached hash for file path"""
        key = f"hash_{file_path}"
        return self.file_hash_cache.get(key)
    
    def cache_file_hash(self, file_path: str, hash_value: str):
        """Cache hash for file path"""
        key = f"hash_{file_path}"
        self.file_hash_cache.set(key, hash_value)
    
    def get_search_result(self, query: str, index_path: str) -> Optional[Any]:
        """Get cached search result"""
        key = hashlib.md5(f"{query}_{index_path}".encode()).hexdigest()
        return self.search_cache.get(key)
    
    def cache_search_result(self, query: str, index_path: str, result: Any):
        """Cache search result"""
        key = hashlib.md5(f"{query}_{index_path}".encode()).hexdigest()
        self.search_cache.set(key, result)
    
    def get_parsed_symbol(self, file_path: str, content_hash: str) -> Optional[Any]:
        """Get cached parsed symbols for file"""
        key = f"symbol_{file_path}_{content_hash}"
        return self.symbol_parse_cache.get(key)
    
    def cache_parsed_symbol(self, file_path: str, content_hash: str, symbols: Any):
        """Cache parsed symbols for file"""
        key = f"symbol_{file_path}_{content_hash}"
        self.symbol_parse_cache.set(key, symbols)
    
    def get_context(self, context_identifier: str) -> Optional[Any]:
        """Get cached context"""
        key = hashlib.md5(context_identifier.encode()).hexdigest()
        return self.context_cache.get(key)
    
    def cache_context(self, context_identifier: str, context: Any):
        """Cache context"""
        key = hashlib.md5(context_identifier.encode()).hexdigest()
        self.context_cache.set(key, context)
    
    def clear_all(self):
        """Clear all cache types"""
        with self._lock:
            self.embedding_cache.clear()
            self.file_hash_cache.clear()
            self.search_cache.clear()
            self.symbol_parse_cache.clear()
            self.context_cache.clear()


# Global cache instance
cache_manager = CacheManager()