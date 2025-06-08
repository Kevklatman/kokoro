"""
G2P Cache wrapper for improved performance
"""
from typing import Any, Callable, Dict
from loguru import logger
import functools

class G2PCache:
    """
    Wrapper class for G2P objects that implements caching for improved performance.
    This caches the results of phoneme generation to avoid redundant processing.
    """
    
    def __init__(self, g2p_obj: Any):
        """
        Initialize the G2P cache wrapper.
        
        Args:
            g2p_obj: The original G2P object to wrap
        """
        self._g2p_obj = g2p_obj
        self._phoneme_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("G2P caching mechanism initialized")
    
    def __call__(self, text: str) -> Any:
        """
        Call the G2P object with caching.
        
        Args:
            text: The text to convert to phonemes
            
        Returns:
            The phoneme representation from the wrapped G2P object
        """
        # Use a hash of the text as the cache key to avoid any issues with very long texts
        cache_key = text
        
        # Check if text is in cache
        if cache_key in self._phoneme_cache:
            self._cache_hits += 1
            if self._cache_hits % 10 == 0:  # Log only occasionally to avoid flooding
                logger.debug(f"Cache hit ({self._cache_hits} hits, {self._cache_misses} misses)")
            return self._phoneme_cache[cache_key]
        
        # Process text with original G2P object
        try:
            result = self._g2p_obj(text)
            
            # Cache the result
            self._phoneme_cache[cache_key] = result
            self._cache_misses += 1
            if self._cache_misses % 10 == 0:  # Log only occasionally
                logger.debug(f"Cache miss ({self._cache_hits} hits, {self._cache_misses} misses)")
            
            return result
        except Exception as e:
            logger.error(f"Error in G2P processing: {str(e)}")
            # If there's an error, don't cache the result and just pass through
            return self._g2p_obj(text)
    
    def __getattr__(self, name: str) -> Any:
        """
        Pass through any attribute access to the wrapped G2P object.
        
        Args:
            name: The attribute name to access
            
        Returns:
            The attribute from the wrapped G2P object
        """
        return getattr(self._g2p_obj, name)
        
    def clear_cache(self):
        """Clear the phoneme cache"""
        cache_size = len(self._phoneme_cache)
        self._phoneme_cache.clear()
        logger.info(f"Cleared phoneme cache ({cache_size} entries)")
        
    def get_stats(self):
        """Get cache statistics"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._phoneme_cache),
            "hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
