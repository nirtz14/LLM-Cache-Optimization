"""Unit tests for basic cache functionality."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cache.basic_cache import init_basic_cache, mock_llm, get_cache_stats

class TestBasicCache:
    """Test basic cache functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Initialize fresh cache for each test
        init_basic_cache()
    
    def test_exact_hit(self):
        """Test that exact duplicate queries produce cache hits."""
        query = "What is AI?"
        
        # First call - should be a miss
        response1 = mock_llm(query)
        assert response1 is not None
        assert "MOCK_REPLY:" in response1
        
        # Second call with same query - should be a hit
        response2 = mock_llm(query)
        assert response2 is not None
        assert response2 == response1  # Should be identical
        
        # Check cache statistics
        stats = get_cache_stats()
        assert stats['hit_ratio'] > 0, f"Expected hit ratio > 0, got {stats['hit_ratio']}"
        assert stats['hit_count'] >= 1, f"Expected at least 1 hit, got {stats['hit_count']}"
        
    def test_different_queries_miss(self):
        """Test that different queries produce misses."""
        query1 = "What is Python?"
        query2 = "What is Java?"
        
        response1 = mock_llm(query1)
        response2 = mock_llm(query2)
        
        # Responses should be different
        assert response1 != response2
        assert "Python" in response1
        assert "Java" in response2
        
        # Both should be misses (no hits yet)
        stats = get_cache_stats()
        assert stats['miss_count'] >= 2
        
    def test_mixed_hits_and_misses(self):
        """Test combination of hits and misses."""
        # Create some cache entries
        mock_llm("Query A")  # miss
        mock_llm("Query B")  # miss
        mock_llm("Query A")  # hit
        mock_llm("Query C")  # miss
        mock_llm("Query B")  # hit
        
        stats = get_cache_stats()
        
        # Should have some hits and some misses
        assert stats['total_queries'] == 5
        assert stats['hit_count'] >= 2, f"Expected at least 2 hits, got {stats['hit_count']}"
        assert stats['miss_count'] >= 3, f"Expected at least 3 misses, got {stats['miss_count']}"
        assert stats['hit_ratio'] > 0, f"Expected positive hit ratio, got {stats['hit_ratio']}"
        
    def test_cache_stats_structure(self):
        """Test that cache statistics have correct structure."""
        # Make at least one call
        mock_llm("Test query")
        
        stats = get_cache_stats()
        
        # Check required fields
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert 'total_queries' in stats
        assert 'hit_ratio' in stats
        
        # Check types
        assert isinstance(stats['hit_count'], int)
        assert isinstance(stats['miss_count'], int) 
        assert isinstance(stats['total_queries'], int)
        assert isinstance(stats['hit_ratio'], float)
        
        # Check relationships
        assert stats['total_queries'] == stats['hit_count'] + stats['miss_count']
        assert 0 <= stats['hit_ratio'] <= 1
