"""Basic GPTCache implementation to demonstrate cache hits."""

import time
from gptcache import cache, Config
from gptcache.adapter.api import put, get
from gptcache.embedding import SBERT
from gptcache.manager import get_data_manager, CacheBase, VectorBase

# Configuration
SIM_THRESHOLD = 0.70  # loose threshold for easy hits

# Manual tracking for statistics
hit_count = 0
miss_count = 0

def init_basic_cache():
    """Initialize GPTCache with basic configuration."""
    # Create SBERT embeddings
    sbert = SBERT()
    
    # Create data manager with SQLite + FAISS
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=sbert.dimension)
    data_manager = get_data_manager(cache_base, vector_base)
    
    # Create config with similarity threshold
    config = Config(similarity_threshold=SIM_THRESHOLD)
    
    # Initialize global cache
    cache.init(
        embedding_func=sbert.to_embeddings,
        data_manager=data_manager,
        config=config
    )
    
    print(f"✓ Basic cache initialized with similarity threshold: {SIM_THRESHOLD}")
    return cache

def mock_llm(prompt: str) -> str:
    """Mock LLM with cache integration."""
    global hit_count, miss_count
    
    start_time = time.time()
    
    # Try to get from cache first using adapter API
    cached_response = get(prompt)
    
    if cached_response is not None:
        # Cache hit
        hit_count += 1
        elapsed = (time.time() - start_time) * 1000
        print(f"�� CACHE HIT ({elapsed:.2f}ms): '{prompt[:30]}...'")
        return cached_response
    else:
        # Cache miss - generate mock response
        miss_count += 1
        response = f"MOCK_REPLY: {prompt}"
        
        # Store in cache using adapter API
        put(prompt, response)
        
        elapsed = (time.time() - start_time) * 1000
        print(f"❌ CACHE MISS ({elapsed:.2f}ms): '{prompt[:30]}...'")
        return response

def get_cache_stats():
    """Get current cache statistics."""
    global hit_count, miss_count
    
    total = hit_count + miss_count
    hit_ratio = hit_count / total if total > 0 else 0.0
    
    return {
        'hit_count': hit_count,
        'miss_count': miss_count,
        'total_queries': total,
        'hit_ratio': hit_ratio
    }
