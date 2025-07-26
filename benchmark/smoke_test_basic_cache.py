#!/usr/bin/env python3
"""Smoke test for basic cache functionality."""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cache.basic_cache import init_basic_cache, mock_llm, get_cache_stats

def run_smoke_test():
    """Run smoke test to verify cache hits."""
    print("=" * 60)
    print("SMOKE TEST: Basic GPTCache Functionality")
    print("=" * 60)
    
    # Initialize cache
    try:
        init_basic_cache()
        print("✓ Cache initialization successful")
    except Exception as e:
        print(f"✗ Cache initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test queries - Q1, Q2, Q1 (expect hit on third)
    queries = [
        "What is Python?",           # Q1 - miss
        "Explain machine learning",  # Q2 - miss  
        "What is Python?"            # Q1 again - should hit
    ]
    
    print(f"\nTesting {len(queries)} queries:")
    print("-" * 40)
    
    timings = []
    responses = []
    
    for i, query in enumerate(queries, 1):
        start_time = time.time()
        response = mock_llm(query)
        elapsed = (time.time() - start_time) * 1000
        
        timings.append(elapsed)
        responses.append(response[:50] + "..." if len(response) > 50 else response)
        
        print(f"Query {i}: {query}")
        print(f"Response: {responses[-1]}")
        print(f"Time: {elapsed:.2f}ms")
        print()
    
    # Get final statistics
    stats = get_cache_stats()
    
    print("-" * 40)
    print("FINAL RESULTS:")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Cache hits: {stats['hit_count']}")
    print(f"Cache misses: {stats['miss_count']}")
    print(f"Hit ratio: {stats['hit_ratio']:.2%}")
    
    # Performance analysis
    if len(timings) >= 3:
        first_call = timings[0]   # Q1 first time
        third_call = timings[2]   # Q1 second time (should be faster)
        
        print(f"\nPerformance Analysis:")
        print(f"First call (Q1):  {first_call:.2f}ms")  
        print(f"Third call (Q1):  {third_call:.2f}ms")
        
        if third_call < first_call:
            speedup = first_call / third_call
            print(f"Speedup: {speedup:.1f}x faster! 🚀")
        else:
            print("No speedup detected")
    
    # Success criteria
    success = stats['hit_ratio'] > 0
    
    if success:
        print(f"\n🎉 SUCCESS: Cache achieved {stats['hit_ratio']:.2%} hit rate!")
    else:
        print(f"\n💥 FAILURE: No cache hits achieved")
    
    # Save metrics for CI
    metrics = {
        'hit_ratio': stats['hit_ratio'],
        'hit_count': stats['hit_count'],
        'miss_count': stats['miss_count'],
        'total_queries': stats['total_queries'],
        'timings_ms': timings,
        'success': success,
        'timestamp': time.time()
    }
    
    try:
        os.makedirs('data/metrics', exist_ok=True)
        with open('data/metrics/basic_cache_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Metrics saved to data/metrics/basic_cache_metrics.json")
    except Exception as e:
        print(f"\n⚠️  Could not save metrics: {e}")
    
    return success

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
