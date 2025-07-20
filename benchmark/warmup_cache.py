"""Warmup cache utility for Enhanced GPTCache benchmarking."""
import json
import argparse
import sys
import os
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.generate_queries import QueryItem
from src.cache.enhanced_cache import create_enhanced_cache

def warmup_cache_with_queries(cache, queries: List[QueryItem]):
    """Warmup cache with a list of queries.
    
    Args:
        cache: Enhanced cache instance
        queries: List of queries to use for warmup
    """
    print(f"Warming up cache with {len(queries)} queries...")
    
    for i, query_item in enumerate(queries):
        # Generate a standard response for the query
        response = f"Response for: {query_item.query[:50]}..."
        
        # Store in cache
        cache.set(
            query=query_item.query,
            response=response,
            conversation_id=query_item.conversation_id
        )
        
        if (i + 1) % 50 == 0:
            print(f"Warmed up {i + 1}/{len(queries)} queries")
    
    print("Cache warmup complete!")
    
    # Get cache stats after warmup
    stats = cache.get_stats()
    print(f"Cache contains {stats['cache_statistics']['total_queries']} entries")

def load_queries_from_file(file_path: str) -> List[QueryItem]:
    """Load queries from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    queries = []
    for query_data in data['queries']:
        query = QueryItem(**query_data)
        queries.append(query)
    
    return queries

def main():
    """Command-line interface for cache warmup."""
    parser = argparse.ArgumentParser(description="Warmup Enhanced GPTCache with queries")
    parser.add_argument("--dataset", required=True, help="Path to queries JSON file")
    parser.add_argument("--first", type=int, default=200, help="Number of first queries to use for warmup")
    parser.add_argument("--output", help="Path to save warmed cache state (optional)")
    
    args = parser.parse_args()
    
    # Load queries
    all_queries = load_queries_from_file(args.dataset)
    warmup_queries = all_queries[:args.first]
    
    print(f"Loaded {len(all_queries)} total queries, using first {len(warmup_queries)} for warmup")
    
    # Create enhanced cache
    cache = create_enhanced_cache(
        enable_context=True,
        enable_pca=True,
        enable_tau=True
    )
    
    # Warmup the cache
    warmup_cache_with_queries(cache, warmup_queries)
    
    if args.output:
        print(f"Warmed cache state would be saved to {args.output} (not implemented)")

if __name__ == "__main__":
    main()
