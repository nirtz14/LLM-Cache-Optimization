#!/usr/bin/env python3
"""
Demo script showing Enhanced GPTCache usage with all three MeanCache features.
"""
import time
from src.cache.enhanced_cache import create_enhanced_cache

def demo_basic_usage():
    """Demonstrate basic enhanced cache usage."""
    print("=== Enhanced GPTCache Demo ===\n")
    
    # Create enhanced cache with all features enabled
    print("Creating enhanced cache with all features...")
    cache = create_enhanced_cache(
        enable_context=True,
        enable_pca=True, 
        enable_tau=True
    )
    print("✓ Enhanced cache created\n")
    
    # Demo conversation with context tracking
    print("Demo 1: Context-aware caching")
    print("-" * 30)
    
    # First conversation
    conv1_queries = [
        "Hello, I need help with Python",
        "How do I create a list?",
        "What about dictionaries?",
        "Can you show me loops?"
    ]
    
    responses = [
        "Hi! I'd be happy to help with Python programming.",
        "You can create a list using square brackets: my_list = []",
        "Dictionaries use curly braces: my_dict = {}",
        "Python has for and while loops. For example: for i in range(10):"
    ]
    
    # Add queries to cache with conversation context
    for i, (query, response) in enumerate(zip(conv1_queries, responses)):
        print(f"Adding to cache: '{query[:30]}...'")
        cache.set(query, response, conversation_id="conv1")
        
        # Query back immediately (should hit)
        result = cache.query(query, conversation_id="conv1")
        status = "HIT" if result['cache_hit'] else "MISS"
        print(f"  Query result: {status} ({result['latency_ms']:.1f}ms)")
    
    print()
    
    # Try querying from different conversation (context should filter)
    print("Querying from different conversation context...")
    result = cache.query("How do I create a list?", conversation_id="conv2")
    status = "HIT" if result['cache_hit'] else "MISS"
    print(f"  Cross-conversation query: {status} ({result['latency_ms']:.1f}ms)")
    print()
    
    # Demo 2: Similar queries (PCA compression and τ-tuning)
    print("Demo 2: Similar query handling")
    print("-" * 30)
    
    similar_queries = [
        "What is machine learning?",
        "Can you explain machine learning?", 
        "Tell me about machine learning",
        "How does machine learning work?",
        "What are the basics of machine learning?"
    ]
    
    base_response = "Machine learning is a subset of AI that enables computers to learn from data."
    
    # Add first query
    cache.set(similar_queries[0], base_response, conversation_id="ml_conv")
    print(f"Added base query: '{similar_queries[0]}'")
    
    # Test similar queries
    for query in similar_queries[1:]:
        result = cache.query(query, conversation_id="ml_conv")
        status = "HIT" if result['cache_hit'] else "MISS"
        threshold = result['similarity_threshold']
        print(f"  '{query[:40]}...': {status} (τ={threshold:.3f}, {result['latency_ms']:.1f}ms)")
    
    print()
    
    # Get comprehensive statistics
    print("=== Cache Statistics ===")
    stats = cache.get_stats()
    
    cache_stats = stats['cache_statistics']
    print(f"Total queries: {cache_stats['total_queries']}")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Cache misses: {cache_stats['cache_misses']}")
    
    if 'context_statistics' in stats:
        ctx_stats = stats['context_statistics']
        print(f"\nContext filtering:")
        print(f"  - Total conversations: {ctx_stats['total_conversations']}")
        print(f"  - Avg turns per conversation: {ctx_stats['avg_turns_per_conversation']:.1f}")
        print(f"  - Cached embeddings: {ctx_stats['cached_context_embeddings']}")
    
    if 'pca_statistics' in stats:
        pca_stats = stats['pca_statistics']
        if pca_stats['model_loaded']:
            print(f"\nPCA compression:")
            print(f"  - Compression ratio: {pca_stats['compression_ratio']:.1f}x")
            print(f"  - Explained variance: {pca_stats['explained_variance']:.2%}")
            print(f"  - Total compressions: {pca_stats['total_compressions']}")
        else:
            print(f"\nPCA compression: Model not yet trained (need {pca_stats['training_threshold']} samples)")
    
    if 'tau_statistics' in stats:
        tau_stats = stats['tau_statistics']
        print(f"\nFederated τ-tuning:")
        print(f"  - Current threshold: {tau_stats['current_threshold']:.3f}")
        print(f"  - Total queries processed: {tau_stats['total_queries']}")
        agg_stats = tau_stats.get('aggregator_statistics', {})
        print(f"  - Federated aggregations: {agg_stats.get('total_aggregations', 0)}")
    
    print(f"\nDemo completed! Enhanced cache is ready for production use.")
    return cache

def demo_benchmark_workflow():
    """Demonstrate the benchmark workflow."""
    print("\n" + "="*50)
    print("Demo: Benchmark Workflow")
    print("="*50)
    
    print("\nTo run a complete benchmark:")
    print("1. Generate queries:")
    print("   python -m benchmark.generate_queries --output data/queries.json --count 1000")
    
    print("\n2. Run benchmarks:")
    print("   python -m benchmark.benchmark_runner --queries data/queries.json --output data/results.json")
    
    print("\n3. Analyze results:")
    print("   python -m benchmark.analyze_results --results data/results.json --output data/analysis/")
    
    print("\nOr use Docker:")
    print("   docker-compose run enhanced-gptcache generate-queries --output data/queries.json --count 500")
    print("   docker-compose run enhanced-gptcache benchmark --queries data/queries.json --output data/results.json")
    print("   docker-compose run enhanced-gptcache analyze --results data/results.json --output data/analysis/")

if __name__ == "__main__":
    try:
        # Run the basic demo
        cache = demo_basic_usage()
        
        # Show benchmark workflow
        demo_benchmark_workflow()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"The Enhanced GPTCache project is ready for research and production use.")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This may be due to missing dependencies or network issues.")
        print("Try running: pip install -e .")
