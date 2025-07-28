#!/usr/bin/env python3
"""
Test the enhanced features (PCA, Context, Tau) to see why hit rates are identical.
"""

import sys
import time
import os
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def test_feature_differences():
    """Test if different feature combinations actually behave differently."""
    print("🔬 TESTING FEATURE DIFFERENCES")
    print("=" * 35)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create test queries that should show differences
    base_queries = [
        "What is Python?",
        "How do I learn Python?",  # Similar to first
        "Tell me about Python programming",  # Also similar
        "Explain Python to me",  # Very similar
        "What is JavaScript?",  # Different topic
    ]
    
    responses = [
        "Python is a programming language",
        "You can learn Python through tutorials", 
        "Python is great for programming",
        "Python is an easy programming language",
        "JavaScript is a web programming language"
    ]
    
    # Test configurations
    configs = {
        'baseline': {'enable_context': False, 'enable_pca': False, 'enable_tau': False},
        'pca_only': {'enable_context': False, 'enable_pca': True, 'enable_tau': False},
        'context_only': {'enable_context': True, 'enable_pca': False, 'enable_tau': False},
        'full': {'enable_context': True, 'enable_pca': True, 'enable_tau': True}
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n📊 Testing {config_name} configuration...")
        
        try:
            # Create cache
            cache = create_enhanced_cache(**config)
            print(f"  Cache created with: {config}")
            
            # Store the first 3 queries
            print("  Storing base queries...")
            for i in range(3):
                cache.set(base_queries[i], responses[i], conversation_id="test_conv")
                print(f"    Stored: '{base_queries[i][:30]}...'")
            
            # Test all queries for hits
            print("  Testing query retrieval...")
            hits = []
            for i, query in enumerate(base_queries):
                result = cache.query(query, conversation_id="test_conv")
                hit = result['cache_hit']
                similarity = result.get('similarity_score', 'N/A')
                hits.append(hit)
                
                print(f"    Query {i+1}: '{query[:25]}...' -> Hit: {hit}")
            
            # Store results
            hit_count = sum(hits)
            hit_rate = hit_count / len(base_queries)
            
            results[config_name] = {
                'hits': hits,
                'hit_count': hit_count,
                'hit_rate': hit_rate,
                'config': config
            }
            
            print(f"  📈 {config_name} hit rate: {hit_rate:.2%} ({hit_count}/{len(base_queries)})")
            
        except Exception as e:
            print(f"  ❌ {config_name} failed: {e}")
            results[config_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print("=" * 30)
    
    hit_rates = {}
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name:12}: ERROR - {result['error']}")
        else:
            hit_rate = result['hit_rate']
            hit_rates[config_name] = hit_rate
            print(f"{config_name:12}: {hit_rate:.2%} ({result['hit_count']}/{len(base_queries)})")
    
    # Analysis
    if len(set(hit_rates.values())) == 1:
        print(f"\n⚠️  WARNING: All configurations have identical hit rates!")
        print("   This suggests enhanced features aren't affecting cache behavior.")
        return False
    else:
        print(f"\n✅ GOOD: Different configurations show different hit rates!")
        return True

def test_pca_compression_detailed():
    """Test PCA compression in detail."""
    print(f"\n🔬 DETAILED PCA COMPRESSION TEST")
    print("=" * 35)
    
    try:
        from src.core.pca_wrapper import PCAEmbeddingWrapper
        from sentence_transformers import SentenceTransformer
        
        # Create base embedding function
        model = SentenceTransformer('all-MiniLM-L6-v2')
        base_func = lambda text: model.encode(text)
        
        print("  Testing original embedding...")
        test_text = "This is a test sentence for PCA compression"
        original_emb = base_func(test_text)
        print(f"  Original embedding shape: {original_emb.shape}")
        
        # Create PCA wrapper
        print("  Creating PCA wrapper...")
        pca_wrapper = PCAEmbeddingWrapper(
            base_embedding_func=base_func,
            target_dimensions=128,
            auto_train=True,
            training_samples_threshold=10
        )
        
        # Generate training data
        print("  Generating training embeddings...")
        training_texts = [
            f"Training sentence number {i} with unique content about topic {i%3}" 
            for i in range(15)
        ]
        
        embeddings = []
        for i, text in enumerate(training_texts):
            emb = pca_wrapper(text)
            embeddings.append(emb)
            if (i + 1) % 5 == 0:
                print(f"    Generated {i+1}/15 embeddings, current shape: {emb.shape}")
        
        # Test final compression
        final_emb = pca_wrapper(test_text)
        print(f"  Final embedding shape: {final_emb.shape}")
        
        # Check compression stats
        stats = pca_wrapper.get_compression_stats()
        print(f"  📊 PCA Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        # Verify compression
        if final_emb.shape[0] < original_emb.shape[0]:
            compression_ratio = original_emb.shape[0] / final_emb.shape[0]
            print(f"  ✅ PCA compression working! {compression_ratio:.1f}x compression")
            return True
        else:
            print(f"  ⚠️  PCA compression not applied yet")
            return False
            
    except Exception as e:
        print(f"  ❌ PCA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_filtering_detailed():
    """Test context filtering in detail."""
    print(f"\n🎯 DETAILED CONTEXT FILTERING TEST")
    print("=" * 35)
    
    try:
        from src.core.context_similarity import ContextAwareSimilarity
        
        print("  Creating context-aware similarity...")
        context_sim = ContextAwareSimilarity(
            embedding_model='all-MiniLM-L6-v2'
        )
        
        # Create different conversation contexts
        print("  Setting up conversation contexts...")
        
        # Conversation 1: Programming
        context_sim.add_conversation_turn("conv1", "I need help with programming", timestamp=time.time())
        context_sim.add_conversation_turn("conv1", "Specifically with Python", timestamp=time.time())
        context_sim.add_conversation_turn("conv1", "How do I use functions?", timestamp=time.time())
        
        # Conversation 2: Cooking
        context_sim.add_conversation_turn("conv2", "I want to cook dinner", timestamp=time.time())
        context_sim.add_conversation_turn("conv2", "Something with pasta", timestamp=time.time())
        context_sim.add_conversation_turn("conv2", "What ingredients do I need?", timestamp=time.time())
        
        print("  Testing context similarity...")
        
        # Test within same conversation (should be high)
        same_context_sim = context_sim._compute_context_similarity(
            'conv1', 'conv1', 
            'What about variables?', 'How do I debug code?'
        )
        
        # Test between different conversations (should be low)
        diff_context_sim = context_sim._compute_context_similarity(
            'conv1', 'conv2',
            'What about variables?', 'What about recipes?'
        )
        
        print(f"  Same conversation similarity: {same_context_sim:.3f}")
        print(f"  Different conversation similarity: {diff_context_sim:.3f}")
        
        # Get context stats
        stats = context_sim.get_context_stats()
        print(f"  📊 Context Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        if same_context_sim > diff_context_sim:
            print(f"  ✅ Context filtering working - same context similarity higher!")
            return True
        else:
            print(f"  ⚠️  Context filtering not working as expected")
            return False
            
    except Exception as e:
        print(f"  ❌ Context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tau_manager_detailed():
    """Test tau manager in detail."""
    print(f"\n🎚️  DETAILED TAU MANAGER TEST")
    print("=" * 30)
    
    try:
        from src.core.tau_manager import TauManager
        
        print("  Creating tau manager...")
        tau_manager = TauManager(
            num_users=5,
            aggregation_frequency=10,
            learning_rate=0.01,
            initial_tau=0.8
        )
        
        print(f"  Initial threshold: {tau_manager.get_current_threshold():.3f}")
        
        # Simulate some queries with performance feedback
        print("  Simulating queries with performance feedback...")
        
        queries = [
            ("What is Python?", 0.95, True, True),      # High similarity, should hit
            ("Tell me about Python", 0.85, True, True), # Medium similarity, should hit
            ("Python programming", 0.75, False, True),  # Lower similarity, might miss
            ("JavaScript help", 0.3, False, False),     # Low similarity, should miss
            ("Random question", 0.1, False, False),     # Very low, should miss
        ]
        
        for i, (query, similarity, cache_hit, ground_truth) in enumerate(queries):
            threshold = tau_manager.evaluate_threshold(
                query=query,
                similarity_score=similarity,
                cache_hit=cache_hit,
                ground_truth_hit=ground_truth
            )
            print(f"    Query {i+1}: sim={similarity:.2f}, threshold={threshold:.3f}")
        
        # Get tau statistics
        stats = tau_manager.get_tau_stats()
        print(f"  📊 Tau Statistics:")
        print(f"    Current threshold: {stats['current_threshold']:.3f}")
        print(f"    Total queries: {stats['total_queries']}")
        print(f"    Number of users: {stats['num_users']}")
        print(f"    Enabled: {stats['enabled']}")
        
        final_threshold = tau_manager.get_current_threshold()
        initial_threshold = 0.8
        
        if abs(final_threshold - initial_threshold) > 0.001:
            print(f"  ✅ Tau tuning working - threshold changed from {initial_threshold:.3f} to {final_threshold:.3f}!")
            return True
        else:
            print(f"  ⚠️  Tau threshold unchanged (might need more queries)")
            return False
            
    except Exception as e:
        print(f"  ❌ Tau test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all enhanced feature tests."""
    print("🚀 ENHANCED FEATURES TESTING SUITE")
    print("=" * 45)
    
    # Run tests
    tests = {
        'Feature Differences': test_feature_differences(),
        'PCA Compression': test_pca_compression_detailed(),
        'Context Filtering': test_context_filtering_detailed(), 
        'Tau Management': test_tau_manager_detailed()
    }
    
    # Summary
    print(f"\n" + "="*45)
    print("🏁 ENHANCED FEATURES TEST SUMMARY")
    print("="*45)
    
    for test_name, result in tests.items():
        status = "✅ WORKING" if result else "❌ NOT WORKING"
        print(f"{test_name:20}: {status}")
    
    working_features = sum(tests.values())
    total_features = len(tests)
    
    print(f"\n📊 OVERALL RESULT: {working_features}/{total_features} features working")
    
    if working_features == 0:
        print("\n🚨 CRITICAL: No enhanced features are working!")
        print("   The 15% hit rate is probably just from exact matches in fallback cache.")
    elif working_features == total_features:
        print("\n🎉 EXCELLENT: All enhanced features are working!")
        print("   The identical hit rates suggest a benchmarking issue, not a feature issue.")
    else:
        print(f"\n⚠️  PARTIAL: {working_features} features working, {total_features - working_features} need fixes.")
    
    print(f"\n🎯 NEXT STEPS:")
    if not tests['Feature Differences']:
        print("1. Fix cache configuration - features not affecting behavior")
    if not tests['PCA Compression']:
        print("2. Debug PCA compression training")
    if not tests['Context Filtering']:
        print("3. Fix context similarity computation")
    if not tests['Tau Management']:
        print("4. Debug tau threshold optimization")
    
    if all(tests.values()):
        print("1. All features work! Focus on improving benchmark query generation")
        print("2. Create more realistic query patterns to show differences")

if __name__ == "__main__":
    main()