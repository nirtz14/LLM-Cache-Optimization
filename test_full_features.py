#!/usr/bin/env python3
"""
Proper test for enhanced features with scenarios designed to show differences.
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

def test_context_filtering_properly():
    """Test context filtering with proper different conversations."""
    print("🎯 TESTING CONTEXT FILTERING PROPERLY")
    print("=" * 38)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create baseline and context-enabled caches
    baseline_cache = create_enhanced_cache(
        enable_context=False, enable_pca=False, enable_tau=False
    )
    context_cache = create_enhanced_cache(
        enable_context=True, enable_pca=False, enable_tau=False
    )
    
    # Store queries in DIFFERENT conversations
    programming_queries = [
        ("What is Python?", "Python is a programming language"),
        ("How do I use functions?", "Functions are defined with def"),
        ("What are variables?", "Variables store data values")
    ]
    
    cooking_queries = [
        ("How do I cook pasta?", "Boil water and add pasta"),
        ("What ingredients for sauce?", "Tomatoes, garlic, herbs"),
        ("How long to cook?", "About 8-10 minutes")
    ]
    
    print("  Setting up different conversation contexts...")
    
    # Store programming queries in "programming" conversation
    for query, response in programming_queries:
        baseline_cache.set(query, response, conversation_id="programming")
        context_cache.set(query, response, conversation_id="programming")
    
    # Store cooking queries in "cooking" conversation  
    for query, response in cooking_queries:
        baseline_cache.set(query, response, conversation_id="cooking")
        context_cache.set(query, response, conversation_id="cooking")
    
    # Test cross-conversation queries
    test_cases = [
        # Same conversation (should hit in both)
        ("Tell me about Python", "programming", "same_context"),
        ("Python programming help", "programming", "same_context"),
        
        # Different conversation context (context cache should filter)
        ("Tell me about Python", "cooking", "different_context"),
        ("Python programming help", "cooking", "different_context"),
        
        # Cooking context queries
        ("Pasta cooking tips", "cooking", "same_context"),
        ("Pasta cooking tips", "programming", "different_context"),
    ]
    
    baseline_hits = 0
    context_hits = 0
    total_tests = len(test_cases)
    
    print("  Testing cross-conversation queries...")
    for query, conv_id, scenario in test_cases:
        # Test baseline cache
        baseline_result = baseline_cache.query(query, conversation_id=conv_id)
        baseline_hit = baseline_result['cache_hit']
        
        # Test context cache
        context_result = context_cache.query(query, conversation_id=conv_id)
        context_hit = context_result['cache_hit']
        
        if baseline_hit:
            baseline_hits += 1
        if context_hit:
            context_hits += 1
        
        baseline_status = "✅" if baseline_hit else "❌"
        context_status = "✅" if context_hit else "❌"
        
        print(f"    {scenario:15} | Baseline: {baseline_status} | Context: {context_status} | '{query[:25]}...' in {conv_id}")
    
    baseline_rate = baseline_hits / total_tests
    context_rate = context_hits / total_tests
    
    print(f"\n  📊 Results:")
    print(f"    Baseline hit rate: {baseline_rate:.1%} ({baseline_hits}/{total_tests})")
    print(f"    Context hit rate:  {context_rate:.1%} ({context_hits}/{total_tests})")
    
    if context_rate < baseline_rate:
        print(f"  ✅ Context filtering working! Reduced hits from {baseline_rate:.1%} to {context_rate:.1%}")
        return True
    else:
        print(f"  ⚠️  Context filtering not working as expected")
        return False

def test_pca_compression_with_training():
    """Test PCA compression by forcing training and checking behavior."""
    print(f"\n🔬 TESTING PCA COMPRESSION WITH PROPER TRAINING")
    print("=" * 45)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create caches
    baseline_cache = create_enhanced_cache(
        enable_context=False, enable_pca=False, enable_tau=False
    )
    pca_cache = create_enhanced_cache(
        enable_context=False, enable_pca=True, enable_tau=False
    )
    
    print("  Generating enough data to trigger PCA training...")
    
    # Generate many diverse queries to trigger PCA training
    training_queries = []
    for i in range(50):  # Generate 50 diverse queries
        topics = ["Python", "JavaScript", "databases", "machine learning", "web development"]
        actions = ["learn", "install", "use", "debug", "optimize"]
        objects = ["functions", "variables", "classes", "modules", "frameworks"]
        
        topic = topics[i % len(topics)]
        action = actions[i % len(actions)]
        obj = objects[i % len(objects)]
        
        query = f"How do I {action} {topic} {obj}?"
        response = f"To {action} {topic} {obj}, you need to..."
        training_queries.append((query, response))
    
    # Store training data in both caches
    print("  Storing training data...")
    for i, (query, response) in enumerate(training_queries):
        baseline_cache.set(query, response, conversation_id="training")
        pca_cache.set(query, response, conversation_id="training")
        
        if (i + 1) % 10 == 0:
            print(f"    Stored {i+1}/50 training queries")
    
    # Check if PCA was trained
    if hasattr(pca_cache, 'pca_wrapper') and pca_cache.pca_wrapper:
        pca_stats = pca_cache.pca_wrapper.get_compression_stats()
        print(f"  📊 PCA Stats: {pca_stats}")
        
        if pca_stats.get('model_loaded', False):
            print("  ✅ PCA model trained and loaded!")
        else:
            print("  ⚠️  PCA model not trained yet")
    
    # Test with similar queries
    test_queries = [
        "How do I learn Python programming?",  # Similar to training data
        "JavaScript learning tutorial",       # Similar but different language
        "Database optimization guide",        # Similar but different topic
        "Machine learning basics",           # Similar structure
        "Web development setup"              # Similar structure
    ]
    
    baseline_hits = 0
    pca_hits = 0
    
    print("  Testing similar queries...")
    for query in test_queries:
        baseline_result = baseline_cache.query(query, conversation_id="training")
        pca_result = pca_cache.query(query, conversation_id="training")
        
        baseline_hit = baseline_result['cache_hit']
        pca_hit = pca_result['cache_hit']
        
        if baseline_hit:
            baseline_hits += 1
        if pca_hit:
            pca_hits += 1
        
        baseline_status = "✅" if baseline_hit else "❌"
        pca_status = "✅" if pca_hit else "❌"
        
        print(f"    Baseline: {baseline_status} | PCA: {pca_status} | '{query[:30]}...'")
    
    baseline_rate = baseline_hits / len(test_queries)
    pca_rate = pca_hits / len(test_queries)
    
    print(f"\n  📊 Results:")
    print(f"    Baseline hit rate: {baseline_rate:.1%}")
    print(f"    PCA hit rate:      {pca_rate:.1%}")
    
    # PCA might have different hit rate due to compression
    if abs(pca_rate - baseline_rate) > 0.1:  # 10% difference
        print(f"  ✅ PCA compression affecting similarity! Difference: {abs(pca_rate - baseline_rate):.1%}")
        return True
    else:
        print(f"  ⚠️  PCA compression not showing significant effect")
        return False

def test_tau_adaptation():
    """Test tau adaptation with performance feedback."""
    print(f"\n🎚️  TESTING TAU ADAPTATION")
    print("=" * 25)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create tau-enabled cache
    tau_cache = create_enhanced_cache(
        enable_context=False, enable_pca=False, enable_tau=True
    )
    
    # Store some base queries
    base_queries = [
        ("What is programming?", "Programming is writing code"),
        ("How to debug code?", "Use a debugger tool"),
        ("Best coding practices?", "Write clean, readable code")
    ]
    
    print("  Storing base queries...")
    for query, response in base_queries:
        tau_cache.set(query, response, conversation_id="main")
    
    # Get initial threshold
    initial_threshold = tau_cache.get_current_threshold() if hasattr(tau_cache, 'get_current_threshold') else 0.95
    print(f"  Initial threshold: {initial_threshold:.3f}")
    
    # Simulate many queries with performance feedback
    print("  Simulating queries with performance feedback...")
    
    simulation_queries = [
        ("Tell me about programming", True),   # Should hit, good performance
        ("Programming explanation", True),    # Should hit, good performance  
        ("Code debugging tips", True),        # Should hit, good performance
        ("Best practices coding", True),      # Should hit, good performance
        ("Random unrelated query", False),   # Should miss, good performance
        ("Weather today", False),            # Should miss, good performance
        ("Cooking recipes", False),          # Should miss, good performance
    ]
    
    hits = 0
    for query, expected_hit in simulation_queries:
        result = tau_cache.query(query, conversation_id="main")
        actual_hit = result['cache_hit']
        
        if actual_hit:
            hits += 1
            
        # Simulate performance feedback to tau manager
        if hasattr(tau_cache, 'tau_manager') and tau_cache.tau_manager:
            # Get similarity score from result or calculate
            similarity_score = 0.9 if expected_hit else 0.1
            tau_cache.tau_manager.evaluate_threshold(
                query=query,
                similarity_score=similarity_score,
                cache_hit=actual_hit,
                ground_truth_hit=expected_hit
            )
        
        status = "✅" if actual_hit else "❌"
        expected_status = "✅" if expected_hit else "❌"
        print(f"    {status} (expected {expected_status}): '{query[:30]}...'")
    
    # Check if threshold changed
    final_threshold = tau_cache.get_current_threshold() if hasattr(tau_cache, 'get_current_threshold') else initial_threshold
    print(f"  Final threshold: {final_threshold:.3f}")
    
    hit_rate = hits / len(simulation_queries)
    print(f"  Hit rate: {hit_rate:.1%}")
    
    threshold_changed = abs(final_threshold - initial_threshold) > 0.001
    if threshold_changed:
        print(f"  ✅ Tau adaptation working! Threshold changed by {abs(final_threshold - initial_threshold):.3f}")
        return True
    else:
        print(f"  ⚠️  Tau threshold unchanged (may need more iterations)")
        return False

def test_combined_effects():
    """Test how all features work together."""
    print(f"\n🚀 TESTING COMBINED EFFECTS")
    print("=" * 25)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create all variants
    variants = {
        'baseline': create_enhanced_cache(enable_context=False, enable_pca=False, enable_tau=False),
        'full': create_enhanced_cache(enable_context=True, enable_pca=True, enable_tau=True)
    }
    
    # Setup realistic scenario
    scenarios = [
        # Store in programming conversation
        ("What is Python?", "Python is a programming language", "programming"),
        ("How to use functions?", "Use def keyword", "programming"),
        
        # Store in cooking conversation
        ("How to cook pasta?", "Boil water first", "cooking"),
        ("Best pasta sauce?", "Try marinara", "cooking"),
    ]
    
    print("  Setting up realistic scenarios...")
    for query, response, conv_id in scenarios:
        for cache in variants.values():
            cache.set(query, response, conversation_id=conv_id)
    
    # Test with mixed queries
    test_cases = [
        # Same conversation, similar queries (enhanced should hit more)
        ("Tell me about Python", "programming", "similar_same_context"),
        ("Python programming guide", "programming", "similar_same_context"),
        
        # Different conversation, similar queries (enhanced should hit less)
        ("Tell me about Python", "cooking", "similar_different_context"),
        ("Python programming guide", "cooking", "similar_different_context"),
        
        # Completely different (both should miss)
        ("Weather forecast", "programming", "unrelated"),
        ("Stock prices", "cooking", "unrelated"),
    ]
    
    results = {}
    for variant_name, cache in variants.items():
        hits = 0
        total = len(test_cases)
        
        print(f"  Testing {variant_name}...")
        for query, conv_id, scenario in test_cases:
            result = cache.query(query, conversation_id=conv_id)
            if result['cache_hit']:
                hits += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"    {status} {scenario}: '{query[:25]}...' in {conv_id}")
        
        hit_rate = hits / total
        results[variant_name] = hit_rate
        print(f"    📊 {variant_name} hit rate: {hit_rate:.1%}")
    
    # Compare results
    improvement = results['full'] - results['baseline']
    if improvement > 0.05:  # 5% improvement
        print(f"\n  ✅ Combined features improve performance by {improvement:.1%}!")
        return True
    elif improvement < -0.05:  # Features hurt performance
        print(f"\n  ⚠️  Features reduce performance by {abs(improvement):.1%}")
        return False
    else:
        print(f"\n  ➡️  Features show minimal impact ({improvement:+.1%})")
        return False

def main():
    """Run all proper feature tests."""
    print("🚀 PROPER ENHANCED FEATURES TESTING")
    print("=" * 40)
    
    tests = {
        'Context Filtering': test_context_filtering_properly(),
        'PCA Compression': test_pca_compression_with_training(),
        'Tau Adaptation': test_tau_adaptation(),
        'Combined Effects': test_combined_effects()
    }
    
    print(f"\n" + "="*40)
    print("🏁 PROPER FEATURE TEST SUMMARY")
    print("="*40)
    
    for test_name, result in tests.items():
        status = "✅ WORKING" if result else "❌ NEEDS WORK"
        print(f"{test_name:18}: {status}")
    
    working_count = sum(tests.values())
    total_count = len(tests)
    
    print(f"\n📊 Overall: {working_count}/{total_count} features demonstrating clear effects")
    
    if working_count >= 2:
        print("🎉 Good! Multiple features showing measurable differences.")
        print("✅ Ready for full benchmark with realistic query dataset.")
    else:
        print("⚠️  Need to debug feature implementations.")
        print("🔧 Focus on the working features for your report.")
    
    return tests

if __name__ == "__main__":
    main()