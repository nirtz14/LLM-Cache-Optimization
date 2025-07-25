#!/usr/bin/env python3
"""
Test cache variants with the fixed similarity threshold to verify different behaviors.
"""

import sys
import time
import os
from pathlib import Path
import yaml

# Add paths
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def update_config_threshold():
    """Update the config file with the new threshold."""
    config_path = project_root / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update threshold
        config['cache']['similarity_threshold'] = 0.95
        config['federated']['initial_tau'] = 0.95
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Updated config.yaml with threshold 0.95")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def test_variants_with_realistic_threshold():
    """Test all variants with realistic similarity threshold."""
    print("🔬 TESTING VARIANTS WITH REALISTIC THRESHOLD")
    print("=" * 45)
    
    from src.cache.enhanced_cache import create_enhanced_cache
    
    # Create diverse test scenarios
    scenarios = {
        'exact_matches': [
            ("What is Python?", "Python is a programming language"),
            ("How do I install Python?", "Download from python.org"),
            ("What are Python lists?", "Lists are ordered collections")
        ],
        'similar_queries': [
            "Tell me about Python",              # Similar to "What is Python?"
            "Python installation guide",        # Similar to "How do I install Python?"
            "Python list tutorial"               # Similar to "What are Python lists?"
        ],
        'different_context': [
            ("I'm cooking dinner", "That sounds delicious"),
            ("What's the weather?", "It's sunny today"),
            ("Stock market update", "Markets are volatile")
        ],
        'test_queries': [
            "Explain Python programming",        # Should hit with context/pca but not baseline
            "Installing Python tutorial",       # Should hit with enhanced features
            "Python list methods",              # Should hit with enhanced features
            "How to cook pasta",                # Should miss (different topic)
            "Weather forecast",                  # Should miss (different topic)
            "Investment advice"                  # Should miss (different topic)
        ]
    }
    
    # Variant configurations
    variants = {
        'baseline': {'enable_context': False, 'enable_pca': False, 'enable_tau': False},
        'context': {'enable_context': True, 'enable_pca': False, 'enable_tau': False},
        'pca': {'enable_context': False, 'enable_pca': True, 'enable_tau': False},
        'tau': {'enable_context': False, 'enable_pca': False, 'enable_tau': True},
        'full': {'enable_context': True, 'enable_pca': True, 'enable_tau': True}
    }
    
    results = {}
    
    for variant_name, config in variants.items():
        print(f"\n--- Testing {variant_name.upper()} variant ---")
        
        try:
            cache = create_enhanced_cache(**config)
            
            # Store exact matches in same conversation
            print("  Storing exact match queries...")
            for query, response in scenarios['exact_matches']:
                cache.set(query, response, conversation_id="main_conv")
            
            # Store different context queries in different conversation
            print("  Storing different context queries...")
            for i, (query, response) in enumerate(scenarios['different_context']):
                cache.set(query, response, conversation_id=f"other_conv_{i}")
            
            # Test all queries
            print("  Testing query retrieval...")
            hits = 0
            total_tests = 0
            
            test_cases = [
                # Exact matches (should always hit)
                *[(q, "main_conv", "exact") for q, _ in scenarios['exact_matches']],
                # Similar queries (enhanced features should hit more)
                *[(q, "main_conv", "similar") for q in scenarios['similar_queries']],
                # Test queries (different hit patterns expected)
                *[(q, "main_conv", "test") for q in scenarios['test_queries']]
            ]
            
            hit_details = []
            for query, conv_id, category in test_cases:
                result = cache.query(query, conversation_id=conv_id)
                is_hit = result['cache_hit']
                
                if is_hit:
                    hits += 1
                    status = "✅"
                else:
                    status = "❌"
                
                hit_details.append((query[:30], category, is_hit))
                print(f"    {status} {category:8}: '{query[:30]}...'")
                total_tests += 1
            
            hit_rate = hits / total_tests if total_tests > 0 else 0
            
            results[variant_name] = {
                'hit_rate': hit_rate,
                'hits': hits,
                'total': total_tests,
                'config': config,
                'details': hit_details
            }
            
            print(f"  📊 {variant_name} hit rate: {hit_rate:.1%} ({hits}/{total_tests})")
            
        except Exception as e:
            print(f"  ❌ {variant_name} failed: {e}")
            results[variant_name] = {'error': str(e)}
    
    return results

def analyze_results(results):
    """Analyze and compare the results."""
    print(f"\n📊 DETAILED ANALYSIS")
    print("=" * 25)
    
    # Extract hit rates
    hit_rates = {}
    for variant, result in results.items():
        if 'error' not in result:
            hit_rates[variant] = result['hit_rate']
    
    # Sort by hit rate
    sorted_variants = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
    
    print("Hit rate ranking:")
    for variant, hit_rate in sorted_variants:
        print(f"  {variant:10}: {hit_rate:.1%}")
    
    # Check for differences
    if len(set(hit_rates.values())) == 1:
        print(f"\n⚠️  WARNING: All variants still have identical hit rates!")
        print("   May need more diverse queries or different thresholds.")
    else:
        print(f"\n✅ SUCCESS: Variants show different hit rates!")
        
        # Expected behavior analysis
        baseline_rate = hit_rates.get('baseline', 0)
        context_rate = hit_rates.get('context', 0) 
        pca_rate = hit_rates.get('pca', 0)
        full_rate = hit_rates.get('full', 0)
        
        print(f"\nExpected vs Actual:")
        print(f"  Baseline: {baseline_rate:.1%} (expected: ~10-15%)")
        
        if context_rate > baseline_rate:
            print(f"  Context: {context_rate:.1%} > Baseline ✅ (context filtering working)")
        else:
            print(f"  Context: {context_rate:.1%} ≤ Baseline ⚠️")
            
        if full_rate >= max(baseline_rate, context_rate):
            print(f"  Full: {full_rate:.1%} ≥ Others ✅ (enhanced features working)")
        else:
            print(f"  Full: {full_rate:.1%} < Others ⚠️")
    
    return len(set(hit_rates.values())) > 1

def main():
    """Run the fixed threshold test."""
    print("🎯 TESTING FIXED SIMILARITY THRESHOLD")
    print("=" * 40)
    
    # Update config first
    config_updated = update_config_threshold()
    if not config_updated:
        print("❌ Cannot proceed without updating config")
        return
    
    # Test variants
    results = test_variants_with_realistic_threshold()
    
    # Analyze results
    success = analyze_results(results)
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if success:
        print("✅ Configuration fixed! Variants show different behaviors.")
        print("🚀 Ready to run full benchmarks with realistic results.")
        print("\nNext steps:")
        print("1. Generate improved query dataset")
        print("2. Run full benchmark suite")
        print("3. Create performance visualizations")
    else:
        print("⚠️  Still need more work on query diversity.")
        print("🔄 Consider generating more diverse query patterns.")
    
    return results

if __name__ == "__main__":
    main()