# test_enhanced_features_working.py
#!/usr/bin/env python3
"""Test enhanced features that actually work."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.cache.enhanced_cache import create_enhanced_cache

def test_features_actually_working():
    """Test that features work even in fallback mode."""
    print("🚀 Testing Enhanced Features (Working Implementation)\n")
    
    # Test different variants
    variants = [
        ("Baseline", {"enable_pca": False, "enable_context": False, "enable_tau": False}),
        ("PCA Only", {"enable_pca": True, "enable_context": False, "enable_tau": False}),
        ("Context Only", {"enable_pca": False, "enable_context": True, "enable_tau": False}),
        ("Tau Only", {"enable_pca": False, "enable_context": False, "enable_tau": True}),
        ("All Features", {"enable_pca": True, "enable_context": True, "enable_tau": True}),
    ]
    
    results = {}
    
    for variant_name, config in variants:
        print(f"\n=== Testing {variant_name} ===")
        
        try:
            # Create cache with specific config
            cache = create_enhanced_cache(**config)
            
            # Check what features are actually enabled
            print(f"Requested: PCA={config['enable_pca']}, Context={config['enable_context']}, Tau={config['enable_tau']}")
            print(f"Actual: PCA={cache.enable_pca}, Context={cache.enable_context}, Tau={cache.enable_tau}")
            
            # Test basic functionality
            cache.set("What is AI?", "AI is artificial intelligence", conversation_id="user1")
            
            # Test 1: Same query, same user (should hit)
            result1 = cache.query("What is AI?", conversation_id="user1")
            
            # Test 2: Same query, different user (context filtering should affect this)
            result2 = cache.query("What is AI?", conversation_id="user2")
            
            # Test 3: Similar query (PCA/tau should affect this)
            result3 = cache.query("What is artificial intelligence?", conversation_id="user1")
            
            hit_rate = sum([r['cache_hit'] for r in [result1, result2, result3]]) / 3
            
            results[variant_name] = {
                'hit_rate': hit_rate,
                'same_user_hit': result1['cache_hit'],
                'diff_user_hit': result2['cache_hit'],
                'similar_query_hit': result3['cache_hit'],
                'features_enabled': (cache.enable_pca, cache.enable_context, cache.enable_tau)
            }
            
            print(f"Results:")
            print(f"  Same user hit: {result1['cache_hit']}")
            print(f"  Different user hit: {result2['cache_hit']}")
            print(f"  Similar query hit: {result3['cache_hit']}")
            print(f"  Overall hit rate: {hit_rate:.1%}")
            
        except Exception as e:
            print(f"❌ {variant_name} failed: {e}")
            results[variant_name] = {'error': str(e)}
    
    # Analyze results
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    for variant_name, result in results.items():
        if 'error' in result:
            print(f"{variant_name:15}: ❌ FAILED - {result['error']}")
        else:
            print(f"{variant_name:15}: {result['hit_rate']:.1%} hit rate - Features: {result['features_enabled']}")
    
    # Check if we see differences
    hit_rates = [r['hit_rate'] for r in results.values() if 'hit_rate' in r]
    if len(set(hit_rates)) > 1:
        print(f"\n✅ SUCCESS: Different variants show different behavior!")
    else:
        print(f"\n⚠️  All variants show same behavior - features may not be working")
    
    return results

if __name__ == "__main__":
    test_features_actually_working()