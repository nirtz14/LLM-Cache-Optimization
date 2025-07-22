#!/usr/bin/env python3
"""
Simple test for your enhanced cache.
Run this to see if everything works.
"""

import sys
from pathlib import Path

# Add both the project root and src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def test_config():
    """Test if configuration loads properly."""
    print("📋 Testing configuration...")
    try:
        from utils.config import get_config
        config = get_config()
        print(f"  ✅ Config loaded")
        print(f"  📊 Similarity threshold: {config.cache.similarity_threshold}")
        print(f"  📊 Embedding model: {config.context.embedding_model}")
        return True
    except Exception as e:
        print(f"  ❌ Config failed: {e}")
        return False

def test_basic_imports():
    """Test if we can import the required packages."""
    print("\n📦 Testing basic imports...")
    
    imports = [
        ("sentence_transformers", "from sentence_transformers import SentenceTransformer"),
        ("sklearn", "from sklearn.decomposition import PCA"),
        ("gptcache", "import gptcache"),
        ("numpy", "import numpy as np"),
    ]
    
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            return False
    
    return True

def test_sentence_transformer():
    """Test sentence transformer works."""
    print("\n🤖 Testing sentence transformer...")
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("test sentence")
        
        print(f"  ✅ Model loaded, embedding shape: {embedding.shape}")
        return True
    except Exception as e:
        print(f"  ❌ Sentence transformer failed: {e}")
        return False

def test_enhanced_cache():
    """Test your enhanced cache."""
    print("\n🚀 Testing Enhanced Cache...")
    
    try:
        # Import your cache using absolute imports
        from src.cache.enhanced_cache import create_enhanced_cache
        print("  ✅ Cache import successful")
        
        # Create cache with minimal features first
        cache = create_enhanced_cache(
            embedding_model='all-MiniLM-L6-v2',
            enable_context=False,  # Disable complex features for now
            enable_pca=False,
            enable_tau=False
        )
        print("  ✅ Cache created")
        
        # Test storing something
        test_query = "What is Python?"
        test_response = "Python is a programming language."
        
        cache.set(test_query, test_response)
        print("  ✅ Cache.set() works")
        
        # Test retrieving it
        result = cache.query(test_query)
        
        if result['cache_hit'] and result['response'] == test_response:
            print("  ✅ Cache.query() works - found exact match!")
            return True
        elif result['cache_hit']:
            print(f"  ⚠️  Cache hit but response mismatch:")
            print(f"     Expected: {test_response}")
            print(f"     Got: {result['response']}")
            return False
        else:
            print(f"  ⚠️  No cache hit, trying different query...")
            
            # Try a similar query
            result2 = cache.query("Tell me about Python")
            print(f"     Similar query result: hit={result2['cache_hit']}, response={result2.get('response', 'None')}")
            
            if result2['cache_hit']:
                print("  ✅ Cache works with similar queries!")
                return True
            else:
                print("  ❌ Cache not finding matches")
                return False
            
    except Exception as e:
        print(f"  ❌ Enhanced cache failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stats():
    """Test cache statistics."""
    print("\n📊 Testing cache statistics...")
    
    try:
        from src.cache.enhanced_cache import create_enhanced_cache
        
        cache = create_enhanced_cache(enable_context=False, enable_pca=False, enable_tau=False)
        
        # Add some data
        cache.set("Question 1", "Answer 1")
        cache.set("Question 2", "Answer 2")
        
        # Query something (should be a hit)
        cache.query("Question 1")
        # Query something else (might be a miss)
        cache.query("Question 3")
        
        stats = cache.get_stats()
        print(f"  ✅ Stats retrieved:")
        print(f"     Total queries: {stats['cache_statistics']['total_queries']}")
        print(f"     Cache hits: {stats['cache_statistics']['cache_hits']}")
        print(f"     Hit rate: {stats['cache_statistics']['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stats test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Enhanced GPTCache Step by Step\n")
    
    tests = [
        ("Configuration", test_config),
        ("Basic Imports", test_basic_imports),
        ("Sentence Transformer", test_sentence_transformer),
        ("Enhanced Cache", test_enhanced_cache),
        ("Cache Statistics", test_stats),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
        if not results[name]:
            print(f"\n❌ Test '{name}' failed. Stopping here.")
            break
    
    print(f"\n" + "="*50)
    print("📊 FINAL RESULTS")
    print("="*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 SUCCESS! Your cache is working!")
        print(f"\n🔧 Next steps:")
        print(f"  1. Try running: python -m benchmark.generate_queries --count 10")
        print(f"  2. Enable advanced features (PCA, context, tau)")
        print(f"  3. Run full benchmarks")
    else:
        print(f"\n🔧 Fix the failing tests above, then re-run this script.")

if __name__ == "__main__":
    main()