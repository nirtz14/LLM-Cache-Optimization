#!/usr/bin/env python3
"""
Quick verification script to test Enhanced GPTCache setup.
"""
import sys
import os

# Add current directory to path at module level
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        from src.utils.config import get_config
        print("✅ Config module imported successfully")
        
        from src.utils.metrics import get_performance_tracker
        print("✅ Metrics module imported successfully")
        
        from src.core.context_similarity import ContextAwareSimilarity
        print("✅ Context similarity module imported successfully")
        
        from src.core.pca_wrapper import PCAEmbeddingWrapper
        print("✅ PCA wrapper module imported successfully")
        
        from src.core.tau_manager import TauManager
        print("✅ Tau manager module imported successfully")
        
        from src.cache.enhanced_cache import create_enhanced_cache
        print("✅ Enhanced cache module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic cache functionality without heavy dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Import at function level to ensure availability
        from src.utils.config import get_config
        from src.utils.metrics import get_performance_tracker, record_cache_request
        
        # Test configuration loading
        config = get_config()
        print(f"✅ Configuration loaded: cache.similarity_threshold = {config.cache.similarity_threshold}")
        
        # Test performance tracker
        tracker = get_performance_tracker()
        print("✅ Performance tracker initialized")
        
        # Test metrics recording (without actual cache operations)
        record_cache_request("test query", 10.5, True)
        stats = tracker.get_current_stats()
        print(f"✅ Metrics recording works: {stats.total_requests} requests tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_without_ml_dependencies():
    """Test components that don't require ML libraries."""
    print("\n🔧 Testing components without ML dependencies...")
    
    try:
        # Test tau manager (simulation only)
        from src.core.tau_manager import TauManager
        
        # Create tau manager with minimal setup
        tau_manager = TauManager(num_users=3, aggregation_frequency=5, initial_tau=0.8)
        print("✅ Tau manager created successfully")
        
        # Test threshold evaluation (simulation)
        threshold = tau_manager.evaluate_threshold("test query", 0.85, True)
        print(f"✅ Threshold evaluation works: threshold = {threshold}")
        
        stats = tau_manager.get_tau_stats()
        print(f"✅ Tau statistics available: {stats['num_users']} simulated users")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Enhanced GPTCache Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Core Components", test_without_ml_dependencies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 50)
    print(f"VERIFICATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("Try installing missing dependencies or use Docker for guaranteed compatibility.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
