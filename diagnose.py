#!/usr/bin/env python3
"""
Diagnostic script to identify issues with the Enhanced GPTCache project.
"""
import sys
import os
import importlib.util

def check_python_version():
    """Check Python version compatibility."""
    print("=== Python Version Check ===")
    print(f"Python version: {sys.version}")
    
    major, minor = sys.version_info[:2]
    if major != 3:
        print("❌ ERROR: Python 3.x required")
        return False
    elif minor < 9:
        print("❌ ERROR: Python 3.9+ required")
        return False
    elif minor > 11:
        print("⚠️  WARNING: Python 3.13 may have compatibility issues with some packages")
        print("   Recommended: Python 3.9-3.11")
        return True
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\n=== Dependency Check ===")
    
    required_packages = [
        'gptcache',
        'sentence_transformers', 
        'sklearn',  # scikit-learn
        'numpy',
        'pandas',
        'tqdm',
        'joblib',
        'yaml'  # pyyaml
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} required packages!")
        print("To install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed")
        return True

def check_project_structure():
    """Check if project files exist."""
    print("\n=== Project Structure Check ===")
    
    required_files = [
        'src/__init__.py',
        'src/utils/__init__.py',
        'src/core/__init__.py', 
        'src/cache/__init__.py',
        'src/utils/config.py',
        'src/utils/metrics.py',
        'src/core/context_similarity.py',
        'src/core/pca_wrapper.py',
        'src/core/tau_manager.py',
        'src/cache/enhanced_cache.py',
        'config.yaml'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files!")
        return False
    else:
        print("\n✅ All required files exist")
        return True

def check_imports():
    """Check if our modules can be imported."""
    print("\n=== Import Check ===")
    
    # Add current directory to Python path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added {current_dir} to Python path")
    
    modules_to_test = [
        'src',
        'src.utils',
        'src.utils.config',
        'src.utils.metrics', 
        'src.core',
        'src.core.context_similarity',
        'src.core.pca_wrapper',
        'src.core.tau_manager',
        'src.cache',
        'src.cache.enhanced_cache'
    ]
    
    import_errors = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module} - ERROR: {str(e)}")
            import_errors.append((module, str(e)))
    
    if import_errors:
        print(f"\n❌ {len(import_errors)} import errors found!")
        print("\nDetailed errors:")
        for module, error in import_errors:
            print(f"  {module}: {error}")
        return False
    else:
        print("\n✅ All modules can be imported")
        return True

def test_basic_functionality():
    """Test basic cache creation."""
    print("\n=== Basic Functionality Test ===")
    
    try:
        # Add current directory to path if not already there
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from src.cache.enhanced_cache import create_enhanced_cache
        
        print("Creating cache with all features disabled...")
        cache = create_enhanced_cache(
            enable_context=False,
            enable_pca=False, 
            enable_tau=False
        )
        print("✅ Basic cache creation successful")
        
        # Test a simple operation
        cache.set("test", "response")
        result = cache.query("test")
        print("✅ Basic cache operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def provide_solutions():
    """Provide solutions for common issues."""
    print("\n=== Recommended Solutions ===")
    
    print("\n1. Install dependencies:")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements.txt")
    
    print("\n2. If you get package conflicts, try:")
    print("   python -m pip install --user -r requirements.txt")
    
    print("\n3. For Python 3.13 compatibility issues:")
    print("   Consider using Python 3.10 or 3.11:")
    print("   - Use conda: conda create -n gptcache python=3.10")
    print("   - Or pyenv: pyenv install 3.10.12")
    
    print("\n4. Alternative: Use Docker:")
    print("   docker-compose up --build")
    print("   docker-compose run enhanced-gptcache shell")
    
    print("\n5. If imports still fail, try:")
    print("   export PYTHONPATH=$PWD:$PYTHONPATH  # Linux/Mac")
    print("   set PYTHONPATH=%CD%;%PYTHONPATH%    # Windows")

def main():
    """Run all diagnostic checks."""
    print("Enhanced GPTCache Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure), 
        ("Imports", check_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} checks")
    
    if passed < len(results):
        provide_solutions()
    else:
        print("\n🎉 All checks passed! You can now run:")
        print("   python demo.py")

if __name__ == "__main__":
    main()
