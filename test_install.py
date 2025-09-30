#!/usr/bin/env python3
"""
Simple test script to verify the package installation and basic functionality.
"""

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import icm_cardiovascular
        print("✓ Package imported successfully")
        
        from icm_cardiovascular import database_creator
        print("✓ Database creator module imported")
        
        from icm_cardiovascular import icm_database
        print("✓ Database viewer module imported")
        
        # Test version info
        print(f"✓ Package version: {icm_cardiovascular.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    dependencies = [
        'dash',
        'dash_bootstrap_components',
        'pandas',
        'numpy',
        'duckdb',
        'plotly'
    ]
    
    all_good = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} is available")
        except ImportError:
            print(f"✗ {dep} is missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("ICM Cardiovascular Package Test")
    print("=" * 40)
    
    import_success = test_imports()
    print()
    
    deps_success = test_dependencies()
    print()
    
    if import_success and deps_success:
        print("✓ All tests passed! Package is ready to use.")
        print()
        print("To run the applications:")
        print("  icm-database-creator    # Create databases")
        print("  icm-database-viewer     # View/analyze databases")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    exit(main())
