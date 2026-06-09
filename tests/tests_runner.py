"""Test runner script that discovers and executes all tests in the tests/ folder.

This script automatically discovers all test modules in the current directory
(excluding itself) and runs them using unittest.
"""

import unittest               # Unit testing framework
import sys                    # System module for path manipulation
from pathlib import Path      # Object-oriented filesystem path handling

def discover_and_run_tests():
    """Discover and run all test files in the tests/ directory except tests_runner.py."""
    
    # Get the directory where this script is located
    tests_dir = Path(__file__).parent
    
    # Create a test loader and discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover all test files matching pattern 'test_*.py' or '*_test.py'
    for test_file in sorted(tests_dir.glob('test_*.py')):
        # Skip the runner script itself
        if test_file.name == 'tests_runner.py':
            continue
        
        # Load tests from the file
        module_name = test_file.stem
        try:
            module = __import__(module_name)
            module_tests = loader.loadTestsFromModule(module)
            suite.addTests(module_tests)
            print(f"✓ Loaded tests from {test_file.name}")
        except Exception as e:
            print(f"✗ Failed to load {test_file.name}: {e}", file=sys.stderr)
    
    # Also check for files matching '*_test.py' pattern
    for test_file in sorted(tests_dir.glob('*_test.py')):
        if test_file.name == 'tests_runner.py':
            continue
        
        module_name = test_file.stem
        try:
            module = __import__(module_name)
            module_tests = loader.loadTestsFromModule(module)
            suite.addTests(module_tests)
            if module_name not in [t for t in dir()]:
                print(f"✓ Loaded tests from {test_file.name}")
        except Exception as e:
            print(f"✗ Failed to load {test_file.name}: {e}", file=sys.stderr)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(discover_and_run_tests())
