#!/usr/bin/env python3
"""
Test runner for SFC-LLM test suite.
"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all tests in the test suite."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


def run_unit_tests():
    """Run only unit tests (no integration tests)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add specific test modules
    from tests.test_config import TestConfig
    from tests.test_embedding import TestEmbedding
    
    suite.addTest(loader.loadTestsFromTestCase(TestConfig))
    suite.addTest(loader.loadTestsFromTestCase(TestEmbedding))
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run only integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    from tests.test_integration import TestIntegration, TestVLMIntegration
    
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestVLMIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SFC-LLM tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Set environment for testing
    os.environ["SFC_LLM_LOG_LEVEL"] = "WARNING"
    
    print("=" * 60)
    print(f"Running SFC-LLM {args.type} tests")
    print("=" * 60)
    
    if args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    print("=" * 60)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)