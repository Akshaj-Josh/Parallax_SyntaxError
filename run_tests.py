"""
TRUSTLENS Deepfake Backend Test Suite Runner.
Executes all unit tests and outputs a structured report.
"""

import sys
import os
import unittest
from pathlib import Path

# Configure utf-8 encoding for Windows console compatibility
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_all_tests():
    print("=================================================================")
    print("[TEST SUITE] RUNNING TRUSTLENS DEEPFAKE BACKEND TEST SUITE")
    print("=================================================================\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add tests
    tests_dir = Path(__file__).resolve().parent / "tests"
    discovered = loader.discover(str(tests_dir), pattern="test_*.py")
    suite.addTests(discovered)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n=================================================================")
    if result.wasSuccessful():
        print(f"[SUCCESS] ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print("=================================================================\n")
        return 0
    else:
        print(f"[FAILURE] {len(result.failures)} FAILURES, {len(result.errors)} ERRORS out of {result.testsRun} tests.")
        print("=================================================================\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
