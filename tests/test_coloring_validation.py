#!/usr/bin/env python3
"""
Test script for the new Coloring sequence validation logic.
This script tests the validate_coloring_sequence_pattern function with various scenarios.
"""

import pandas as pd
import sys
import os

# Add the current directory to the path so we can import process_csv_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from process_csv_data import (
        validate_coloring_sequence_pattern,
        validate_coloring_sequence_pattern_by_touchdata_id,
        validate_coloring_sequences_by_touchdata_id
    )
    print("Successfully imported validation functions")
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

def create_test_sequence(touch_phases, times=None):
    """Create a test sequence DataFrame with the given touchPhase values."""
    if times is None:
        times = list(range(len(touch_phases)))

    return pd.DataFrame({
        'touchPhase': touch_phases,
        'time': times,
        'fingerId': [1] * len(touch_phases),
        'seqId': [1] * len(touch_phases)
    })

def test_valid_sequences():
    """Test sequences that should be considered valid."""
    print("\n=== Testing Valid Sequences ===")

    # Test 1: Simple valid sequence
    seq1 = create_test_sequence(['Began', 'Moved', 'Ended'])
    result1 = validate_coloring_sequence_pattern(seq1)
    print(f"Test 1 - Simple valid sequence: {result1} (expected: True)")

    # Test 2: Valid sequence with multiple Moved events
    seq2 = create_test_sequence(['Began', 'Moved', 'Moved', 'Moved', 'Ended'])
    result2 = validate_coloring_sequence_pattern(seq2)
    print(f"Test 2 - Multiple Moved events: {result2} (expected: True)")

    # Test 3: Valid sequence with Stationary events
    seq3 = create_test_sequence(['Began', 'Stationary', 'Stationary', 'Ended'])
    result3 = validate_coloring_sequence_pattern(seq3)
    print(f"Test 3 - Stationary events: {result3} (expected: True)")

    # Test 4: Valid sequence with mixed Moved and Stationary
    seq4 = create_test_sequence(['Began', 'Moved', 'Stationary', 'Moved', 'Ended'])
    result4 = validate_coloring_sequence_pattern(seq4)
    print(f"Test 4 - Mixed Moved/Stationary: {result4} (expected: True)")

    # Test 5: Valid sequence with one Canceled at the end
    seq5 = create_test_sequence(['Began', 'Moved', 'Stationary', 'Canceled', 'Ended'])
    result5 = validate_coloring_sequence_pattern(seq5)
    print(f"Test 5 - With Canceled at end: {result5} (expected: True)")

    # Test 6: Minimal valid sequence (just Began and Ended)
    seq6 = create_test_sequence(['Began', 'Ended'])
    result6 = validate_coloring_sequence_pattern(seq6)
    print(f"Test 6 - Minimal sequence: {result6} (expected: True)")

    return all([result1, result2, result3, result4, result5, result6])

def test_invalid_sequences():
    """Test sequences that should be considered invalid."""
    print("\n=== Testing Invalid Sequences ===")

    # Test 1: Missing Began
    seq1 = create_test_sequence(['Moved', 'Ended'])
    result1 = validate_coloring_sequence_pattern(seq1)
    print(f"Test 1 - Missing Began: {result1} (expected: False)")

    # Test 2: Missing Ended
    seq2 = create_test_sequence(['Began', 'Moved'])
    result2 = validate_coloring_sequence_pattern(seq2)
    print(f"Test 2 - Missing Ended: {result2} (expected: False)")

    # Test 3: Multiple Canceled events
    seq3 = create_test_sequence(['Began', 'Moved', 'Canceled', 'Canceled', 'Ended'])
    result3 = validate_coloring_sequence_pattern(seq3)
    print(f"Test 3 - Multiple Canceled: {result3} (expected: False)")

    # Test 4: Canceled not at the end
    seq4 = create_test_sequence(['Began', 'Canceled', 'Moved', 'Ended'])
    result4 = validate_coloring_sequence_pattern(seq4)
    print(f"Test 4 - Canceled not at end: {result4} (expected: False)")

    # Test 5: Invalid touchPhase in middle
    seq5 = create_test_sequence(['Began', 'Moved', 'InvalidPhase', 'Ended'])
    result5 = validate_coloring_sequence_pattern(seq5)
    print(f"Test 5 - Invalid touchPhase: {result5} (expected: False)")

    # Test 6: Empty sequence
    seq6 = create_test_sequence([])
    result6 = validate_coloring_sequence_pattern(seq6)
    print(f"Test 6 - Empty sequence: {result6} (expected: False)")

    # Test 7: Wrong start phase
    seq7 = create_test_sequence(['Moved', 'Moved', 'Ended'])
    result7 = validate_coloring_sequence_pattern(seq7)
    print(f"Test 7 - Wrong start phase: {result7} (expected: False)")

    # Test 8: Wrong end phase
    seq8 = create_test_sequence(['Began', 'Moved', 'Moved'])
    result8 = validate_coloring_sequence_pattern(seq8)
    print(f"Test 8 - Wrong end phase: {result8} (expected: False)")

    return all([not result1, not result2, not result3, not result4, not result5, not result6, not result7, not result8])

def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n=== Testing Edge Cases ===")

    # Test 1: Out of order times (function should sort by time)
    # When sorted by time [1,2,3], the sequence should be ['Began', 'Ended', 'Moved'] which is invalid
    # Let's create a valid sequence that's out of order: ['Moved', 'Began', 'Ended'] with times [2,1,3]
    seq1 = create_test_sequence(['Moved', 'Began', 'Ended'], times=[2, 1, 3])
    result1 = validate_coloring_sequence_pattern(seq1)
    print(f"Test 1 - Out of order times (should sort to valid): {result1} (expected: True)")

    # Test 2: Single event
    seq2 = create_test_sequence(['Began'])
    result2 = validate_coloring_sequence_pattern(seq2)
    print(f"Test 2 - Single event: {result2} (expected: False)")

    return result1 and not result2

def main():
    """Run all tests."""
    print("Testing Coloring Sequence Validation Logic")
    print("=" * 50)

    valid_tests_passed = test_valid_sequences()
    invalid_tests_passed = test_invalid_sequences()
    edge_tests_passed = test_edge_cases()

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Valid sequences tests: {'PASSED' if valid_tests_passed else 'FAILED'}")
    print(f"Invalid sequences tests: {'PASSED' if invalid_tests_passed else 'FAILED'}")
    print(f"Edge cases tests: {'PASSED' if edge_tests_passed else 'FAILED'}")

    all_passed = valid_tests_passed and invalid_tests_passed and edge_tests_passed
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
