#!/usr/bin/env python3
"""
Test script for the new Touchdata_id-based Coloring sequence validation logic.
This script tests the validate_coloring_sequences_by_touchdata_id function with various scenarios.
"""

import pandas as pd
import sys
import os

# Add the current directory to the path so we can import process_csv_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from process_csv_data import (
        validate_coloring_sequence_pattern_by_touchdata_id,
        validate_coloring_sequences_by_touchdata_id
    )
    print("Successfully imported Touchdata_id validation functions")
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

def create_touchdata_sequence(touchdata_id, touch_phases, event_indices=None):
    """Create a test sequence DataFrame with Touchdata_id and event_index."""
    if event_indices is None:
        event_indices = list(range(len(touch_phases)))
    
    return pd.DataFrame({
        'Touchdata_id': [touchdata_id] * len(touch_phases),
        'touchPhase': touch_phases,
        'event_index': event_indices,
        'time': [i * 100 for i in event_indices],  # Mock time values
        'fingerId': [touchdata_id] * len(touch_phases),  # Usually same as Touchdata_id
        'seqId': [1] * len(touch_phases),
        'x': [100 + i * 10 for i in event_indices],
        'y': [100 + i * 10 for i in event_indices],
        'color': ['Red'] * len(touch_phases),
        'completionPerc': [i * 0.1 for i in event_indices],
        'zone': ['Area1'] * len(touch_phases)
    })

def test_single_touchdata_id_validation():
    """Test validation of individual Touchdata_id sequences."""
    print("\n=== Testing Single Touchdata_id Validation ===")
    
    # Test 1: Valid sequence
    seq1 = create_touchdata_sequence(1, ['Began', 'Moved', 'Ended'], [0, 1, 2])
    result1 = validate_coloring_sequence_pattern_by_touchdata_id(seq1)
    print(f"Test 1 - Valid sequence: {result1} (expected: True)")
    
    # Test 2: Valid sequence with Canceled
    seq2 = create_touchdata_sequence(2, ['Began', 'Moved', 'Canceled', 'Ended'], [0, 1, 2, 3])
    result2 = validate_coloring_sequence_pattern_by_touchdata_id(seq2)
    print(f"Test 2 - Valid with Canceled: {result2} (expected: True)")
    
    # Test 3: Invalid - missing Began
    seq3 = create_touchdata_sequence(3, ['Moved', 'Ended'], [0, 1])
    result3 = validate_coloring_sequence_pattern_by_touchdata_id(seq3)
    print(f"Test 3 - Missing Began: {result3} (expected: False)")
    
    # Test 4: Invalid - multiple Canceled
    seq4 = create_touchdata_sequence(4, ['Began', 'Canceled', 'Canceled', 'Ended'], [0, 1, 2, 3])
    result4 = validate_coloring_sequence_pattern_by_touchdata_id(seq4)
    print(f"Test 4 - Multiple Canceled: {result4} (expected: False)")
    
    # Test 5: Out of order event_index (should be sorted correctly)
    seq5 = create_touchdata_sequence(5, ['Moved', 'Began', 'Ended'], [1, 0, 2])
    result5 = validate_coloring_sequence_pattern_by_touchdata_id(seq5)
    print(f"Test 5 - Out of order events: {result5} (expected: True)")
    
    return all([result1, result2, not result3, not result4, result5])

def test_multiple_touchdata_id_validation():
    """Test validation of multiple Touchdata_id sequences in one DataFrame."""
    print("\n=== Testing Multiple Touchdata_id Validation ===")
    
    # Create a DataFrame with multiple Touchdata_id sequences
    data_frames = []
    
    # Touchdata_id 1: Valid sequence
    data_frames.append(create_touchdata_sequence(1, ['Began', 'Moved', 'Stationary', 'Ended'], [0, 1, 2, 3]))
    
    # Touchdata_id 2: Valid sequence with Canceled
    data_frames.append(create_touchdata_sequence(2, ['Began', 'Moved', 'Canceled', 'Ended'], [0, 1, 2, 3]))
    
    # Touchdata_id 3: Invalid - missing Began
    data_frames.append(create_touchdata_sequence(3, ['Moved', 'Ended'], [0, 1]))
    
    # Touchdata_id 4: Invalid - missing Ended
    data_frames.append(create_touchdata_sequence(4, ['Began', 'Moved'], [0, 1]))
    
    # Touchdata_id 5: Invalid - Canceled not at end
    data_frames.append(create_touchdata_sequence(5, ['Began', 'Canceled', 'Moved', 'Ended'], [0, 1, 2, 3]))
    
    # Combine all sequences
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    # Validate all sequences
    validation_results = validate_coloring_sequences_by_touchdata_id(combined_df)
    
    print(f"Validation results: {validation_results}")
    
    # Check expected results
    expected = {1: True, 2: True, 3: False, 4: False, 5: False}
    
    all_correct = True
    for touchdata_id, expected_result in expected.items():
        actual_result = validation_results.get(touchdata_id, False)
        is_correct = actual_result == expected_result
        print(f"Touchdata_id {touchdata_id}: Expected {expected_result}, Got {actual_result} - {'✓' if is_correct else '✗'}")
        if not is_correct:
            all_correct = False
    
    return all_correct

def test_edge_cases():
    """Test edge cases for Touchdata_id validation."""
    print("\n=== Testing Edge Cases ===")
    
    # Test 1: Empty DataFrame
    empty_df = pd.DataFrame()
    result1 = validate_coloring_sequences_by_touchdata_id(empty_df)
    print(f"Test 1 - Empty DataFrame: {result1} (expected: empty dict)")
    
    # Test 2: DataFrame without required columns
    df_no_cols = pd.DataFrame({'other_col': [1, 2, 3]})
    result2 = validate_coloring_sequences_by_touchdata_id(df_no_cols)
    print(f"Test 2 - Missing columns: {result2} (expected: empty dict)")
    
    # Test 3: Single event sequence
    single_event = create_touchdata_sequence(1, ['Began'], [0])
    result3 = validate_coloring_sequence_pattern_by_touchdata_id(single_event)
    print(f"Test 3 - Single event: {result3} (expected: False)")
    
    # Test 4: Minimal valid sequence
    minimal_valid = create_touchdata_sequence(1, ['Began', 'Ended'], [0, 1])
    result4 = validate_coloring_sequence_pattern_by_touchdata_id(minimal_valid)
    print(f"Test 4 - Minimal valid: {result4} (expected: True)")
    
    return (len(result1) == 0 and len(result2) == 0 and 
            not result3 and result4)

def test_real_world_scenario():
    """Test a realistic scenario with mixed valid and invalid sequences."""
    print("\n=== Testing Real-World Scenario ===")
    
    # Create realistic touch sequences
    data_frames = []
    
    # User 1: Complete drawing sequence
    data_frames.append(create_touchdata_sequence(
        1, ['Began', 'Moved', 'Moved', 'Moved', 'Stationary', 'Moved', 'Ended'], 
        [0, 1, 2, 3, 4, 5, 6]
    ))
    
    # User 2: Started but canceled
    data_frames.append(create_touchdata_sequence(
        2, ['Began', 'Moved', 'Moved', 'Canceled', 'Ended'], 
        [0, 1, 2, 3, 4]
    ))
    
    # User 3: Incomplete sequence (app crash?)
    data_frames.append(create_touchdata_sequence(
        3, ['Began', 'Moved', 'Moved'], 
        [0, 1, 2]
    ))
    
    # User 4: Invalid start (data corruption?)
    data_frames.append(create_touchdata_sequence(
        4, ['Stationary', 'Moved', 'Ended'], 
        [0, 1, 2]
    ))
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    validation_results = validate_coloring_sequences_by_touchdata_id(combined_df)
    
    print("Real-world validation results:")
    for touchdata_id, is_valid in validation_results.items():
        status = "Valid" if is_valid else "Invalid"
        print(f"  User {touchdata_id}: {status}")
    
    # Expected: User 1 and 2 should be valid, User 3 and 4 should be invalid
    expected_valid = {1, 2}
    expected_invalid = {3, 4}
    
    actual_valid = {tid for tid, valid in validation_results.items() if valid}
    actual_invalid = {tid for tid, valid in validation_results.items() if not valid}
    
    is_correct = (actual_valid == expected_valid and actual_invalid == expected_invalid)
    print(f"Real-world test: {'✓ PASSED' if is_correct else '✗ FAILED'}")
    
    return is_correct

def main():
    """Run all tests."""
    print("Testing Touchdata_id-based Coloring Sequence Validation")
    print("=" * 60)
    
    single_tests_passed = test_single_touchdata_id_validation()
    multiple_tests_passed = test_multiple_touchdata_id_validation()
    edge_tests_passed = test_edge_cases()
    real_world_passed = test_real_world_scenario()
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Single Touchdata_id tests: {'PASSED' if single_tests_passed else 'FAILED'}")
    print(f"Multiple Touchdata_id tests: {'PASSED' if multiple_tests_passed else 'FAILED'}")
    print(f"Edge cases tests: {'PASSED' if edge_tests_passed else 'FAILED'}")
    print(f"Real-world scenario test: {'PASSED' if real_world_passed else 'FAILED'}")
    
    all_passed = all([single_tests_passed, multiple_tests_passed, edge_tests_passed, real_world_passed])
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
