#!/usr/bin/env python3
"""
Integration test for the new Coloring sequence validation logic.
This script tests the complete flag rules system with the new validation.
"""

import pandas as pd
import sys
import os

# Add the current directory to the path so we can import process_csv_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from process_csv_data import apply_flag_rules, compute_sequence_metrics
    print("Successfully imported required functions")
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

def create_test_data():
    """Create test data with various sequence patterns including Touchdata_id and event_index."""
    data = []

    # Valid sequence 1: Touchdata_id=1, fingerId=1, seqId=1
    data.extend([
        {'fingerId': 1, 'seqId': 1, 'Touchdata_id': 1, 'event_index': 0, 'time': 1.0, 'touchPhase': 'Began', 'x': 100, 'y': 100, 'color': 'Red', 'completionPerc': 0.0, 'zone': 'Area1'},
        {'fingerId': 1, 'seqId': 1, 'Touchdata_id': 1, 'event_index': 1, 'time': 2.0, 'touchPhase': 'Moved', 'x': 110, 'y': 110, 'color': 'Red', 'completionPerc': 0.1, 'zone': 'Area1'},
        {'fingerId': 1, 'seqId': 1, 'Touchdata_id': 1, 'event_index': 2, 'time': 3.0, 'touchPhase': 'Stationary', 'x': 110, 'y': 110, 'color': 'Red', 'completionPerc': 0.1, 'zone': 'Area1'},
        {'fingerId': 1, 'seqId': 1, 'Touchdata_id': 1, 'event_index': 3, 'time': 4.0, 'touchPhase': 'Ended', 'x': 110, 'y': 110, 'color': 'Red', 'completionPerc': 0.1, 'zone': 'Area1'},
    ])

    # Valid sequence 2 with Canceled: Touchdata_id=2, fingerId=2, seqId=1
    data.extend([
        {'fingerId': 2, 'seqId': 1, 'Touchdata_id': 2, 'event_index': 0, 'time': 5.0, 'touchPhase': 'Began', 'x': 200, 'y': 200, 'color': 'Blue', 'completionPerc': 0.0, 'zone': 'Area2'},
        {'fingerId': 2, 'seqId': 1, 'Touchdata_id': 2, 'event_index': 1, 'time': 6.0, 'touchPhase': 'Moved', 'x': 210, 'y': 210, 'color': 'Blue', 'completionPerc': 0.2, 'zone': 'Area2'},
        {'fingerId': 2, 'seqId': 1, 'Touchdata_id': 2, 'event_index': 2, 'time': 7.0, 'touchPhase': 'Canceled', 'x': 210, 'y': 210, 'color': 'Blue', 'completionPerc': 0.2, 'zone': 'Area2'},
        {'fingerId': 2, 'seqId': 1, 'Touchdata_id': 2, 'event_index': 3, 'time': 8.0, 'touchPhase': 'Ended', 'x': 210, 'y': 210, 'color': 'Blue', 'completionPerc': 0.2, 'zone': 'Area2'},
    ])

    # Invalid sequence 1 - missing Began: Touchdata_id=3, fingerId=3, seqId=1
    data.extend([
        {'fingerId': 3, 'seqId': 1, 'Touchdata_id': 3, 'event_index': 0, 'time': 9.0, 'touchPhase': 'Moved', 'x': 300, 'y': 300, 'color': 'Green', 'completionPerc': 0.3, 'zone': 'Area3'},
        {'fingerId': 3, 'seqId': 1, 'Touchdata_id': 3, 'event_index': 1, 'time': 10.0, 'touchPhase': 'Ended', 'x': 310, 'y': 310, 'color': 'Green', 'completionPerc': 0.3, 'zone': 'Area3'},
    ])

    # Invalid sequence 2 - missing Ended: Touchdata_id=4, fingerId=4, seqId=1
    data.extend([
        {'fingerId': 4, 'seqId': 1, 'Touchdata_id': 4, 'event_index': 0, 'time': 11.0, 'touchPhase': 'Began', 'x': 400, 'y': 400, 'color': 'Yellow', 'completionPerc': 0.0, 'zone': 'Area4'},
        {'fingerId': 4, 'seqId': 1, 'Touchdata_id': 4, 'event_index': 1, 'time': 12.0, 'touchPhase': 'Moved', 'x': 410, 'y': 410, 'color': 'Yellow', 'completionPerc': 0.4, 'zone': 'Area4'},
    ])

    # Invalid sequence 3 - multiple Canceled: Touchdata_id=5, fingerId=5, seqId=1
    data.extend([
        {'fingerId': 5, 'seqId': 1, 'Touchdata_id': 5, 'event_index': 0, 'time': 13.0, 'touchPhase': 'Began', 'x': 500, 'y': 500, 'color': 'Purple', 'completionPerc': 0.0, 'zone': 'Area5'},
        {'fingerId': 5, 'seqId': 1, 'Touchdata_id': 5, 'event_index': 1, 'time': 14.0, 'touchPhase': 'Canceled', 'x': 500, 'y': 500, 'color': 'Purple', 'completionPerc': 0.0, 'zone': 'Area5'},
        {'fingerId': 5, 'seqId': 1, 'Touchdata_id': 5, 'event_index': 2, 'time': 15.0, 'touchPhase': 'Canceled', 'x': 500, 'y': 500, 'color': 'Purple', 'completionPerc': 0.0, 'zone': 'Area5'},
        {'fingerId': 5, 'seqId': 1, 'Touchdata_id': 5, 'event_index': 3, 'time': 16.0, 'touchPhase': 'Ended', 'x': 500, 'y': 500, 'color': 'Purple', 'completionPerc': 0.0, 'zone': 'Area5'},
    ])

    return pd.DataFrame(data)

def test_flag_rules_integration():
    """Test the complete flag rules system with the new validation."""
    print("Testing Flag Rules Integration")
    print("=" * 40)

    # Create test data
    df = create_test_data()
    print(f"Created test data with {len(df)} rows and {df['seqId'].nunique()} sequences")

    # Compute sequence metrics
    seq_metrics = compute_sequence_metrics(df)
    print(f"Computed metrics for {len(seq_metrics)} sequences")

    # Apply flag rules
    df_flagged, seq_metrics_flagged = apply_flag_rules(df, seq_metrics)

    # Analyze results
    print("\nSequence Analysis:")
    print("-" * 20)

    for idx, row in seq_metrics_flagged.iterrows():
        finger_id = row['fingerId']
        seq_id = row['seqId']
        flags = row['flags']

        # Get the sequence data
        seq_data = df[df['fingerId'] == finger_id]
        touch_phases = seq_data['touchPhase'].tolist()

        print(f"Finger {finger_id}, Seq {seq_id}: {touch_phases}")
        print(f"  Flags: '{flags}' (empty means valid sequence)")
        print()

    # Check expected results - we check for presence of key flags, not exact matches
    expected_results = {
        1: {"should_be_empty": True},  # Valid sequence - should have no flags
        2: {"should_be_empty": True},  # Valid sequence with Canceled - should have no flags
        3: {"should_contain": ["missing_Began"]},  # Invalid - should contain missing_Began
        4: {"should_contain": ["missing_Ended"]},  # Invalid - should contain missing_Ended
        5: {"should_not_be_empty": True},  # Invalid - multiple Canceled, should be flagged
    }

    print("Validation Results:")
    print("-" * 20)

    all_correct = True
    for finger_id, expected in expected_results.items():
        actual_flags = seq_metrics_flagged[seq_metrics_flagged['fingerId'] == finger_id]['flags'].iloc[0]

        if "should_be_empty" in expected and expected["should_be_empty"]:
            is_correct = actual_flags == ""
            print(f"Finger {finger_id}: Expected no flags, Got '{actual_flags}' - {'✓' if is_correct else '✗'}")
        elif "should_contain" in expected:
            required_flags = expected["should_contain"]
            is_correct = all(flag in actual_flags for flag in required_flags)
            print(f"Finger {finger_id}: Expected to contain {required_flags}, Got '{actual_flags}' - {'✓' if is_correct else '✗'}")
        elif "should_not_be_empty" in expected:
            is_correct = actual_flags != ""
            print(f"Finger {finger_id}: Expected some flags, Got '{actual_flags}' - {'✓' if is_correct else '✗'}")
        else:
            is_correct = False
            print(f"Finger {finger_id}: Invalid test configuration - ✗")

        if not is_correct:
            all_correct = False

    return all_correct

def main():
    """Run the integration test."""
    print("Coloring Sequence Validation - Integration Test")
    print("=" * 50)

    success = test_flag_rules_integration()

    print("\n" + "=" * 50)
    print(f"Integration Test Result: {'PASSED' if success else 'FAILED'}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
