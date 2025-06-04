#!/usr/bin/env python3
"""
Test script to verify that Coloring data exports use the correct column ordering.
This script tests that the column order matches the exact specification for Google Sheets export.
"""

import pandas as pd
import sys
import os
import tempfile

# Add the current directory to the path so we can import process_csv_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from process_csv_data import process_coloring_data
    print("Successfully imported process_coloring_data function")
except ImportError as e:
    print(f"Error importing function: {e}")
    sys.exit(1)

def create_test_coloring_data():
    """Create test Coloring data with enhanced fields."""
    data = {
        'Touchdata_id': ['TD001', 'TD001', 'TD001', 'TD002', 'TD002'],
        'event_index': [1, 2, 3, 1, 2],
        'x': [100.0, 105.0, 110.0, 200.0, 205.0],
        'y': [50.0, 55.0, 60.0, 150.0, 155.0],
        'time': [1000, 1100, 1200, 2000, 2100],
        'touchPhase': ['Began', 'Moved', 'Ended', 'Began', 'Ended'],
        'fingerId': [1, 1, 1, 2, 2],
        'color': ['red', 'red', 'red', 'blue', 'blue'],
        'completionPerc': [10.0, 50.0, 100.0, 20.0, 100.0],
        'zone': ['A', 'A', 'A', 'B', 'B']
    }
    return pd.DataFrame(data)

def create_legacy_coloring_data():
    """Create legacy Coloring data without enhanced fields."""
    data = {
        'fingerId': [1, 1, 1, 2, 2],
        'seqId': [1, 1, 1, 2, 2],
        'x': [100.0, 105.0, 110.0, 200.0, 205.0],
        'y': [50.0, 55.0, 60.0, 150.0, 155.0],
        'time': [1000, 1100, 1200, 2000, 2100],
        'touchPhase': ['Began', 'Moved', 'Ended', 'Began', 'Ended'],
        'color': ['red', 'red', 'red', 'blue', 'blue'],
        'completionPerc': [10.0, 50.0, 100.0, 20.0, 100.0],
        'zone': ['A', 'A', 'A', 'B', 'B']
    }
    return pd.DataFrame(data)

def test_enhanced_column_ordering():
    """Test that enhanced Coloring data uses the correct column order."""
    print("\n=== Testing Enhanced Column Ordering ===")
    
    # Expected column order for enhanced data
    expected_order = [
        'Touchdata_id', 'event_index', 'x', 'y', 'time', 
        'touchPhase', 'fingerId', 'color', 'completionPerc', 'zone', 'flags'
    ]
    
    # Create test data
    df = create_test_coloring_data()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
        df.to_csv(input_file.name, index=False)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Process the data
        processed_df, file_stats = process_coloring_data(df, input_path, output_path)
        
        # Read the output file to check column order
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            
            # Get the actual column order (excluding any extra columns like blank columns)
            actual_columns = [col for col in output_df.columns if not col.startswith(' ')]
            
            # Check if the expected columns are present and in the right order
            matching_columns = []
            for expected_col in expected_order:
                if expected_col in actual_columns:
                    matching_columns.append(expected_col)
            
            print(f"Expected order: {expected_order}")
            print(f"Actual order:   {matching_columns}")
            
            # Check if the order matches
            order_correct = matching_columns == expected_order
            print(f"Column order correct: {order_correct}")
            
            if not order_correct:
                print("❌ Enhanced column ordering test FAILED")
                return False
            else:
                print("✅ Enhanced column ordering test PASSED")
                return True
        else:
            print("❌ Output file was not created")
            return False
            
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def test_legacy_column_ordering():
    """Test that legacy Coloring data uses the correct column order."""
    print("\n=== Testing Legacy Column Ordering ===")
    
    # Expected column order for legacy data
    expected_order = [
        'fingerId', 'seqId', 'x', 'y', 'time', 
        'touchPhase', 'color', 'completionPerc', 'zone', 'flags'
    ]
    
    # Create test data
    df = create_legacy_coloring_data()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
        df.to_csv(input_file.name, index=False)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Process the data
        processed_df, file_stats = process_coloring_data(df, input_path, output_path)
        
        # Read the output file to check column order
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            
            # Get the actual column order (excluding any extra columns like blank columns)
            actual_columns = [col for col in output_df.columns if not col.startswith(' ')]
            
            # Check if the expected columns are present and in the right order
            matching_columns = []
            for expected_col in expected_order:
                if expected_col in actual_columns:
                    matching_columns.append(expected_col)
            
            print(f"Expected order: {expected_order}")
            print(f"Actual order:   {matching_columns}")
            
            # Check if the order matches
            order_correct = matching_columns == expected_order
            print(f"Column order correct: {order_correct}")
            
            if not order_correct:
                print("❌ Legacy column ordering test FAILED")
                return False
            else:
                print("✅ Legacy column ordering test PASSED")
                return True
        else:
            print("❌ Output file was not created")
            return False
            
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def main():
    """Run all column ordering tests."""
    print("Testing Coloring Data Column Ordering")
    print("=" * 50)
    
    enhanced_test_passed = test_enhanced_column_ordering()
    legacy_test_passed = test_legacy_column_ordering()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Enhanced column ordering: {'PASSED' if enhanced_test_passed else 'FAILED'}")
    print(f"Legacy column ordering:   {'PASSED' if legacy_test_passed else 'FAILED'}")
    
    all_passed = enhanced_test_passed and legacy_test_passed
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ Column ordering is correctly implemented for Google Sheets export")
    else:
        print("\n❌ Column ordering issues detected")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
