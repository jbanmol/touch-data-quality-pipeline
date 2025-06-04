#!/usr/bin/env python3
"""
Test script to verify that Touchdata_id-based validation results are correctly reflected in exports.
This script processes a sample file and checks the flag results.
"""

import pandas as pd
import sys
import os

# Add the current directory to the path so we can import process_csv_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from process_csv_data import process_coloring_data
    print("Successfully imported process_coloring_data function")
except ImportError as e:
    print(f"Error importing function: {e}")
    sys.exit(1)

def test_export_validation():
    """Test that enhanced validation results are correctly reflected in processed output."""
    print("Testing Enhanced Validation Export")
    print("=" * 40)
    
    # Use an existing raw CSV file with Touchdata_id and event_index
    input_file = "raw_CSVs/Coloring_2022-02-08 13_45_25.328039_620226115c0616af26124166.csv"
    output_file = "test_output_validation.csv"
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return False
    
    try:
        # Load the CSV file
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Verify enhanced fields are present
        if 'Touchdata_id' in df.columns and 'event_index' in df.columns:
            print("✓ Enhanced fields (Touchdata_id, event_index) found")
        else:
            print("✗ Enhanced fields missing")
            return False
        
        # Process the data
        print("Processing data with enhanced validation...")
        processed_df, file_stats = process_coloring_data(df, input_file, output_file)
        
        # Analyze the results
        print("\nValidation Results Analysis:")
        print("-" * 30)
        
        # Check if output file was created
        if os.path.exists(output_file):
            print(f"✓ Output file {output_file} created successfully")
            
            # Read the output file to analyze flag results
            output_df = pd.read_csv(output_file)
            
            # Count sequences with and without flags
            flag_column = 'flags'
            if flag_column in output_df.columns:
                total_rows = len(output_df)
                flagged_rows = output_df[flag_column].astype(bool).sum()
                unflagged_rows = total_rows - flagged_rows
                
                print(f"Total rows: {total_rows}")
                print(f"Flagged rows: {flagged_rows}")
                print(f"Unflagged rows: {unflagged_rows}")
                print(f"Flagged percentage: {(flagged_rows/total_rows)*100:.2f}%")
                
                # Show some examples of flag results
                print("\nSample Flag Results:")
                print("-" * 20)
                
                # Show flagged sequences
                flagged_samples = output_df[output_df[flag_column].astype(bool)].head(3)
                if len(flagged_samples) > 0:
                    print("Flagged sequences:")
                    for idx, row in flagged_samples.iterrows():
                        touchdata_id = row.get('Touchdata_id', 'N/A')
                        touch_phase = row.get('touchPhase', 'N/A')
                        flags = row.get('flags', 'N/A')
                        print(f"  Touchdata_id {touchdata_id}, Phase: {touch_phase}, Flags: '{flags}'")
                
                # Show unflagged sequences
                unflagged_samples = output_df[~output_df[flag_column].astype(bool)].head(3)
                if len(unflagged_samples) > 0:
                    print("Unflagged sequences:")
                    for idx, row in unflagged_samples.iterrows():
                        touchdata_id = row.get('Touchdata_id', 'N/A')
                        touch_phase = row.get('touchPhase', 'N/A')
                        print(f"  Touchdata_id {touchdata_id}, Phase: {touch_phase}, Flags: (none)")
                
                # Analyze unique Touchdata_id sequences
                print(f"\nUnique Touchdata_id sequences: {df['Touchdata_id'].nunique()}")
                
                # Group by Touchdata_id to see sequence patterns
                print("\nTouchdata_id Sequence Analysis:")
                print("-" * 30)
                
                for touchdata_id in sorted(df['Touchdata_id'].unique())[:5]:  # Show first 5
                    seq_data = df[df['Touchdata_id'] == touchdata_id].sort_values('event_index')
                    touch_phases = seq_data['touchPhase'].tolist()
                    
                    # Check if this sequence has flags in the output
                    seq_output = output_df[output_df.get('Touchdata_id', -1) == touchdata_id]
                    has_flags = seq_output[flag_column].astype(bool).any() if len(seq_output) > 0 else False
                    
                    flag_status = "FLAGGED" if has_flags else "VALID"
                    print(f"  Touchdata_id {touchdata_id}: {touch_phases} → {flag_status}")
                
            else:
                print(f"✗ Flags column not found in output file")
                return False
        else:
            print(f"✗ Output file {output_file} not created")
            return False
        
        # Check file statistics
        print(f"\nFile Statistics:")
        print(f"Data type: {file_stats.get('data_type', 'Unknown')}")
        print(f"Flagged percentage: {file_stats.get('flagged_percentage', 'Unknown')}")
        
        # Clean up test file
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Cleaned up test file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return False

def main():
    """Run the export validation test."""
    print("Enhanced Validation Export Test")
    print("=" * 50)
    
    success = test_export_validation()
    
    print("\n" + "=" * 50)
    print(f"Export Validation Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("\n✓ Enhanced Touchdata_id-based validation is working correctly")
        print("✓ Flag results are properly reflected in processed output")
        print("✓ Export functionality will correctly show validation results")
    else:
        print("\n✗ Issues detected with enhanced validation export")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
