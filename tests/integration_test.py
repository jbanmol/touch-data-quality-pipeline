#!/usr/bin/env python3
"""
Integration test for the Kidaura Data Processing application.

This script tests the entire pipeline by:
1. Creating sample JSON files
2. Running the JSON to CSV conversion
3. Running the CSV processing
4. Verifying the results

Note: This does not test the Google Sheets export functionality.
"""

import os
import sys
import json
import shutil
import tempfile

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json_to_csv_converter
import process_csv_data

def create_sample_json(directory, filename, data):
    """Create a sample JSON file."""
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(data, f)

def main():
    """Run the integration test."""
    print("Starting integration test...")

    # Create temporary directories
    raw_json_dir = 'raw_JSONs'
    raw_csv_dir = 'raw_CSVs'
    flagged_data_dir = 'flagged_data'

    # Create directories if they don't exist
    for directory in [raw_json_dir, raw_csv_dir, flagged_data_dir]:
        os.makedirs(directory, exist_ok=True)

    # Clean up any existing files
    for directory in [raw_json_dir, raw_csv_dir, flagged_data_dir]:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # Create sample JSON data
    sample_data = {
        "json": {
            "touchData": {
                "session1": [
                    {
                        "x": 100,
                        "y": 200,
                        "time": 1000,
                        "touchPhase": "Began",
                        "fingerId": 1,
                        "accx": 0.1,
                        "accy": 0.2,
                        "accz": 0.3,
                        "color": "red",
                        "zone": "A",
                        "completionPerc": 0.5
                    },
                    {
                        "x": 150,
                        "y": 250,
                        "time": 1100,
                        "touchPhase": "Moved",
                        "fingerId": 1,
                        "accx": 0.2,
                        "accy": 0.3,
                        "accz": 0.4,
                        "color": "red",
                        "zone": "A",
                        "completionPerc": 0.6
                    },
                    {
                        "x": 200,
                        "y": 300,
                        "time": 1200,
                        "touchPhase": "Ended",
                        "fingerId": 1,
                        "accx": 0.3,
                        "accy": 0.4,
                        "accz": 0.5,
                        "color": "red",
                        "zone": "A",
                        "completionPerc": 0.7
                    }
                ],
                "session2": [
                    {
                        "x": 300,
                        "y": 400,
                        "time": 2000,
                        "touchPhase": "Began",
                        "fingerId": 2,
                        "accx": 0.1,
                        "accy": 0.2,
                        "accz": 0.3,
                        "color": "blue",
                        "zone": "B",
                        "completionPerc": 0.5
                    },
                    {
                        "x": 350,
                        "y": 450,
                        "time": 2100,
                        "touchPhase": "Ended",
                        "fingerId": 2,
                        "accx": 0.2,
                        "accy": 0.3,
                        "accz": 0.4,
                        "color": "blue",
                        "zone": "B",
                        "completionPerc": 0.6
                    }
                ]
            }
        }
    }

    # Create sample JSON files
    create_sample_json(raw_json_dir, 'sample1.json', sample_data)
    create_sample_json(raw_json_dir, 'sample2.json', sample_data)

    print(f"Created {len(os.listdir(raw_json_dir))} sample JSON files")

    # Step 1: Convert JSON to CSV
    print("\nStep 1: Converting JSON to CSV...")
    converted = json_to_csv_converter.convert_json_to_csv(raw_json_dir, raw_csv_dir)
    print(f"Converted {converted} JSON files to CSV")

    # Verify the CSV files were created
    csv_files = os.listdir(raw_csv_dir)
    print(f"Found {len(csv_files)} CSV files in {raw_csv_dir}")

    if len(csv_files) != 2:
        print("Error: Expected 2 CSV files")
        return False

    # Step 2: Process CSV files
    print("\nStep 2: Processing CSV files...")
    processed = process_csv_data.batch_process_csv_files(raw_csv_dir, flagged_data_dir)
    print(f"Processed {processed} CSV files")

    # Verify the processed files were created
    processed_files = os.listdir(flagged_data_dir)
    print(f"Found {len(processed_files)} files in {flagged_data_dir}")

    if len(processed_files) < 2:  # At least 2 files (2 CSVs + possibly a summary)
        print("Error: Expected at least 2 processed files")
        return False

    # Check if summary.csv was created
    if 'summary.csv' in processed_files:
        print("Summary file was created successfully")
    else:
        print("Warning: Summary file was not created")

    print("\nIntegration test completed successfully!")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
