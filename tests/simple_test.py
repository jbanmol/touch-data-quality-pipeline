#!/usr/bin/env python3
"""
Simple test for the Kidaura Data Processing application.
"""

import os
import sys
import json
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json_to_csv_converter
import process_csv_data

def main():
    """Run a simple test."""
    print("Starting simple test...")

    # Create directories if they don't exist
    raw_json_dir = 'raw_JSONs'
    raw_csv_dir = 'raw_CSVs'
    flagged_data_dir = 'flagged_data'

    for directory in [raw_json_dir, raw_csv_dir, flagged_data_dir]:
        os.makedirs(directory, exist_ok=True)

    # Create a simple JSON file
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
                        "touchPhase": "Ended",
                        "fingerId": 1,
                        "accx": 0.2,
                        "accy": 0.3,
                        "accz": 0.4,
                        "color": "red",
                        "zone": "A",
                        "completionPerc": 0.6
                    }
                ]
            }
        }
    }

    # Write the sample JSON to a file
    json_path = os.path.join(raw_json_dir, 'simple_test.json')
    with open(json_path, 'w') as f:
        json.dump(sample_data, f)

    print(f"Created JSON file: {json_path}")

    # Convert JSON to CSV
    print("\nConverting JSON to CSV...")
    start_time = time.time()
    converted = json_to_csv_converter.convert_json_to_csv(raw_json_dir, raw_csv_dir)
    json_to_csv_time = time.time() - start_time

    print(f"JSON to CSV conversion: {json_to_csv_time:.2f} seconds")
    print(f"Converted {converted} JSON files to CSV")

    # Process CSV files
    print("\nProcessing CSV files...")
    start_time = time.time()
    processed = process_csv_data.batch_process_csv_files(raw_csv_dir, flagged_data_dir)
    csv_processing_time = time.time() - start_time

    print(f"CSV processing: {csv_processing_time:.2f} seconds")
    print(f"Processed {processed} CSV files")

    print("\nSimple test completed successfully!")
    return True

if __name__ == '__main__':
    main()
