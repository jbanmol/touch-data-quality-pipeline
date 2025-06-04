#!/usr/bin/env python3
"""
Performance test for the Kidaura Data Processing application.

This script tests the performance of the application by:
1. Generating large JSON files with varying numbers of touch points
2. Measuring the time taken to convert them to CSV
3. Measuring the time taken to process the CSV files
"""

import os
import sys
import json
import time
import random
import shutil

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json_to_csv_converter
import process_csv_data

def generate_touch_point(finger_id, time_value, phase):
    """Generate a random touch point."""
    return {
        "x": random.randint(0, 1000),
        "y": random.randint(0, 1000),
        "time": time_value,
        "touchPhase": phase,
        "fingerId": finger_id,
        "accx": random.random(),
        "accy": random.random(),
        "accz": random.random(),
        "color": random.choice(["red", "green", "blue"]),
        "zone": random.choice(["A", "B", "C"]),
        "completionPerc": random.random()
    }

def generate_sequence(finger_id, start_time, length):
    """Generate a sequence of touch points."""
    sequence = []

    # Add "Began" phase
    sequence.append(generate_touch_point(finger_id, start_time, "Began"))

    # Add "Moved" phases
    for i in range(1, length - 1):
        sequence.append(generate_touch_point(finger_id, start_time + i * 100, "Moved"))

    # Add "Ended" phase
    sequence.append(generate_touch_point(finger_id, start_time + (length - 1) * 100, "Ended"))

    return sequence

def generate_large_json(num_sequences, points_per_sequence):
    """Generate a large JSON file with the specified number of sequences and points per sequence."""
    data = {"json": {"touchData": {}}}

    for i in range(num_sequences):
        finger_id = i + 1
        start_time = i * 1000
        sequence = generate_sequence(finger_id, start_time, points_per_sequence)
        data["json"]["touchData"][f"session{i}"] = sequence

    return data

def run_performance_test(sizes):
    """Run performance tests with different file sizes."""
    print("Starting performance test...")

    # Create directories if they don't exist
    raw_json_dir = 'raw_JSONs'
    raw_csv_dir = 'raw_CSVs'
    flagged_data_dir = 'flagged_data'

    for directory in [raw_json_dir, raw_csv_dir, flagged_data_dir]:
        os.makedirs(directory, exist_ok=True)

    results = []

    for size in sizes:
        num_sequences, points_per_sequence = size
        total_points = num_sequences * points_per_sequence

        print(f"\nTesting with {num_sequences} sequences, {points_per_sequence} points per sequence ({total_points} total points)")

        # Clean up any existing files
        for directory in [raw_json_dir, raw_csv_dir, flagged_data_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        # Generate the JSON file
        json_data = generate_large_json(num_sequences, points_per_sequence)
        json_path = os.path.join(raw_json_dir, f"test_{num_sequences}_{points_per_sequence}.json")

        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        print(f"Generated JSON file: {json_path}")

        # Measure JSON to CSV conversion time
        start_time = time.time()
        converted = json_to_csv_converter.convert_json_to_csv(raw_json_dir, raw_csv_dir)
        json_to_csv_time = time.time() - start_time

        print(f"JSON to CSV conversion: {json_to_csv_time:.2f} seconds")

        # Measure CSV processing time
        start_time = time.time()
        processed = process_csv_data.batch_process_csv_files(raw_csv_dir, flagged_data_dir)
        csv_processing_time = time.time() - start_time

        print(f"CSV processing: {csv_processing_time:.2f} seconds")

        # Record results
        results.append({
            "num_sequences": num_sequences,
            "points_per_sequence": points_per_sequence,
            "total_points": total_points,
            "json_to_csv_time": json_to_csv_time,
            "csv_processing_time": csv_processing_time,
            "total_time": json_to_csv_time + csv_processing_time
        })

    # Print summary
    print("\nPerformance Test Results:")
    print("------------------------")
    print(f"{'Sequences':<10} {'Points/Seq':<10} {'Total Points':<15} {'JSON->CSV (s)':<15} {'Processing (s)':<15} {'Total (s)':<10}")
    print("-" * 75)

    for result in results:
        print(f"{result['num_sequences']:<10} {result['points_per_sequence']:<10} {result['total_points']:<15} {result['json_to_csv_time']:<15.2f} {result['csv_processing_time']:<15.2f} {result['total_time']:<10.2f}")

if __name__ == '__main__':
    # Test with different file sizes
    # (num_sequences, points_per_sequence)
    sizes = [
        (10, 10),      # Small: 100 points
        (100, 10),     # Medium: 1,000 points
        (100, 100),    # Large: 10,000 points
    ]

    run_performance_test(sizes)
