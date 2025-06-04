#!/usr/bin/env python3
"""
JSON to CSV Converter

This script converts JSON files from the raw_data folder to CSV format and saves them in the rawCSVs folder.
It maintains the exact structure and order of fields from the original JSON files.

Features:
- Parallel processing for faster conversion of multiple files
- Configurable number of worker processes
- Robust error handling and logging

Usage:
    python json_to_csv_converter.py

"""

import json
import csv
import os
import sys
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def convert_json_to_csv(raw_data_folder, output_folder, max_workers=None):
    """
    Convert JSON files to CSV format using parallel processing.

    Args:
        raw_data_folder (str): Path to the folder containing JSON files
        output_folder (str): Path to the folder where CSV files will be saved
        max_workers (int, optional): Maximum number of worker processes to use.
                                    If None, uses CPU count - 1 (or 1 if single core)

    Returns:
        int: Number of files successfully converted
    """
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output directory {output_folder} is ready")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return 0

    # Get list of JSON files
    try:
        json_files = [f for f in os.listdir(raw_data_folder) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in {raw_data_folder}")
    except Exception as e:
        logger.error(f"Failed to read files from {raw_data_folder}: {e}")
        return 0

    if not json_files:
        logger.warning(f"No JSON files found in {raw_data_folder}")
        return 0

    # Determine the number of worker processes
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {max_workers} worker processes for parallel conversion")

    # Create a list of (input_path, output_path) tuples for parallel processing
    file_pairs = []
    for json_file in json_files:
        input_path = os.path.join(raw_data_folder, json_file)
        output_path = os.path.join(output_folder, os.path.splitext(json_file)[0] + '.csv')
        file_pairs.append((input_path, output_path, json_file))

    # Process files in parallel using ProcessPoolExecutor
    successful_conversions = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and map them to their original filenames for better error reporting
        future_to_file = {
            executor.submit(_process_json_file_wrapper, input_path, output_path): json_file
            for input_path, output_path, json_file in file_pairs
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                if future.result():
                    successful_conversions += 1
                    logger.info(f"Successfully converted {json_file}")
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Successfully converted {successful_conversions} out of {len(json_files)} files using parallel processing")
    return successful_conversions

def _process_json_file_wrapper(input_path, output_path):
    """
    Wrapper function for process_json_file to be used with ProcessPoolExecutor.
    Handles exceptions internally to prevent worker crashes.

    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path to the output CSV file

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        return process_json_file(input_path, output_path)
    except Exception as e:
        # Log the error and return False instead of raising the exception
        # This prevents the worker from crashing
        logger.error(f"Error in worker processing {input_path}: {e}")
        return False

def process_json_file(input_path, output_path):
    """
    Process a single JSON file and convert it to CSV.

    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path to the output CSV file

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    logger.info(f"Processing {input_path}")

    try:
        # Read JSON file
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Extract touch data
        if 'json' not in data or 'touchData' not in data['json']:
            logger.warning(f"File {input_path} does not contain expected JSON structure")
            return False

        # Extract all touch data entries
        all_entries = []
        touch_data = data['json']['touchData']

        # Log the file type for informational purposes
        if 'dataSet' in data['json'] and data['json']['dataSet'] == 'Tracing':
            logger.info(f"Detected Tracing JSON file: {input_path}")
        elif os.path.basename(input_path).startswith('Tracing_'):
            logger.info(f"Detected Tracing JSON file from filename: {input_path}")
        else:
            logger.info(f"Processing as Coloring JSON file: {input_path}")

        # Process entries, preserving all original attributes
        for finger_id, entries in touch_data.items():
            # For each entry, add the finger_id if it's not already present, and add Touchdata_id and event_index
            for i, entry in enumerate(entries):
                # Create a copy of the entry to avoid modifying the original
                processed_entry = entry.copy()

                # Add fingerId from the key if it's not already in the entry
                if 'fingerId' not in processed_entry:
                    processed_entry['fingerId'] = finger_id

                # Add Touchdata_id and event_index
                processed_entry['Touchdata_id'] = finger_id
                processed_entry['event_index'] = i

                all_entries.append(processed_entry)

        if not all_entries:
            logger.warning(f"No touch data entries found in {input_path}")
            return False

        # Sort all entries by time to ensure chronological order
        try:
            all_entries.sort(key=lambda x: float(x.get('time', 0)))
            logger.info(f"Sorted {len(all_entries)} entries by time")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not sort entries by time: {e}. Continuing with unsorted data.")

        # Get all unique fields from all entries to ensure we capture everything
        all_fields = set()
        for entry in all_entries:
            all_fields.update(entry.keys())

        # Convert to sorted list for consistent field order
        field_order = sorted(list(all_fields))

        logger.info(f"Found {len(field_order)} fields in {input_path}: {', '.join(field_order)}")

        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_order, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_entries)

        logger.info(f"Successfully converted {input_path} to {output_path}")
        return True

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {input_path}")
        return False
    except KeyError as e:
        logger.error(f"Missing expected key in {input_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {input_path}: {e}")
        return False

def main(raw_data_folder='raw_JSONs', output_folder='raw_CSVs', max_workers=None):
    """
    Main function to run the converter.

    Args:
        raw_data_folder (str): Path to the folder containing JSON files
        output_folder (str): Path to the folder where CSV files will be saved
        max_workers (int, optional): Maximum number of worker processes to use.
                                    If None, uses CPU count - 1 (or 1 if single core)
    """
    logger.info("Starting JSON to CSV conversion with parallel processing")
    start_time = __import__('time').time()

    successful = convert_json_to_csv(raw_data_folder, output_folder, max_workers)

    elapsed_time = __import__('time').time() - start_time
    logger.info(f"Conversion process completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully converted {successful} files")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert JSON files to CSV format using parallel processing')
    parser.add_argument('--input', default='raw_JSONs', help='Input folder containing JSON files')
    parser.add_argument('--output', default='raw_CSVs', help='Output folder for CSV files')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count - 1)')

    args = parser.parse_args()

    main(args.input, args.output, args.workers)
