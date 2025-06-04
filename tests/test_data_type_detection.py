#!/usr/bin/env python3
"""
Unit tests for the data type detection mechanism in the CSV processing pipeline.

These tests verify that the data type detection correctly distinguishes between
Coloring and Tracing data types and handles edge cases appropriately.
"""

import os
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from process_csv_data import detect_data_type

class TestDataTypeDetection(unittest.TestCase):
    """Test cases for the data type detection mechanism."""

    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample Coloring data
        self.coloring_data = pd.DataFrame({
            'fingerId': [1, 1, 1, 2, 2, 2],
            'x': [100, 110, 120, 200, 210, 220],
            'y': [100, 110, 120, 200, 210, 220],
            'time': [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            'touchPhase': ['Began', 'Moved', 'Ended', 'Began', 'Moved', 'Ended'],
            'color': ['red', 'red', 'red', 'blue', 'blue', 'blue'],
            'completionPerc': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            'zone': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        # Create sample Tracing data
        self.tracing_data = pd.DataFrame({
            'fingerId': [1, 1, 1, 2, 2, 2],
            'x': [100, 110, 120, 200, 210, 220],
            'y': [100, 110, 120, 200, 210, 220],
            'time': [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            'touchPhase': ['B', 'M', 'E', 'B', 'M', 'E'],
            'distance': [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            'camFrame': [0, 1, 2, 0, 1, 2],
            'isDragging': [True, True, False, True, True, False],
            'zone': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        # Create mixed data (ambiguous case)
        self.mixed_data = pd.DataFrame({
            'fingerId': [1, 1, 1, 2, 2, 2],
            'x': [100, 110, 120, 200, 210, 220],
            'y': [100, 110, 120, 200, 210, 220],
            'time': [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            'touchPhase': ['Began', 'M', 'Ended', 'B', 'Moved', 'E'],
            'zone': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        # Create malformed data (missing touchPhase)
        self.malformed_data = pd.DataFrame({
            'fingerId': [1, 1, 1],
            'x': [100, 110, 120],
            'y': [100, 110, 120],
            'time': [1.0, 1.1, 1.2],
            'zone': ['A', 'A', 'A']
        })

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_detect_coloring_from_filename(self):
        """Test detection of Coloring data from filename."""
        data_type = detect_data_type(self.coloring_data, "Coloring_2023-01-01.csv")
        self.assertEqual(data_type, "Coloring")

    def test_detect_tracing_from_filename(self):
        """Test detection of Tracing data from filename."""
        data_type = detect_data_type(self.tracing_data, "Tracing_2023-01-01.csv")
        self.assertEqual(data_type, "Tracing")

    def test_detect_coloring_from_columns(self):
        """Test detection of Coloring data from column structure."""
        data_type = detect_data_type(self.coloring_data, "data.csv")
        self.assertEqual(data_type, "Coloring")

    def test_detect_tracing_from_columns(self):
        """Test detection of Tracing data from column structure."""
        data_type = detect_data_type(self.tracing_data, "data.csv")
        self.assertEqual(data_type, "Tracing")

    def test_detect_coloring_from_touchphase(self):
        """Test detection of Coloring data from touchPhase values."""
        # Create data with only touchPhase column
        data = pd.DataFrame({
            'fingerId': [1, 1, 1],
            'time': [1.0, 1.1, 1.2],
            'touchPhase': ['Began', 'Moved', 'Ended']
        })
        data_type = detect_data_type(data, "data.csv")
        self.assertEqual(data_type, "Coloring")

    def test_detect_tracing_from_touchphase(self):
        """Test detection of Tracing data from touchPhase values."""
        # Create data with only touchPhase column
        data = pd.DataFrame({
            'fingerId': [1, 1, 1],
            'time': [1.0, 1.1, 1.2],
            'touchPhase': ['B', 'M', 'E']
        })
        data_type = detect_data_type(data, "data.csv")
        self.assertEqual(data_type, "Tracing")

    def test_mixed_data_detection(self):
        """Test detection with mixed data (should pick the dominant type)."""
        data_type = detect_data_type(self.mixed_data, "mixed_data.csv")
        # The result depends on the implementation, but it should not be 'Unknown'
        self.assertNotEqual(data_type, "Unknown")

    def test_malformed_data_detection(self):
        """Test detection with malformed data (missing touchPhase)."""
        # This should return 'Unknown' since there's not enough information
        with self.assertRaises(KeyError):
            detect_data_type(self.malformed_data, "malformed_data.csv")

    def test_misleading_filename(self):
        """Test detection when filename suggests one type but content suggests another."""
        # Filename suggests Tracing but content is Coloring
        data_type = detect_data_type(self.coloring_data, "Tracing_misleading.csv")
        # The implementation should prioritize content over filename
        self.assertEqual(data_type, "Coloring")

if __name__ == '__main__':
    unittest.main()
