#!/usr/bin/env python3
"""
CSV Data Processing Pipeline

This script processes CSV files from the rawCSVs folder and applies a series of
transformations to create cleaned data files with flags for data quality issues.

The pipeline includes processing paths for both data types:
- Coloring data: Full sequence analysis with metrics and flags
- Tracing data: Full sequence analysis with metrics and flags

Key improvements in sequence handling:
- Sequences are considered complete only when they have proper ending events
- New sequence IDs start only after the current sequence has properly ended
- Sequences without proper ending events are flagged
- Each sequence ID represents a complete touch interaction from beginning to end

Common steps for both data types:
1. Loading and sorting data by time
2. Detecting data type (Coloring vs Tracing)
3. Segmenting sequences based on start and end events
4. Computing sequence metrics
5. Applying appropriate flag rules
6. Saving outputs with flags
7. Generating summary statistics

"""

import os
import pandas as pd
import numpy as np
import logging
import sys

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import ML enhancers (try enhanced first, fallback to consolidated)
try:
    from ..ml.ml_integration import enhance_dataframe_with_advanced_ml
    ML_ENHANCED_AVAILABLE = True
    logger.info("Enhanced ML Integration system loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced ML Integration not available: {e}")
    ML_ENHANCED_AVAILABLE = False

try:
    from ..ml.consolidated_enhancer import ConsolidatedMLEnhancer
    ML_ENHANCER_AVAILABLE = True
    logger.info("ML Consolidated Enhancer loaded successfully")
except ImportError as e:
    ML_ENHANCER_AVAILABLE = False
    logger.warning(f"ML Consolidated Enhancer not available: {e}")
    logger.info("Will use rule-based fallback for ML metadata")

def prepare_output_folder(output_folder):
    """Create output folder if it doesn't exist."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output directory {output_folder} is ready")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

def detect_data_type(df, filename=""):
    """
    Enhanced detection of whether the data is Coloring or Tracing based on multiple indicators.
    Uses a multi-layered approach with confidence scoring and detailed logging.

    Args:
        df (DataFrame): The loaded DataFrame
        filename (str): Optional filename for additional detection

    Returns:
        tuple: (data_type, confidence, detection_methods)
            - data_type: 'Coloring', 'Tracing', or 'Unknown'
            - confidence: Float between 0.0 and 1.0 indicating detection confidence
            - detection_methods: List of successful detection methods
    """
    # Initialize detection variables
    coloring_score = 0.0
    tracing_score = 0.0
    detection_methods = []
    ambiguities = []

    # Method 1: Check filename pattern (highest confidence)
    if filename:
        if 'Tracing_' in filename:
            tracing_score += 0.6
            detection_methods.append("filename_pattern")
            logger.info(f"Detected Tracing data from filename pattern: {filename}")
        elif 'Coloring_' in filename:
            coloring_score += 0.6
            detection_methods.append("filename_pattern")
            logger.info(f"Detected Coloring data from filename pattern: {filename}")
        elif 'Tracing' in filename:
            tracing_score += 0.4
            detection_methods.append("filename_keyword")
            logger.info(f"Detected possible Tracing data from filename keyword: {filename}")
        elif 'Coloring' in filename:
            coloring_score += 0.4
            detection_methods.append("filename_keyword")
            logger.info(f"Detected possible Coloring data from filename keyword: {filename}")

    # Method 2: Check columns (high confidence)
    if 'distance' in df.columns and 'camFrame' in df.columns and 'isDragging' in df.columns:
        tracing_score += 0.3
        detection_methods.append("column_structure")
        logger.info("Detected Tracing data from column structure (distance, camFrame, isDragging)")
    elif 'color' in df.columns and 'completionPerc' in df.columns:
        coloring_score += 0.3
        detection_methods.append("column_structure")
        logger.info("Detected Coloring data from column structure (color, completionPerc)")

    # Method 3: Check touchPhase values (medium confidence)
    # Count occurrences of different touchPhase values
    touch_phases = df['touchPhase'].value_counts().to_dict()

    # Tracing touchPhase indicators: 'B', 'M', 'S', 'E'
    tracing_phases = {'B': 0, 'M': 0, 'S': 0, 'E': 0}
    for phase, count in touch_phases.items():
        if phase in tracing_phases:
            tracing_phases[phase] = count

    # Coloring touchPhase indicators: 'Began', 'Moved', 'Stationary', 'Ended', 'Canceled'
    coloring_phases = {'Began': 0, 'Moved': 0, 'Stationary': 0, 'Ended': 0, 'Canceled': 0}
    for phase, count in touch_phases.items():
        if phase in coloring_phases:
            coloring_phases[phase] = count

    # Calculate phase scores
    tracing_phase_count = sum(tracing_phases.values())
    coloring_phase_count = sum(coloring_phases.values())
    total_phases = len(df['touchPhase'].dropna())

    if total_phases > 0:
        tracing_phase_ratio = tracing_phase_count / total_phases
        coloring_phase_ratio = coloring_phase_count / total_phases

        # Check for mixed data (potential ambiguity)
        if tracing_phase_ratio > 0.1 and coloring_phase_ratio > 0.1:
            ambiguities.append(f"Mixed touchPhase values detected: Tracing {tracing_phase_ratio:.2f}, Coloring {coloring_phase_ratio:.2f}")

            # Determine dominant type
            if tracing_phase_ratio > coloring_phase_ratio:
                tracing_score += 0.2 * tracing_phase_ratio
                detection_methods.append("touchphase_values")
                logger.warning(f"Mixed data with dominant Tracing touchPhase values: {tracing_phases}")
            else:
                coloring_score += 0.2 * coloring_phase_ratio
                detection_methods.append("touchphase_values")
                logger.warning(f"Mixed data with dominant Coloring touchPhase values: {coloring_phases}")
        elif tracing_phase_ratio > 0.5:
            tracing_score += 0.2
            detection_methods.append("touchphase_values")
            logger.info(f"Detected Tracing data from touchPhase values: {tracing_phases}")
        elif coloring_phase_ratio > 0.5:
            coloring_score += 0.2
            detection_methods.append("touchphase_values")
            logger.info(f"Detected Coloring data from touchPhase values: {coloring_phases}")

    # Method 4: Check for specific sequence patterns (low confidence)
    # Sample a subset of the data to check for sequence patterns
    if len(df) > 100:
        sample_df = df.sample(min(100, len(df)))
    else:
        sample_df = df

    # Check for Tracing-specific patterns
    if 'B' in sample_df['touchPhase'].values and 'E' in sample_df['touchPhase'].values:
        tracing_score += 0.1
        detection_methods.append("sequence_pattern")
        logger.info("Detected Tracing data from B→E sequence pattern")

    # Check for Coloring-specific patterns
    if 'Began' in sample_df['touchPhase'].values and 'Ended' in sample_df['touchPhase'].values:
        coloring_score += 0.1
        detection_methods.append("sequence_pattern")
        logger.info("Detected Coloring data from Began→Ended sequence pattern")

    # Determine final data type based on scores
    # Calculate content-only scores (excluding filename)
    content_tracing_score = tracing_score
    content_coloring_score = coloring_score

    if "filename_pattern" in detection_methods or "filename_keyword" in detection_methods:
        # Subtract filename-based scores
        if "filename_pattern" in detection_methods:
            if 'Tracing_' in filename:
                content_tracing_score -= 0.6
            elif 'Coloring_' in filename:
                content_coloring_score -= 0.6

        if "filename_keyword" in detection_methods:
            if 'Tracing' in filename and 'Tracing_' not in filename:
                content_tracing_score -= 0.4
            elif 'Coloring' in filename and 'Coloring_' not in filename:
                content_coloring_score -= 0.4

    # Check for misleading filename (when content strongly suggests a different type)

    # If we have strong content evidence for one type
    if content_tracing_score >= 0.3 and "filename_pattern" in detection_methods and 'Coloring_' in filename:
        logger.warning(f"Misleading filename detected: filename suggests Coloring but content suggests Tracing")
        data_type = 'Tracing'
        confidence = content_tracing_score
    elif content_coloring_score >= 0.3 and "filename_pattern" in detection_methods and 'Tracing_' in filename:
        logger.warning(f"Misleading filename detected: filename suggests Tracing but content suggests Coloring")
        data_type = 'Coloring'
        confidence = content_coloring_score
    # Normal case - use total scores
    elif tracing_score > coloring_score:
        data_type = 'Tracing'
        confidence = min(1.0, tracing_score)
    elif coloring_score > tracing_score:
        data_type = 'Coloring'
        confidence = min(1.0, coloring_score)
    else:
        data_type = 'Unknown'
        confidence = 0.0

    # Log detection results
    if data_type == 'Unknown':
        logger.warning("Could not determine data type, defaulting to Unknown")
    else:
        logger.info(f"Detected {data_type} data with {confidence:.2f} confidence using methods: {', '.join(detection_methods)}")

    # Log any ambiguities
    for ambiguity in ambiguities:
        logger.warning(f"Ambiguity in data type detection: {ambiguity}")

    # For backward compatibility, return just the data_type when used in existing code
    return data_type

def load_and_sort_data(csv_path, chunk_size=100000):
    """
    Load CSV data and verify/ensure sorting by time with memory optimization for large files.

    Args:
        csv_path (str): Path to the CSV file to load
        chunk_size (int): Number of rows to process at a time for large files

    Returns:
        DataFrame: The loaded and sorted DataFrame

    Raises:
        Exception: If there's an error loading or sorting the data
    """
    try:
        # First check file size to determine approach
        file_size = os.path.getsize(csv_path)
        file_size_mb = file_size / (1024 * 1024)  # Convert to MB

        # For small files (< 100MB), load directly with optimized dtypes
        if file_size_mb < 100:
            logger.info(f"Loading file {csv_path} ({file_size_mb:.1f} MB) using standard approach")

            # Use optimized dtypes to reduce memory usage
            df = pd.read_csv(csv_path)

            # Optimize numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')

            # Check if data is already sorted by time
            is_sorted = df['time'].is_monotonic_increasing

            if is_sorted:
                logger.info(f"Loaded {len(df)} rows from {csv_path} - data is already sorted by time")
            else:
                # Sort by time in ascending order if not already sorted
                logger.info(f"Data not chronologically sorted, performing sort operation")
                df = df.sort_values(by='time')
                logger.info(f"Successfully sorted {len(df)} rows by time")

            return df

        # For large files, use chunking approach
        else:
            logger.info(f"Loading large file {csv_path} ({file_size_mb:.1f} MB) using chunked processing")

            # First pass: determine column types and count rows
            dtypes = {}
            total_rows = 0

            # Sample the first chunk to determine dtypes
            sample_df = pd.read_csv(csv_path, nrows=1000)

            # Determine optimal dtypes based on the sample
            for col in sample_df.select_dtypes(include=['float64']).columns:
                dtypes[col] = 'float32'
            for col in sample_df.select_dtypes(include=['int64']).columns:
                dtypes[col] = 'int32'
            for col in sample_df.select_dtypes(include=['object']).columns:
                if col == 'touchPhase':  # Categorical column
                    dtypes[col] = 'category'

            # Count total rows
            with pd.read_csv(csv_path, chunksize=chunk_size) as reader:
                for _ in reader:
                    total_rows += len(_)

            logger.info(f"File contains {total_rows} rows, loading with optimized dtypes")

            # Second pass: load with optimized dtypes
            chunks = []
            with pd.read_csv(csv_path, dtype=dtypes, chunksize=chunk_size) as reader:
                for chunk in reader:
                    # Sort each chunk by time
                    chunks.append(chunk.sort_values(by='time'))
                    logger.info(f"Processed chunk with {len(chunk)} rows")

            # Combine all chunks and sort again
            df = pd.concat(chunks)
            df = df.sort_values(by='time')

            logger.info(f"Successfully loaded and sorted {len(df)} rows from {csv_path}")
            return df

    except pd.errors.EmptyDataError:
        logger.error(f"File {csv_path} is empty")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parser error in {csv_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading or sorting data from {csv_path}: {e}")
        raise

def segment_sequences(df):
    """
    Segment sequences based on touchPhase.
    For each fingerId, a sequence is considered complete only when it has a proper ending event.
    A new sequence ID is started only after the current sequence has properly ended.
    Uses a more sophisticated approach to ensure complete touch interactions.

    Additional tracking for sequence completeness:
    - sequence_interrupted: Tracks sequences interrupted by a new begin event
    - orphaned_events: Tracks events that occur after a sequence ends but before a new one begins
    - multiple_end_events: Tracks sequences with multiple ending events
    - improper_sequence_order: Tracks sequences with unexpected touchPhase order

    This implementation uses optimized operations where possible for better performance.
    """
    # Initialize seqId column with zeros
    df['seqId'] = 0

    # Initialize columns for tracking sequence completeness issues
    df['sequence_interrupted'] = False
    df['orphaned_events'] = False
    df['multiple_end_events'] = False
    df['improper_sequence_order'] = False

    # Get unique finger IDs
    finger_ids = df['fingerId'].unique()

    # Determine if this is a Coloring or Tracing file based on touchPhase values
    has_began = 'Began' in df['touchPhase'].values
    has_b = 'B' in df['touchPhase'].values

    if has_began:
        logger.info("Detected Coloring data format (touchPhase='Began')")
        begin_phase = 'Began'
        end_phases = ['Ended', 'Canceled']
        # Expected sequence for Coloring: "Began" → "Moved"/"Stationary" → "Ended"
        middle_phases = ['Moved', 'Stationary']
    elif has_b:
        logger.info("Detected Tracing data format (touchPhase='B')")
        begin_phase = 'B'
        end_phases = ['E', 'S']
        # Expected sequence for Tracing: "B" → "M"/"S" → "E"
        middle_phases = ['M', 'S']
    else:
        logger.warning("Could not determine begin phase type, defaulting to both 'Began' and 'B'")
        begin_phase = None  # Will handle both cases below
        end_phases = ['Ended', 'Canceled', 'E', 'S']
        middle_phases = ['Moved', 'Stationary', 'M', 'S']

    # Create a function to process each finger's data more efficiently
    def process_finger_data(finger_df, begin_phase, middle_phases, end_phases):
        # Sort by time to ensure chronological order
        finger_df = finger_df.sort_values(by='time').copy()

        # Get the touchPhase values as a numpy array for faster access
        touch_phases = finger_df['touchPhase'].values
        n_rows = len(finger_df)

        # Initialize arrays for tracking using numpy for better performance
        seq_ids = np.zeros(n_rows, dtype=int)
        interrupted = np.zeros(n_rows, dtype=bool)
        orphaned = np.zeros(n_rows, dtype=bool)
        multiple_end_events = np.zeros(n_rows, dtype=bool)
        improper_sequence_order = np.zeros(n_rows, dtype=bool)

        # Initialize sequence tracking variables
        current_seq_id = 0
        in_sequence = False
        last_end_idx = -1
        current_seq_start_idx = -1

        # Track end events per sequence
        end_events_count = {}

        # Track phase order
        last_phase_type = None

        # Process each row to assign sequence IDs
        for i in range(n_rows):
            touch_phase = touch_phases[i]

            # Check phase types
            if begin_phase:
                is_begin = touch_phase == begin_phase
            else:
                is_begin = touch_phase == 'Began' or touch_phase == 'B'

            is_middle = touch_phase in middle_phases
            is_end = touch_phase in end_phases

            if is_begin and not in_sequence:
                # Start a new sequence
                current_seq_id += 1
                in_sequence = True
                current_seq_start_idx = i
                end_events_count[current_seq_id] = 0
                last_phase_type = 'begin'
            elif is_end and in_sequence:
                # Check for improper phase order
                if last_phase_type == 'begin':
                    improper_sequence_order[current_seq_start_idx:i+1] = True

                # End the current sequence
                in_sequence = False
                last_end_idx = i
                end_events_count[current_seq_id] = end_events_count.get(current_seq_id, 0) + 1
            elif is_begin and in_sequence:
                # Found a new begin phase while still in a sequence
                # Mark the current sequence as interrupted
                interrupted[current_seq_start_idx:i] = True
                improper_sequence_order[current_seq_start_idx:i] = True

                # Start a new sequence
                current_seq_id += 1
                current_seq_start_idx = i
                end_events_count[current_seq_id] = 0
                last_phase_type = 'begin'
                in_sequence = True
            elif is_middle and in_sequence:
                # Middle phase in a sequence
                last_phase_type = 'middle'
            elif is_end and not in_sequence:
                # Found an end event outside of a sequence
                orphaned[i] = True
            elif not is_begin and not is_end and not in_sequence and last_end_idx >= 0 and i > last_end_idx:
                # This is an event after an end but before a new begin
                orphaned[i] = True

            # Assign the current sequence ID
            if in_sequence or i == last_end_idx:
                seq_ids[i] = current_seq_id

        # Mark sequences with multiple end events using vectorized operations
        for seq_id, count in end_events_count.items():
            if count > 1:
                multiple_end_events[seq_ids == seq_id] = True

        # Update the finger_df with sequence IDs and tracking information
        finger_df['seqId'] = seq_ids
        finger_df['sequence_interrupted'] = interrupted
        finger_df['orphaned_events'] = orphaned
        finger_df['multiple_end_events'] = multiple_end_events
        finger_df['improper_sequence_order'] = improper_sequence_order

        return finger_df

    # Process fingers in parallel using a list comprehension for better performance
    processed_dfs = []
    for finger_id in finger_ids:
        # Get the subset for this finger ID
        finger_df = df[df['fingerId'] == finger_id]
        if len(finger_df) > 0:
            # Process this finger's data
            processed_df = process_finger_data(finger_df, begin_phase, middle_phases, end_phases)
            processed_dfs.append(processed_df)

    # Combine all processed finger dataframes if any exist
    if processed_dfs:
        # Use concat once instead of updating the original dataframe multiple times
        processed_df = pd.concat(processed_dfs)

        # Create a mapping from index to new values for each column
        # This is more efficient than updating each column separately
        for col in ['seqId', 'sequence_interrupted', 'orphaned_events', 'multiple_end_events', 'improper_sequence_order']:
            # Update the original dataframe with processed values
            df.loc[processed_df.index, col] = processed_df[col].values

    logger.info(f"Segmented sequences: found {df['seqId'].max()} unique sequences")
    return df

def compute_sequence_metrics(df):
    """
    Compute metrics for each sequence (fingerId, seqId).
    Returns the sequence metrics DataFrame.
    """
    # Group by fingerId and seqId
    grouped = df.groupby(['fingerId', 'seqId'])

    # Initialize the sequence metrics DataFrame
    seq_metrics = pd.DataFrame()

    # Calculate metrics for each group
    seq_metrics['startTime'] = grouped['time'].min()

    # Determine if this is a Coloring or Tracing file based on touchPhase values
    has_ended = 'Ended' in df['touchPhase'].values
    has_s = 'S' in df['touchPhase'].values

    # Calculate endTime based on the data type
    if has_ended and not has_s:
        logger.info("Using 'Ended' as end phase for Coloring data")
        # For Coloring data: Calculate endTime only if there's at least one "Ended" touchPhase
        def get_end_time(group):
            if 'Ended' in group['touchPhase'].values:
                ended_rows = group[group['touchPhase'] == 'Ended']
                return ended_rows['time'].max()
            return np.nan
    elif has_s and not has_ended:
        logger.info("Using 'S' as end phase for Tracing data")
        # For Tracing data: Calculate endTime only if there's at least one "S" touchPhase
        def get_end_time(group):
            if 'S' in group['touchPhase'].values:
                ended_rows = group[group['touchPhase'] == 'S']
                return ended_rows['time'].max()
            return np.nan
    else:
        logger.info("Checking for both 'Ended' and 'S' as end phases")
        # Handle both cases
        def get_end_time(group):
            if 'Ended' in group['touchPhase'].values:
                ended_rows = group[group['touchPhase'] == 'Ended']
                return ended_rows['time'].max()
            elif 'S' in group['touchPhase'].values:
                ended_rows = group[group['touchPhase'] == 'S']
                return ended_rows['time'].max()
            return np.nan

    seq_metrics['endTime'] = grouped.apply(get_end_time)

    # Calculate duration in seconds
    seq_metrics['durationSec'] = (seq_metrics['endTime'] - seq_metrics['startTime'])

    # Count points in each sequence
    seq_metrics['nPoints'] = grouped.size()

    # Check if any row has touchPhase == "Canceled"
    seq_metrics['hasCanceled'] = grouped.apply(lambda g: 'Canceled' in g['touchPhase'].values)

    # Reset index to make fingerId and seqId regular columns
    seq_metrics = seq_metrics.reset_index()

    logger.info(f"Computed metrics for {len(seq_metrics)} sequences")
    return seq_metrics

def validate_and_resolve_flag_conflicts(flags):
    """
    Validate flags and resolve conflicts based on priority rules.

    Args:
        flags (list): List of flags to validate

    Returns:
        list: Validated flags with conflicts resolved
    """
    if not flags:
        return flags

    # Define incompatible flag pairs with priority order (first flag has higher priority)
    incompatible_pairs = [
        # A sequence cannot both be missing an end event and have multiple end events
        ('multiple_end_events', 'missing_Ended'),
        ('multiple_end_events', 'missing_E'),

        # A sequence that's interrupted typically won't have multiple end events
        ('sequence_interrupted', 'multiple_end_events'),

        # An event can't be both missing a begin event and orphaned
        ('orphaned_events', 'missing_Began'),
        ('orphaned_events', 'missing_B'),

        # Sequence interruption takes precedence over missing end events
        ('sequence_interrupted', 'missing_Ended'),
        ('sequence_interrupted', 'missing_E'),
    ]

    # Resolve conflicts based on priority
    for high_priority, low_priority in incompatible_pairs:
        if high_priority in flags and low_priority in flags:
            logger.debug(f"Resolving flag conflict: keeping '{high_priority}', removing '{low_priority}'")
            flags.remove(low_priority)

    return flags

def validate_coloring_sequence_pattern_by_touchdata_id(sequence_data):
    """
    Validate if a Coloring sequence follows the specific valid pattern using Touchdata_id and event_index:
    1. Starts with "Began"
    2. Ends with "Ended"
    3. Between "Began" and "Ended", contains only:
       - Any number of "Moved" or "Stationary" events
       - At most one "Canceled" event
    4. Follows pattern: "Began" → ("Moved"/"Stationary")* → (optional "Canceled") → "Ended"

    Args:
        sequence_data (DataFrame): Data for a single Touchdata_id sequence

    Returns:
        bool: True if sequence follows the valid pattern, False otherwise
    """
    if len(sequence_data) == 0:
        return False

    # Sort by event_index to ensure proper chronological order
    sequence_data = sequence_data.sort_values('event_index')
    touch_phases = sequence_data['touchPhase'].tolist()

    # Check if sequence starts with "Began"
    if touch_phases[0] != 'Began':
        return False

    # Check if sequence ends with "Ended"
    if touch_phases[-1] != 'Ended':
        return False

    # Check the middle events (between first and last)
    middle_phases = touch_phases[1:-1]

    # Count "Canceled" events
    canceled_count = middle_phases.count('Canceled')

    # At most one "Canceled" event is allowed
    if canceled_count > 1:
        return False

    # Check if all middle events are valid
    valid_middle_phases = {'Moved', 'Stationary', 'Canceled'}
    for phase in middle_phases:
        if phase not in valid_middle_phases:
            return False

    # If there's a "Canceled" event, it should be the last middle event
    if canceled_count == 1:
        # Find the position of "Canceled" in middle phases
        canceled_index = middle_phases.index('Canceled')
        # Check if "Canceled" is the last middle event
        if canceled_index != len(middle_phases) - 1:
            return False

    return True

def validate_coloring_sequences_by_touchdata_id(df):
    """
    Validate all Coloring sequences in a DataFrame using Touchdata_id and event_index.
    Groups data by Touchdata_id and validates each sequence independently.

    Args:
        df (DataFrame): DataFrame containing Coloring data with Touchdata_id and event_index columns

    Returns:
        dict: Dictionary mapping Touchdata_id to validation result (True/False)
    """
    validation_results = {}

    # Check if required columns exist
    if 'Touchdata_id' not in df.columns or 'event_index' not in df.columns:
        logger.warning("Touchdata_id or event_index columns not found, falling back to legacy validation")
        return {}

    # Group by Touchdata_id
    grouped = df.groupby('Touchdata_id')

    for touchdata_id, group_data in grouped:
        # Validate this Touchdata_id sequence
        is_valid = validate_coloring_sequence_pattern_by_touchdata_id(group_data)
        validation_results[touchdata_id] = is_valid

        # Log validation result for debugging
        touch_phases = group_data.sort_values('event_index')['touchPhase'].tolist()
        logger.debug(f"Touchdata_id {touchdata_id}: {touch_phases} -> {'Valid' if is_valid else 'Invalid'}")

    return validation_results

def validate_coloring_sequence_pattern(sequence_data):
    """
    Legacy function for backward compatibility.
    Automatically detects if enhanced fields are available and uses appropriate validation.

    Args:
        sequence_data (DataFrame): Data for a single sequence

    Returns:
        bool: True if sequence follows the valid pattern, False otherwise
    """
    if len(sequence_data) == 0:
        return False

    # Check if enhanced fields are available
    if 'event_index' in sequence_data.columns:
        # Use enhanced validation
        return validate_coloring_sequence_pattern_by_touchdata_id(sequence_data)
    else:
        # Use legacy validation with time-based sorting
        # Sort by time to ensure proper order
        sequence_data = sequence_data.sort_values('time')
        touch_phases = sequence_data['touchPhase'].tolist()

        # Check if sequence starts with "Began"
        if touch_phases[0] != 'Began':
            return False

        # Check if sequence ends with "Ended"
        if touch_phases[-1] != 'Ended':
            return False

        # Check the middle events (between first and last)
        middle_phases = touch_phases[1:-1]

        # Count "Canceled" events
        canceled_count = middle_phases.count('Canceled')

        # At most one "Canceled" event is allowed
        if canceled_count > 1:
            return False

        # Check if all middle events are valid
        valid_middle_phases = {'Moved', 'Stationary', 'Canceled'}
        for phase in middle_phases:
            if phase not in valid_middle_phases:
                return False

        # If there's a "Canceled" event, it should be the last middle event
        if canceled_count == 1:
            # Find the position of "Canceled" in middle phases
            canceled_index = middle_phases.index('Canceled')
            # Check if "Canceled" is the last middle event
            if canceled_index != len(middle_phases) - 1:
                return False

        return True

def apply_flag_rules(df, seq_metrics):
    """
    Apply flag rules to each sequence and update both DataFrames.
    Handles both Coloring and Tracing data types with their specific flag conditions.
    Includes enhanced flags for sequence completeness with validation to prevent contradictory flags.
    Uses vectorized operations for improved performance.

    For Coloring data, implements specific sequence validation:
    - Valid sequences (no flags) must follow: "Began" → ("Moved"/"Stationary")* → (optional "Canceled") → "Ended"
    - Invalid sequences receive appropriate flags based on existing rules
    """
    # Initialize flags column in sequence metrics with empty lists
    seq_metrics['flags'] = [[] for _ in range(len(seq_metrics))]

    # Determine if this is Coloring or Tracing data based on columns and touchPhase values
    is_tracing = ('distance' in df.columns and 'camFrame' in df.columns) or ('B' in df['touchPhase'].values)
    is_coloring = ('color' in df.columns and 'completionPerc' in df.columns) or ('Began' in df['touchPhase'].values)

    if is_tracing:
        logger.info("Applying Tracing data flag rules")
    elif is_coloring:
        logger.info("Applying Coloring data flag rules with enhanced sequence validation")
    else:
        logger.warning("Could not determine data type, applying both Coloring and Tracing flag rules")
        # Default to both if can't determine
        is_tracing = True
        is_coloring = True

    # Define the maximum time gap in milliseconds for data
    MAX_TIME_GAP_MS = 100

    # Filter out sequences with seqId = 0
    valid_sequences = seq_metrics[seq_metrics['seqId'] > 0]

    # Group data by fingerId and seqId for vectorized operations
    grouped = df.groupby(['fingerId', 'seqId'])

    # Create a dictionary to store flags for each sequence
    sequence_flags = {}

    # Process all sequences using vectorized operations where possible
    if is_coloring:
        # Use enhanced Touchdata_id-based validation if available
        touchdata_validation_results = validate_coloring_sequences_by_touchdata_id(df)

        if touchdata_validation_results:
            # Use Touchdata_id-based validation
            logger.info(f"Using Touchdata_id-based validation for {len(touchdata_validation_results)} sequences")

            # Initialize flags for each sequence based on Touchdata_id validation
            for idx, row in valid_sequences.iterrows():
                finger_id, seq_id = row['fingerId'], row['seqId']
                flags = []

                # Get the Touchdata_id for this sequence
                seq_mask = (df['fingerId'] == finger_id) & (df['seqId'] == seq_id)
                sequence_data = df[seq_mask]

                if len(sequence_data) > 0:
                    # Get the Touchdata_id from the sequence data
                    touchdata_id = sequence_data['Touchdata_id'].iloc[0] if 'Touchdata_id' in sequence_data.columns else None

                    # Check if this Touchdata_id sequence is valid
                    is_valid_pattern = touchdata_validation_results.get(touchdata_id, False) if touchdata_id is not None else False

                    # Only apply flags if the sequence doesn't follow the valid pattern
                    if not is_valid_pattern:
                        # Apply traditional flag checks for invalid sequences

                        # Check if first event is not "Began"
                        first_phase = sequence_data.sort_values('event_index' if 'event_index' in sequence_data.columns else 'time').iloc[0]['touchPhase']
                        if first_phase != 'Began':
                            flags.append('missing_Began')

                        # Check if there's no "Ended" event
                        if 'Ended' not in sequence_data['touchPhase'].values:
                            flags.append('missing_Ended')

                        # Check for time gaps
                        if len(sequence_data) > 1:
                            sorted_data = sequence_data.sort_values('time')
                            time_diffs = sorted_data['time'].diff().dropna()
                            if (time_diffs > MAX_TIME_GAP_MS).any():
                                flags.append('sequence_gap')

                        # Check duration in seconds (from seq_metrics)
                        if not np.isnan(row['durationSec']) and row['durationSec'] < 0.01:
                            flags.append('short_duration')

                        # Check number of points (from seq_metrics)
                        if row['nPoints'] < 3:
                            flags.append('too_few_points')

                        # Check if has canceled events (from seq_metrics)
                        if row['hasCanceled']:
                            flags.append('has_canceled')

                # If sequence follows valid pattern, no flags are applied (empty list)
                sequence_flags[(finger_id, seq_id)] = flags
        else:
            # Fall back to legacy validation method
            logger.info("Falling back to legacy sequence validation method")

            # Get first touchPhase for each sequence
            first_touch_phases = grouped.apply(lambda g: g.sort_values('time').iloc[0]['touchPhase'] if len(g) > 0 else None)
            # Check if first event is not "Began"
            missing_began_seqs = first_touch_phases[first_touch_phases != 'Began'].index.tolist()

            # Check if there's no "Ended" event
            has_ended = grouped.apply(lambda g: 'Ended' in g['touchPhase'].values)
            missing_ended_seqs = has_ended[~has_ended].index.tolist()

            # Check for sequence gap (time gap between consecutive events)
            has_sequence_gap = grouped.apply(lambda g:
                (g.sort_values('time')['time'].diff().dropna() > MAX_TIME_GAP_MS).any()
                if len(g) > 1 else False)
            sequence_gap_seqs = has_sequence_gap[has_sequence_gap].index.tolist()

            # Initialize flags for each sequence with legacy validation
            for idx, row in valid_sequences.iterrows():
                finger_id, seq_id = row['fingerId'], row['seqId']
                flags = []

                # Get sequence data for pattern validation
                seq_mask = (df['fingerId'] == finger_id) & (df['seqId'] == seq_id)
                sequence_data = df[seq_mask]

                # Check if sequence follows the valid pattern
                is_valid_pattern = validate_coloring_sequence_pattern(sequence_data)

                # Only apply flags if the sequence doesn't follow the valid pattern
                if not is_valid_pattern:
                    # Add flags based on vectorized checks
                    if (finger_id, seq_id) in missing_began_seqs:
                        flags.append('missing_Began')

                    if (finger_id, seq_id) in missing_ended_seqs:
                        flags.append('missing_Ended')

                    if (finger_id, seq_id) in sequence_gap_seqs:
                        flags.append('sequence_gap')

                    # Check duration in seconds (from seq_metrics)
                    if not np.isnan(row['durationSec']) and row['durationSec'] < 0.01:
                        flags.append('short_duration')

                    # Check number of points (from seq_metrics)
                    if row['nPoints'] < 3:
                        flags.append('too_few_points')

                    # Check if has canceled events (from seq_metrics)
                    if row['hasCanceled']:
                        flags.append('has_canceled')

                # If sequence follows valid pattern, no flags are applied (empty list)
                sequence_flags[(finger_id, seq_id)] = flags

    if is_tracing:
        # Get first touchPhase for each sequence
        first_touch_phases = grouped.apply(lambda g: g.sort_values('time').iloc[0]['touchPhase'] if len(g) > 0 else None)
        # Check if first event is not "B"
        missing_b_seqs = first_touch_phases[first_touch_phases != 'B'].index.tolist()

        # Check if there's no "E" event
        has_e = grouped.apply(lambda g: 'E' in g['touchPhase'].values)
        missing_e_seqs = has_e[~has_e].index.tolist()

        # Check for invalid touchPhase values
        valid_touch_phases = ['B', 'M', 'S', 'E']
        has_invalid_phase = grouped.apply(lambda g:
            not g['touchPhase'].isin(valid_touch_phases).all())
        invalid_phase_seqs = has_invalid_phase[has_invalid_phase].index.tolist()

        # Check for zero distance (if distance column exists)
        if 'distance' in df.columns:
            zero_distance = grouped.apply(lambda g:
                g['distance'].sum() == 0 if len(g) > 1 else False)
            zero_distance_seqs = zero_distance[zero_distance].index.tolist()
        else:
            zero_distance_seqs = []

        # Check for time gaps
        has_time_gap = grouped.apply(lambda g:
            (g.sort_values('time')['time'].diff().dropna() > MAX_TIME_GAP_MS).any()
            if len(g) > 1 else False)
        time_gap_seqs = has_time_gap[has_time_gap].index.tolist()

        # Add Tracing flags to existing flags or initialize new ones
        for idx, row in valid_sequences.iterrows():
            finger_id, seq_id = row['fingerId'], row['seqId']
            flags = sequence_flags.get((finger_id, seq_id), [])

            # Add flags based on vectorized checks
            if (finger_id, seq_id) in missing_b_seqs:
                flags.append('missing_B')

            if (finger_id, seq_id) in missing_e_seqs:
                flags.append('missing_E')

            if (finger_id, seq_id) in invalid_phase_seqs:
                flags.append('invalid_TouchPhase')

            if (finger_id, seq_id) in zero_distance_seqs:
                flags.append('zero_distance')

            if (finger_id, seq_id) in time_gap_seqs:
                flags.append('time_gap')

            sequence_flags[(finger_id, seq_id)] = flags

    # Apply sequence completeness flags using vectorized operations
    if 'sequence_interrupted' in df.columns:
        has_interrupted = grouped['sequence_interrupted'].any()
        interrupted_seqs = has_interrupted[has_interrupted].index.tolist()

        for finger_id, seq_id in interrupted_seqs:
            if (finger_id, seq_id) in sequence_flags:
                sequence_flags[(finger_id, seq_id)].append('sequence_interrupted')

    if 'orphaned_events' in df.columns:
        has_orphaned = grouped['orphaned_events'].any()
        orphaned_seqs = has_orphaned[has_orphaned].index.tolist()

        for finger_id, seq_id in orphaned_seqs:
            if (finger_id, seq_id) in sequence_flags:
                sequence_flags[(finger_id, seq_id)].append('orphaned_events')

    if 'multiple_end_events' in df.columns:
        has_multiple_end = grouped['multiple_end_events'].any()
        multiple_end_seqs = has_multiple_end[has_multiple_end].index.tolist()

        for finger_id, seq_id in multiple_end_seqs:
            if (finger_id, seq_id) in sequence_flags:
                sequence_flags[(finger_id, seq_id)].append('multiple_end_events')

    if 'improper_sequence_order' in df.columns:
        has_improper_order = grouped['improper_sequence_order'].any()
        improper_order_seqs = has_improper_order[has_improper_order].index.tolist()

        for finger_id, seq_id in improper_order_seqs:
            if (finger_id, seq_id) in sequence_flags:
                sequence_flags[(finger_id, seq_id)].append('improper_sequence_order')

    # Validate and resolve flag conflicts for each sequence
    for (finger_id, seq_id), flags in sequence_flags.items():
        sequence_flags[(finger_id, seq_id)] = validate_and_resolve_flag_conflicts(flags)

    # Update flags in sequence metrics
    for idx, row in valid_sequences.iterrows():
        finger_id, seq_id = row['fingerId'], row['seqId']
        if (finger_id, seq_id) in sequence_flags:
            seq_metrics.at[idx, 'flags'] = sequence_flags[(finger_id, seq_id)]

    # Convert flags lists to strings in sequence metrics
    seq_metrics['flags'] = seq_metrics['flags'].apply(lambda x: ','.join(x) if x else '')

    # Create a mapping of (fingerId, seqId) to flags for efficient updating
    flag_mapping = seq_metrics.set_index(['fingerId', 'seqId'])['flags'].to_dict()

    # Create a new flags column
    df['flags'] = ''

    # Update flags in main DataFrame using vectorized operations
    for (finger_id, seq_id), flags_str in flag_mapping.items():
        mask = (df['fingerId'] == finger_id) & (df['seqId'] == seq_id)
        df.loc[mask, 'flags'] = flags_str

    # Handle rows with seqId = 0 (not part of any sequence)
    no_seq_mask = df['seqId'] == 0
    if no_seq_mask.any():
        # Check if these are orphaned events
        if 'orphaned_events' in df.columns:
            orphaned_mask = no_seq_mask & df['orphaned_events']
            if orphaned_mask.any():
                df.loc[orphaned_mask, 'flags'] = 'orphaned_events'

            # Check for rows that are just missing a begin phase
            missing_begin_mask = no_seq_mask & ~df['orphaned_events']
            if missing_begin_mask.any():
                if is_coloring:
                    df.loc[missing_begin_mask, 'flags'] = 'missing_Began'
                else:
                    df.loc[missing_begin_mask, 'flags'] = 'missing_B'

    logger.info("Applied flag rules to all sequences with vectorized operations and conflict resolution")
    return df, seq_metrics

def assemble_and_save_output(df, seq_metrics, output_path):
    """
    Assemble the two tables side by side and save to CSV.
    Handles the enhanced sequence completeness tracking columns.
    """
    # Drop the hasCanceled column from sequence metrics as it's not needed in the output
    seq_metrics = seq_metrics.drop(columns=['hasCanceled'])

    # Create a blank column to separate the tables
    blank_column_name = ' ' * 5  # Five spaces

    # Reorder columns in the main DataFrame according to the specified sequence
    # First, drop the accelerometer columns
    df = df.drop(columns=['accx', 'accy', 'accz'], errors='ignore')

    # Remove the tracking columns from the main output (they're reflected in the flags)
    tracking_columns = ['sequence_interrupted', 'orphaned_events', 'multiple_end_events', 'improper_sequence_order']
    df = df.drop(columns=tracking_columns, errors='ignore')

    # Determine if this is a Coloring or Tracing file based on columns
    is_tracing = 'distance' in df.columns and 'camFrame' in df.columns
    is_coloring = 'color' in df.columns and 'completionPerc' in df.columns

    if is_tracing:
        logger.info("Processing as Tracing data")
        # Define the base column order for Tracing data (includes seqId)
        base_column_order = ['fingerId', 'seqId', 'x', 'y', 'time', 'touchPhase', 'zone', 'flags']

        # Add Tracing-specific columns
        tracing_columns = ['distance', 'camFrame', 'isDragging']
        new_column_order = base_column_order.copy()

        # Insert Tracing-specific columns after touchPhase
        insert_index = base_column_order.index('zone')
        for col in reversed(tracing_columns):
            if col in df.columns:
                new_column_order.insert(insert_index, col)

    elif is_coloring:
        logger.info("Processing as Coloring data")

        # For Coloring data, use the exact column order specified for Google Sheets export
        if 'Touchdata_id' in df.columns and 'event_index' in df.columns:
            logger.info("Using enhanced Touchdata_id and event_index columns for Coloring data export")
            # Exact order: Touchdata_id, event_index, x, y, time, touchPhase, fingerId, color, completionPerc, zone, flags, quality_score, interaction_type, anomaly_flag, research_suitability
            new_column_order = ['Touchdata_id', 'event_index', 'x', 'y', 'time', 'touchPhase', 'fingerId', 'color', 'completionPerc', 'zone', 'flags', 'quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']
        else:
            logger.info("Using legacy seqId column for Coloring data export (enhanced fields not available)")
            # Legacy order with seqId instead of Touchdata_id and event_index
            new_column_order = ['fingerId', 'seqId', 'x', 'y', 'time', 'touchPhase', 'color', 'completionPerc', 'zone', 'flags', 'quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']

    else:
        # If can't determine type, use all available columns
        logger.warning("Could not determine data type (Coloring or Tracing), using all available columns")
        new_column_order = [col for col in df.columns if col != 'flags'] + ['flags']

    # Ensure all columns in new_column_order exist in df
    new_column_order = [col for col in new_column_order if col in df.columns]

    # Reorder columns
    df = df[new_column_order]

    # Add blank column
    df[blank_column_name] = ''

    # Determine the number of rows in each DataFrame
    rows_df = len(df)
    rows_seq = len(seq_metrics)

    # Create a combined DataFrame with the right number of rows
    combined_df = df.copy()

    # Add sequence metrics columns to the combined DataFrame
    for col in seq_metrics.columns:
        col_name = f"{blank_column_name}{col}"
        combined_df[col_name] = ''

        # Fill in values from sequence metrics
        for i in range(min(rows_df, rows_seq)):
            combined_df.iloc[i, combined_df.columns.get_loc(col_name)] = seq_metrics.iloc[i][col]

    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined output to {output_path}")

    # Get the filename without path and extension
    filename = os.path.basename(output_path)
    filename = os.path.splitext(filename)[0]

    # Return the completeness data for this file
    return collect_completeness_data(df, filename)

def collect_completeness_data(df, filename):
    """
    Collect completeness data from a processed DataFrame.

    Args:
        df (DataFrame): The processed DataFrame with flags
        filename (str): The name of the file being processed

    Returns:
        dict: A dictionary containing completeness data
    """
    try:
        # Count occurrences of each completeness flag
        completeness_flags = [
            'missing_Began', 'missing_Ended', 'missing_B', 'missing_E',
            'sequence_interrupted', 'improper_sequence_order',
            'multiple_end_events', 'orphaned_events'
        ]

        flag_counts = {}
        for flag in completeness_flags:
            count = df['flags'].str.contains(flag, regex=False).sum()
            if count > 0:
                flag_counts[flag] = count

        # Determine data type based on filename
        data_type = 'Tracing' if 'Tracing' in filename else 'Coloring' if 'Coloring' in filename else 'Unknown'

        # Count sequences with each flag - use appropriate grouping based on data type and available columns
        if data_type == 'Coloring' and 'Touchdata_id' in df.columns:
            # For Coloring data with enhanced fields, group by Touchdata_id
            seq_groups = df[df['Touchdata_id'].notna()].groupby('Touchdata_id')
            total_sequences = len(seq_groups)
            rows_with_no_sequence = df['Touchdata_id'].isna().sum()
            logger.info(f"Using Touchdata_id-based sequence grouping for completeness data: {total_sequences} sequences")
        elif 'seqId' in df.columns:
            # For Tracing data or legacy Coloring data, group by fingerId and seqId
            seq_groups = df[df['seqId'] > 0].groupby(['fingerId', 'seqId'])
            total_sequences = len(seq_groups)
            rows_with_no_sequence = (df['seqId'] == 0).sum()
            logger.info(f"Using legacy seqId-based sequence grouping for completeness data: {total_sequences} sequences")
        else:
            # Fallback if no sequence identification columns are available
            seq_groups = []
            total_sequences = 0
            rows_with_no_sequence = len(df)
            logger.warning("No sequence identification columns found for completeness data")

        seq_flag_counts = {}
        for flag in completeness_flags:
            # For each sequence, check if any row has this flag
            seq_with_flag = 0
            for _, group in seq_groups:
                if group['flags'].str.contains(flag, regex=False).any():
                    seq_with_flag += 1

            if seq_with_flag > 0:
                seq_flag_counts[f'sequences_with_{flag}'] = seq_with_flag
                seq_flag_counts[f'percent_sequences_with_{flag}'] = f"{(seq_with_flag / total_sequences * 100):.2f}%" if total_sequences > 0 else "0.00%"

        # Calculate sequences with any completeness issues
        sequences_with_issues = 0
        for flag in completeness_flags:
            seq_flag = f'sequences_with_{flag}'
            if seq_flag in seq_flag_counts:
                sequences_with_issues = max(sequences_with_issues, seq_flag_counts[seq_flag])

        # Create summary data
        summary_data = {
            'filename': filename,
            'data_type': data_type,
            'total_rows': len(df),
            'total_sequences': total_sequences,
            'rows_with_no_sequence': rows_with_no_sequence,
            'complete_sequences': total_sequences - sequences_with_issues,
            'incomplete_sequences': sequences_with_issues,
            'percent_complete_sequences': f"{((total_sequences - sequences_with_issues) / total_sequences * 100):.2f}%" if total_sequences > 0 else "0.00%"
        }

        # Add flag counts
        summary_data.update(flag_counts)
        summary_data.update(seq_flag_counts)

        return summary_data
    except Exception as e:
        logger.error(f"Error collecting completeness data for {filename}: {e}")
        # Return a minimal data set if there's an error
        return {
            'filename': filename,
            'data_type': 'Unknown',
            'error': str(e)
        }

# Function create_sequence_completeness_summary has been removed as per requirements
# to only generate summary.csv

def create_summary_csv(output_folder, file_stats):
    """
    Create a summary CSV file with statistics for all processed files.

    Args:
        output_folder (str): Path to the folder where the summary CSV will be saved
        file_stats (list): List of dictionaries containing statistics for each file
    """
    if not file_stats:
        logger.warning("No file statistics available to create summary")
        return

    # Create a copy of file_stats without the flag_counts dictionary and completeness_data
    clean_stats = []
    for stats in file_stats:
        clean_stat = {
            'filename': stats['filename'],
            'data_type': stats['data_type'],
            'flagged_rows': stats['flagged_rows'],
            'total_rows': stats['total_rows'],
            'flagged_percentage': stats['flagged_percentage']
        }

        # Extract flag counts into separate columns
        if 'flag_counts' in stats:
            for flag, count in stats['flag_counts'].items():
                clean_stat[f'flag_{flag}'] = count

        clean_stats.append(clean_stat)

    # Create a DataFrame from the cleaned statistics
    summary_df = pd.DataFrame(clean_stats)

    # Add percent sign to flagged_percentage values
    summary_df['flagged_percentage'] = summary_df['flagged_percentage'].apply(lambda x: f"{x}%")

    # Save the summary to CSV
    summary_path = os.path.join(output_folder, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Created summary CSV at {summary_path}")

def count_flagged_rows(df):
    """
    Count the number of rows that have flags and analyze flag types.

    Args:
        df (DataFrame): The processed DataFrame with a 'flags' column

    Returns:
        tuple: (flagged_rows, total_rows, flagged_percentage, flag_counts)
    """
    # Count rows where flags column is not empty
    total_rows = len(df)
    flagged_rows = df['flags'].astype(bool).sum()

    # Calculate percentage (rounded to 2 decimal places)
    if total_rows > 0:
        flagged_percentage = round((flagged_rows / total_rows) * 100, 2)
    else:
        flagged_percentage = 0.0

    # Count occurrences of each flag type
    flag_counts = {}

    # Get all unique flags
    all_flags = set()
    for flags_str in df['flags'].dropna():
        if flags_str:
            all_flags.update(flags_str.split(','))

    # Count occurrences of each flag
    for flag in all_flags:
        if flag:  # Skip empty flags
            count = df['flags'].str.contains(flag, regex=False).sum()
            flag_counts[flag] = int(count)  # Convert to int to avoid numpy types

    return flagged_rows, total_rows, flagged_percentage, flag_counts

def process_coloring_data(df, input_path, output_path):
    """
    Process Coloring data with full sequence analysis and metrics.
    Now includes ML-based consolidated metadata enhancement.

    Args:
        df (DataFrame): The loaded and sorted DataFrame
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file

    Returns:
        tuple: (processed_df, file_stats_dict)
    """
    logger.info(f"Processing {input_path} as Coloring data")

    # Apply the full Coloring data processing pipeline
    df = segment_sequences(df)
    seq_metrics = compute_sequence_metrics(df)
    df, seq_metrics = apply_flag_rules(df, seq_metrics)

    # Add ML-based consolidated metadata columns (try enhanced first)
    if ML_ENHANCED_AVAILABLE:
        try:
            logger.info("Adding enhanced ML metadata columns...")
            # Run algorithm comparison on first file or periodically
            run_comparison = not os.path.exists("ML/models/algorithm_comparison_results.json")
            df = enhance_dataframe_with_advanced_ml(df, run_algorithm_comparison=run_comparison)
            logger.info("Successfully added enhanced ML metadata")
        except Exception as e:
            logger.warning(f"Enhanced ML enhancement failed: {e}")
            logger.info("Falling back to consolidated ML enhancer")
            # Fallback to consolidated enhancer
            if ML_ENHANCER_AVAILABLE:
                try:
                    ml_enhancer = ConsolidatedMLEnhancer()
                    df = ml_enhancer.enhance_dataframe(df)
                    logger.info("Successfully added consolidated ML metadata")
                except Exception as e2:
                    logger.warning(f"Consolidated ML enhancement also failed: {e2}")
                    logger.info("Continuing without ML metadata")
    elif ML_ENHANCER_AVAILABLE:
        try:
            logger.info("Adding ML consolidated metadata columns...")
            ml_enhancer = ConsolidatedMLEnhancer()
            df = ml_enhancer.enhance_dataframe(df)
            logger.info("Successfully added ML consolidated metadata")
        except Exception as e:
            logger.warning(f"ML enhancement failed: {e}")
            logger.info("Continuing without ML metadata")
    else:
        logger.info("ML enhancer not available, skipping ML metadata")

    # Count flagged rows and analyze flag types
    flagged_rows, total_rows, flagged_percentage, flag_counts = count_flagged_rows(df)

    # Save the processed file with sequence metrics and get completeness data
    completeness_data = assemble_and_save_output(df, seq_metrics, output_path)

    # Create file stats dictionary
    filename = os.path.basename(input_path)
    filename = os.path.splitext(filename)[0]  # Remove .csv extension

    file_stats = {
        'filename': filename,
        'data_type': 'Coloring',
        'flagged_rows': flagged_rows,
        'total_rows': total_rows,
        'flagged_percentage': flagged_percentage,
        'flag_counts': flag_counts,
        'completeness_data': completeness_data
    }

    return df, file_stats

def segment_tracing_sequences(df):
    """
    Segment sequences for Tracing data based on touchPhase.
    Each fingerId is tracked independently, and sequences are properly segmented
    based on the B → M/S → E lifecycle.

    Args:
        df (DataFrame): The DataFrame to segment

    Returns:
        DataFrame: The DataFrame with added sequence tracking columns
    """
    # Initialize seqId column with zeros
    df['seqId'] = 0

    # Initialize columns for tracking sequence issues
    df['ORPHANED_FINGER'] = False
    df['UNTERMINATED'] = False
    df['OVERLAPPING_FINGERIDS'] = False
    df['PHANTOM_MOVE'] = False

    # Get unique finger IDs, excluding the header row if it was accidentally included
    finger_ids = [fid for fid in df['fingerId'].unique() if str(fid).isdigit()]

    # Track active fingers (those that have started but not ended)
    active_fingers = set()

    # Process each fingerId separately
    for finger_id in finger_ids:
        # Create a mask for this finger ID
        mask = df['fingerId'] == finger_id

        if not mask.any():
            continue

        # Get the subset for this finger ID
        finger_df = df[mask].copy()

        # Sort by time to ensure chronological order
        finger_df = finger_df.sort_values(by='time')

        # Initialize sequence ID for this finger
        current_seq_id = 0
        in_sequence = False

        # Initialize arrays for tracking
        seq_ids = [0] * len(finger_df)
        orphaned = [False] * len(finger_df)
        unterminated = [False] * len(finger_df)
        phantom = [False] * len(finger_df)

        # Track previous coordinates and distance for PHANTOM_MOVE detection
        prev_x, prev_y, prev_distance = None, None, None

        # Iterate through the rows to assign sequence IDs
        for i, row in enumerate(finger_df.itertuples()):
            touch_phase = row.touchPhase

            # Check if this is a begin phase (B)
            is_begin = touch_phase == 'B'

            # Check if this is a middle phase (M or S)
            is_middle = touch_phase in ['M', 'S']

            # Check if this is an end phase (E)
            is_end = touch_phase == 'E'

            # Check for PHANTOM_MOVE: touchPhase is M, coordinates haven't changed, but distance increases
            if touch_phase == 'M' and prev_x is not None and prev_y is not None and prev_distance is not None:
                # Check if coordinates haven't changed (within a small threshold)
                coords_unchanged = abs(row.x - prev_x) < 0.01 and abs(row.y - prev_y) < 0.01
                # Check if distance has increased
                distance_increased = hasattr(row, 'distance') and row.distance > prev_distance

                # Only flag as PHANTOM_MOVE if coordinates don't change BUT distance increases
                if coords_unchanged and distance_increased:
                    phantom[i] = True

            # Update previous coordinates and distance
            prev_x, prev_y = row.x, row.y
            if hasattr(row, 'distance'):
                prev_distance = row.distance

            # Handle sequence state transitions
            if is_begin and not in_sequence:
                # Start a new sequence
                current_seq_id += 1
                in_sequence = True

                # Add this finger to active fingers
                active_fingers.add(finger_id)

            elif is_begin and in_sequence:
                # Found a new begin phase while still in a sequence
                # This indicates the previous sequence didn't end properly

                # Mark the current sequence as UNTERMINATED
                for j in range(i):
                    if seq_ids[j] == current_seq_id:
                        unterminated[j] = True

                # Start a new sequence
                current_seq_id += 1
                in_sequence = True

            elif is_end and in_sequence:
                # End the current sequence
                in_sequence = False

                # Remove this finger from active fingers
                if finger_id in active_fingers:
                    active_fingers.remove(finger_id)

            elif (is_middle or is_end) and not in_sequence:
                # Found a middle or end event without a begin event
                # This is an ORPHANED_FINGER event
                orphaned[i] = True

            # Assign the current sequence ID
            seq_ids[i] = current_seq_id

        # If we're still in a sequence at the end, mark it as UNTERMINATED
        if in_sequence:
            for i in range(len(finger_df)):
                if seq_ids[i] == current_seq_id:
                    unterminated[i] = True

            # Remove this finger from active fingers
            if finger_id in active_fingers:
                active_fingers.remove(finger_id)

        # Update the finger_df with sequence IDs and tracking information
        finger_df['seqId'] = seq_ids
        finger_df['ORPHANED_FINGER'] = orphaned
        finger_df['UNTERMINATED'] = unterminated
        finger_df['PHANTOM_MOVE'] = phantom

        # Update the original dataframe
        for col in ['seqId', 'ORPHANED_FINGER', 'UNTERMINATED', 'PHANTOM_MOVE']:
            df.loc[mask, col] = finger_df[col].values

    # Process OVERLAPPING_FINGERIDS by analyzing the entire dataset chronologically
    df_sorted = df.sort_values(by='time')
    active_fingers = set()

    for idx, row in enumerate(df_sorted.itertuples()):
        try:
            finger_id = int(row.fingerId)  # Convert to int to ensure it's a valid finger ID
            touch_phase = row.touchPhase

            if touch_phase == 'B':
                # If this finger is starting and other fingers are active, mark as overlapping
                if active_fingers and finger_id not in active_fingers:
                    df.loc[df_sorted.index[idx], 'OVERLAPPING_FINGERIDS'] = True

                # Add this finger to active set
                active_fingers.add(finger_id)

            elif touch_phase == 'E':
                # Remove this finger from active set
                if finger_id in active_fingers:
                    active_fingers.remove(finger_id)
        except (ValueError, TypeError):
            # Skip rows with non-numeric finger IDs (like header rows)
            continue

    logger.info(f"Segmented Tracing sequences: found {df['seqId'].max()} unique sequences")
    return df

def compute_tracing_metrics(df):
    """
    Compute metrics for Tracing data sequences.

    Args:
        df (DataFrame): The DataFrame with segmented sequences

    Returns:
        DataFrame: A DataFrame with sequence metrics
    """
    # Group by fingerId and seqId
    grouped = df.groupby(['fingerId', 'seqId'])

    # Initialize the sequence metrics DataFrame
    seq_metrics = pd.DataFrame()

    # Calculate basic metrics
    seq_metrics['startTime'] = grouped['time'].min()

    # Calculate endTime using 'E' as the end phase
    def get_tracing_end_time(group):
        if 'E' in group['touchPhase'].values:
            ended_rows = group[group['touchPhase'] == 'E']
            return ended_rows['time'].max()
        return np.nan

    seq_metrics['endTime'] = grouped.apply(get_tracing_end_time, include_groups=False)

    # Calculate duration in seconds
    seq_metrics['durationSec'] = (seq_metrics['endTime'] - seq_metrics['startTime'])

    # Count points in each sequence
    seq_metrics['nPoints'] = grouped.size()

    # Calculate total distance if available
    if 'distance' in df.columns:
        seq_metrics['totalDistance'] = grouped['distance'].sum()

    # Check for flag conditions
    seq_metrics['hasOrphanedFinger'] = grouped['ORPHANED_FINGER'].any()
    seq_metrics['isUnterminated'] = grouped['UNTERMINATED'].any()
    seq_metrics['hasOverlappingFingers'] = grouped['OVERLAPPING_FINGERIDS'].any()
    seq_metrics['hasPhantomMove'] = grouped['PHANTOM_MOVE'].any()

    # Reset index to make fingerId and seqId regular columns
    seq_metrics = seq_metrics.reset_index()

    logger.info(f"Computed metrics for {len(seq_metrics)} Tracing sequences")
    return seq_metrics

def apply_tracing_flag_rules(df, seq_metrics):
    """
    Apply flag rules specific to Tracing data.

    Args:
        df (DataFrame): The DataFrame with segmented sequences
        seq_metrics (DataFrame): The sequence metrics DataFrame

    Returns:
        tuple: (processed_df, updated_seq_metrics)
    """
    # Initialize flags column in sequence metrics with empty lists
    seq_metrics['flags'] = [[] for _ in range(len(seq_metrics))]

    # Filter out sequences with seqId = 0
    valid_sequences = seq_metrics[seq_metrics['seqId'] > 0]

    # Process each valid sequence
    for idx, row in valid_sequences.iterrows():
        finger_id = row['fingerId']
        seq_id = row['seqId']

        # Get sequence data
        seq_mask = (df['fingerId'] == finger_id) & (df['seqId'] == seq_id)
        seq_data = df[seq_mask]

        if len(seq_data) == 0:
            continue

        flags = []

        # Check for ORPHANED_FINGER
        if row['hasOrphanedFinger']:
            flags.append('ORPHANED_FINGER')

        # Check for UNTERMINATED
        if row['isUnterminated']:
            flags.append('UNTERMINATED')

        # Check for OVERLAPPING_FINGERIDS
        if row['hasOverlappingFingers']:
            flags.append('OVERLAPPING_FINGERIDS')

        # Check for PHANTOM_MOVE
        if row['hasPhantomMove']:
            flags.append('PHANTOM_MOVE')

        # Update flags in sequence metrics
        seq_metrics.at[idx, 'flags'] = flags

    # Convert flags lists to strings in sequence metrics
    seq_metrics['flags'] = seq_metrics['flags'].apply(lambda x: ','.join(x) if x else '')

    # Create a new flags column in the main DataFrame
    df['flags'] = ''

    # Apply flags directly from the tracking columns
    # This ensures each row gets the appropriate flags based on its own issues
    flag_columns = ['ORPHANED_FINGER', 'UNTERMINATED', 'OVERLAPPING_FINGERIDS', 'PHANTOM_MOVE']

    # For each row, collect all applicable flags
    for idx, row in df.iterrows():
        row_flags = []
        for flag_col in flag_columns:
            if row[flag_col]:
                row_flags.append(flag_col)

        if row_flags:
            df.at[idx, 'flags'] = ','.join(row_flags)

    logger.info("Applied Tracing flag rules")
    return df, seq_metrics

def assemble_tracing_output(df, seq_metrics, output_path):
    """
    Assemble the Tracing data output and save to CSV.

    According to the requirements:
    1. Do NOT add a seqId column to the main data section
    2. Remove the sequence metrics columns that appear after the main data
    3. Only include the original data columns plus a single "flags" column

    Args:
        df (DataFrame): The processed DataFrame
        seq_metrics (DataFrame): The sequence metrics DataFrame (used for internal tracking but not included in output)
        output_path (str): Path to save the output CSV file

    Returns:
        dict: Completeness data for summary
    """
    # Drop the tracking columns as they're reflected in the flags
    tracking_columns = ['ORPHANED_FINGER', 'UNTERMINATED', 'OVERLAPPING_FINGERIDS', 'PHANTOM_MOVE']
    df = df.drop(columns=tracking_columns, errors='ignore')

    # Remove the seqId column from the main data section as per requirements
    if 'seqId' in df.columns:
        df = df.drop(columns=['seqId'])
        logger.info("Removed seqId column from Tracing data output as per requirements")

    # Define the column order for Tracing data (without seqId) including ML metadata
    column_order = ['fingerId', 'x', 'y', 'time', 'touchPhase', 'distance',
                    'camFrame', 'isDragging', 'zone', 'flags', 'quality_score',
                    'interaction_type', 'anomaly_flag', 'research_suitability']

    # Ensure all columns in column_order exist in df
    column_order = [col for col in column_order if col in df.columns]

    # Reorder columns
    df = df[column_order]

    # Save to CSV without sequence metrics columns
    df.to_csv(output_path, index=False)
    logger.info(f"Saved Tracing output to {output_path} with simplified format")

    # Get the filename without path and extension
    filename = os.path.basename(output_path)
    filename = os.path.splitext(filename)[0]

    # Collect completeness data (still using the sequence metrics for internal tracking)
    completeness_data = collect_tracing_completeness_data(df, filename)

    return completeness_data

def collect_tracing_completeness_data(df, filename):
    """
    Collect completeness data for Tracing data.

    Note: This function is simplified since we no longer include seqId in the output.
    It now only counts flag occurrences without sequence-level statistics.

    Args:
        df (DataFrame): The processed DataFrame with flags
        filename (str): The name of the file being processed

    Returns:
        dict: A dictionary containing completeness data
    """
    try:
        # Count occurrences of each flag
        tracing_flags = ['ORPHANED_FINGER', 'UNTERMINATED', 'OVERLAPPING_FINGERIDS', 'PHANTOM_MOVE']

        flag_counts = {}
        for flag in tracing_flags:
            count = df['flags'].str.contains(flag, regex=False).sum()
            if count > 0:
                flag_counts[flag] = count

        # Create summary data (simplified version without sequence metrics)
        summary_data = {
            'filename': filename,
            'data_type': 'Tracing',
            'total_rows': len(df),
            'flagged_rows': df['flags'].astype(bool).sum(),
        }

        # Calculate flagged percentage
        if len(df) > 0:
            summary_data['flagged_percentage'] = f"{(summary_data['flagged_rows'] / len(df) * 100):.2f}%"
        else:
            summary_data['flagged_percentage'] = "0.00%"

        # Add flag counts
        summary_data.update(flag_counts)

        return summary_data
    except Exception as e:
        logger.error(f"Error collecting Tracing completeness data for {filename}: {e}")
        # Return a minimal data set if there's an error
        return {
            'filename': filename,
            'data_type': 'Tracing',
            'error': str(e)
        }

def process_tracing_data(df, input_path, output_path):
    """
    Process Tracing data with specialized sequence segmentation and flagging.
    Now includes ML-based consolidated metadata enhancement.

    This implementation correctly handles Tracing data with the following requirements:
    1. Each fingerId is tracked independently
    2. Sequences follow the B → M/S → E pattern
    3. 'E' is the correct end phase, not 'S'
    4. Implements the required flags: ORPHANED_FINGER, UNTERMINATED, OVERLAPPING_FINGERIDS, PHANTOM_MOVE

    Args:
        df (DataFrame): The loaded and sorted DataFrame
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file

    Returns:
        tuple: (processed_df, file_stats_dict)
    """
    logger.info(f"Processing {input_path} as Tracing data")

    # Segment sequences using the specialized Tracing logic
    df = segment_tracing_sequences(df)

    # Compute sequence metrics
    seq_metrics = compute_tracing_metrics(df)

    # Apply Tracing-specific flag rules
    df, seq_metrics = apply_tracing_flag_rules(df, seq_metrics)

    # Add ML-based consolidated metadata columns (try enhanced first)
    if ML_ENHANCED_AVAILABLE:
        try:
            logger.info("Adding enhanced ML metadata columns...")
            # Run algorithm comparison on first file or periodically
            run_comparison = not os.path.exists("ML/models/algorithm_comparison_results.json")
            df = enhance_dataframe_with_advanced_ml(df, run_algorithm_comparison=run_comparison)
            logger.info("Successfully added enhanced ML metadata")
        except Exception as e:
            logger.warning(f"Enhanced ML enhancement failed: {e}")
            logger.info("Falling back to consolidated ML enhancer")
            # Fallback to consolidated enhancer
            if ML_ENHANCER_AVAILABLE:
                try:
                    ml_enhancer = ConsolidatedMLEnhancer()
                    df = ml_enhancer.enhance_dataframe(df)
                    logger.info("Successfully added consolidated ML metadata")
                except Exception as e2:
                    logger.warning(f"Consolidated ML enhancement also failed: {e2}")
                    logger.info("Continuing without ML metadata")
    elif ML_ENHANCER_AVAILABLE:
        try:
            logger.info("Adding ML consolidated metadata columns...")
            ml_enhancer = ConsolidatedMLEnhancer()
            df = ml_enhancer.enhance_dataframe(df)
            logger.info("Successfully added ML consolidated metadata")
        except Exception as e:
            logger.warning(f"ML enhancement failed: {e}")
            logger.info("Continuing without ML metadata")
    else:
        logger.info("ML enhancer not available, skipping ML metadata")

    # Count flagged rows and analyze flag types
    flagged_rows, total_rows, flagged_percentage, flag_counts = count_flagged_rows(df)

    # Save the processed file with sequence metrics and get completeness data
    completeness_data = assemble_tracing_output(df, seq_metrics, output_path)

    # Create file stats dictionary
    filename = os.path.basename(input_path)
    filename = os.path.splitext(filename)[0]  # Remove .csv extension

    file_stats = {
        'filename': filename,
        'data_type': 'Tracing',
        'flagged_rows': flagged_rows,
        'total_rows': total_rows,
        'flagged_percentage': flagged_percentage,
        'flag_counts': flag_counts,
        'completeness_data': completeness_data
    }

    return df, file_stats


def batch_process_csv_files(input_folder, output_folder):
    """
    Process all CSV files in the input folder and generate a summary CSV.
    Uses enhanced data type detection with strict routing logic.
    """
    # Prepare output folder
    prepare_output_folder(output_folder)

    # Get list of CSV files
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {input_folder}")
    except Exception as e:
        logger.error(f"Failed to read files from {input_folder}: {e}")
        return 0

    if not csv_files:
        logger.warning(f"No CSV files found in {input_folder}")
        return 0

    # Process each CSV file and collect statistics
    successful_conversions = 0
    file_stats = []
    detection_stats = {
        'Coloring': 0,
        'Tracing': 0,
        'Unknown': 0,
        'Ambiguous': 0,
        'Error': 0
    }

    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_path = os.path.join(output_folder, csv_file)
        logger.info(f"Processing {input_path}")

        try:
            # Load and sort data
            df = load_and_sort_data(input_path)

            # Enhanced data type detection
            data_type = detect_data_type(df, csv_file)

            # Strict routing logic based on data type
            if data_type == 'Unknown':
                logger.warning(f"Unknown data type for {input_path}, skipping file")
                detection_stats['Unknown'] += 1
                continue

            # Process based on data type with strict routing
            if data_type == 'Coloring':
                detection_stats['Coloring'] += 1
                logger.info(f"Routing {csv_file} to Coloring data processing")
                _, file_stat = process_coloring_data(df, input_path, output_path)
            elif data_type == 'Tracing':
                detection_stats['Tracing'] += 1
                logger.info(f"Routing {csv_file} to Tracing data processing")
                _, file_stat = process_tracing_data(df, input_path, output_path)
            else:
                # This should never happen with the current implementation
                logger.error(f"Unexpected data type '{data_type}' for {input_path}, skipping file")
                detection_stats['Error'] += 1
                continue

            # Add file stats to the list
            file_stats.append(file_stat)

            logger.info(f"Successfully processed {input_path}")
            successful_conversions += 1

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            detection_stats['Error'] += 1

    # Create summary CSV
    if file_stats:
        create_summary_csv(output_folder, file_stats)

    # Log detection statistics
    logger.info("Data type detection statistics:")
    for data_type, count in detection_stats.items():
        if count > 0:
            logger.info(f"  {data_type}: {count} files")

    logger.info(f"Successfully processed {successful_conversions} out of {len(csv_files)} files")
    return successful_conversions

def validate_flag_consistency(df):
    """
    Validate flag consistency and report contradictions in a DataFrame.

    Args:
        df (DataFrame): DataFrame with flags applied

    Returns:
        DataFrame: DataFrame with contradiction information
    """
    # Define incompatible flag pairs
    incompatible_pairs = [
        ('missing_Ended', 'multiple_end_events'),
        ('missing_E', 'multiple_end_events'),
        ('sequence_interrupted', 'multiple_end_events'),
        ('orphaned_events', 'missing_Began'),
        ('orphaned_events', 'missing_B'),
        ('sequence_interrupted', 'missing_Ended'),
        ('sequence_interrupted', 'missing_E'),
    ]

    # Create results DataFrame
    results = []

    # Group by fingerId and seqId
    for (finger_id, seq_id), group in df.groupby(['fingerId', 'seqId']):
        if seq_id == 0:
            continue

        # Get flags for this sequence
        flags_str = group['flags'].iloc[0]
        if not flags_str:
            continue

        flags = flags_str.split(',')

        # Check for contradictions
        contradictions = []
        for flag1, flag2 in incompatible_pairs:
            if flag1 in flags and flag2 in flags:
                contradictions.append(f"{flag1} + {flag2}")

        if contradictions:
            results.append({
                'fingerId': finger_id,
                'seqId': seq_id,
                'flags': flags_str,
                'contradictions': ', '.join(contradictions),
                'nPoints': len(group)
            })

    return pd.DataFrame(results)

def test_flag_consistency(input_folder='data/raw/csv', output_folder='data/processed/flagged'):
    """
    Test flag consistency on processed data.

    Args:
        input_folder (str): Path to the folder with raw CSV files
        output_folder (str): Path to save the processed files and reports

    Returns:
        dict: Dictionary with test results
    """
    # Prepare output folder
    prepare_output_folder(output_folder)

    # Get list of CSV files
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {input_folder}")
    except Exception as e:
        logger.error(f"Failed to read files from {input_folder}: {e}")
        return {'error': str(e)}

    if not csv_files:
        logger.warning(f"No CSV files found in {input_folder}")
        return {'error': 'No CSV files found'}

    # Process each CSV file and collect validation results
    validation_results = {}

    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_path = os.path.join(output_folder, csv_file)
        logger.info(f"Testing flag consistency for {input_path}")

        try:
            # Load and sort data
            df = load_and_sort_data(input_path)

            # Enhanced data type detection
            data_type = detect_data_type(df, csv_file)

            # Strict routing logic based on data type
            if data_type == 'Unknown':
                logger.warning(f"Unknown data type for {input_path}, skipping file")
                validation_results[csv_file] = {'error': 'Unknown data type'}
                continue

            # Process based on data type with strict routing
            if data_type == 'Coloring':
                logger.info(f"Routing {csv_file} to Coloring data processing for testing")
                df, _ = process_coloring_data(df, input_path, output_path)
            elif data_type == 'Tracing':
                logger.info(f"Routing {csv_file} to Tracing data processing for testing")
                df, _ = process_tracing_data(df, input_path, output_path)
            else:
                # This should never happen with the current implementation
                logger.error(f"Unexpected data type '{data_type}' for {input_path}, skipping file")
                validation_results[csv_file] = {'error': f'Unexpected data type: {data_type}'}
                continue

            # Validate flag consistency
            contradictions = validate_flag_consistency(df)

            # Save contradictions report if any found
            if len(contradictions) > 0:
                filename = os.path.basename(input_path)
                filename = os.path.splitext(filename)[0]
                contradictions_path = os.path.join(output_folder, f"{filename}_contradictions.csv")
                contradictions.to_csv(contradictions_path, index=False)
                logger.warning(f"Found {len(contradictions)} sequences with contradictory flags in {filename}")

            # Store validation results
            validation_results[csv_file] = {
                'data_type': data_type,
                'total_sequences': df['seqId'].nunique() - 1,  # Exclude seqId 0
                'contradictions_found': len(contradictions),
                'contradictions_details': contradictions.to_dict('records') if len(contradictions) > 0 else []
            }

            logger.info(f"Successfully tested {input_path}")

        except Exception as e:
            logger.error(f"Error testing {input_path}: {e}")
            validation_results[csv_file] = {'error': str(e)}

    # Create summary report
    summary = {
        'total_files': len(csv_files),
        'files_with_contradictions': sum(1 for _, result in validation_results.items()
                                        if isinstance(result, dict) and result.get('contradictions_found', 0) > 0),
        'total_contradictions': sum(result.get('contradictions_found', 0)
                                   for _, result in validation_results.items()
                                   if isinstance(result, dict)),
        'file_results': validation_results
    }

    # Save summary to JSON
    import json
    with open(os.path.join(output_folder, 'flag_validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Flag consistency test completed. Found {summary['total_contradictions']} contradictions in {summary['files_with_contradictions']} files.")
    return summary

def main(input_folder='data/raw/csv', output_folder='data/processed/flagged'):
    """Main function to run the batch processing."""
    logger.info("Starting CSV data processing")
    batch_process_csv_files(input_folder, output_folder)
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
