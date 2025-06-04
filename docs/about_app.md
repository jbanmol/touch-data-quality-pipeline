
## Overview

The Touch Data Analysis Tool processes data from touch interactions to identify potential issues. It follows these main steps:

1. Convert JSON files to CSV format
2. Process the CSV files to identify and flag problematic touch sequences
3. Generate summary statistics
4. Export the results to Google Sheets for easy viewing and sharing

## Data Processing Pipeline

The core of this application is the data processing pipeline in `process_csv_data.py`. Here's how it works:

### 1. Loading and Sorting Data

The pipeline starts by loading data from CSV files and sorting all entries by time. This ensures that all touch events are in chronological order, which is essential for correctly identifying sequences.

```python
df = pd.read_csv(csv_path)
df['time'] = pd.to_numeric(df['time'])
df = df.sort_values(by='time')
```

### 2. Segmenting Sequences Based on touchPhase

The next step is to group touch events into sequences. This is a critical part of the analysis.

#### How Sequence Segmentation Works

Touch events in the data have a `touchPhase` value that indicates what type of event it is:
- `Began`: When a finger first touches the screen
- `Moved`: When a finger moves across the screen
- `Stationary`: When a finger remains still on the screen
- `Ended`: When a finger is lifted from the screen
- `Canceled`: When a touch is interrupted or canceled

The code identifies sequences by looking for `Began` events. Each time a `Began` event is encountered for a particular `fingerId`, a new sequence is started. All subsequent events for that finger are part of the same sequence until another `Began` event is encountered.

```python
# Initialize sequence ID counter
seq_counter = 0
# Create a dictionary to track the current sequence ID for each finger
current_seq_ids = {}
# Create a new column to store sequence IDs
df['seqId'] = -1

# Process rows in order
for idx, row in df.iterrows():
    finger_id = row['fingerId']
    touch_phase = row['touchPhase']

    # If this is a 'Began' phase, start a new sequence
    if touch_phase == 'Began':
        seq_counter += 1
        current_seq_ids[finger_id] = seq_counter

    # Assign the current sequence ID for this finger
    if finger_id in current_seq_ids:
        df.at[idx, 'seqId'] = current_seq_ids[finger_id]
```

This segmentation is crucial because a proper touch sequence should start with `Began`, include zero or more `Moved` events, and end with `Ended`. Any deviation from this pattern could indicate a problem.

### 3. Computing Sequence Metrics

After segmenting the data into sequences, the pipeline calculates important metrics for each sequence:

- `startTime`: When the sequence began
- `endTime`: When the sequence ended (if it has an `Ended` touchPhase)
- `durationSec`: How long the sequence lasted in seconds
- `nPoints`: How many touch points are in the sequence
- `hasCanceled`: Whether the sequence contains any `Canceled` touchPhase events

These metrics help identify potential issues with the touch sequences.

### 4. Applying Flag Rules

This is where the pipeline identifies potential problems in the touch sequences. Each sequence is checked against several flag rules.


## Sequence Segmentation

Understanding how touch sequences work is key to understanding this tool. A proper touch sequence follows this pattern:

1. It starts with a `Began` event when a finger first touches the screen
2. It may include multiple `Moved` or `Stationary` events as the finger moves across or remains still on the screen
3. It ends with an `Ended` event when the finger is lifted from the screen

Each finger (identified by `fingerId`) can only be involved in one sequence at a time. When a finger starts a new sequence (with a `Began` event), any previous sequence for that finger should have already ended.

The segmentation algorithm works by:
1. Tracking the current sequence ID for each finger
2. Starting a new sequence whenever a `Began` event is encountered
3. Assigning all events for a finger to its current sequence ID

This approach allows the tool to identify when sequences don't follow the expected pattern, which could indicate issues with the touch detection or data collection.

## Flag Rules

The tool applies several flag rules to identify potential issues in the touch sequences. These flags are organized into categories based on the type of issue they identify.

### Sequence Completeness Flags

These flags indicate issues with the basic structure of a touch sequence.

#### missing_Began / missing_B

**What it means**: The sequence doesn't start with a `Began` touchPhase (for Coloring data) or `B` (for Tracing data).

**Why it matters**: Every touch sequence should start with a beginning event when the finger first touches the screen. If this is missing, it could indicate that the beginning of the touch was not properly detected or recorded.

**Example**: A touch that began outside the active sensing area and then moved into it.

**How it's detected**: The code checks if the first event in a sequence has the correct beginning touchPhase:

```python
# Check if sequence starts with 'Began'
first_touch_phase = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['touchPhase'].iloc[0]
if first_touch_phase != 'Began':
    seq_metrics.at[idx, 'flags'].append('missing_Began')
```

#### missing_Ended / missing_E

**What it means**: The sequence doesn't have an `Ended` touchPhase (for Coloring data) or `E` (for Tracing data).

**Why it matters**: Every touch sequence should end with an ending event when the finger is lifted from the screen. If this is missing, it could indicate that the end of the touch was not properly detected or recorded.

**Example**: A user lifted their finger outside the active sensing area.

**How it's detected**: The code checks if any event in a sequence has the correct ending touchPhase:

```python
# Check if sequence has an 'Ended' phase
has_ended = 'Ended' in df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['touchPhase'].values
if not has_ended:
    seq_metrics.at[idx, 'flags'].append('missing_Ended')
```

#### multiple_end_events

**What it means**: The sequence has more than one ending event.

**Why it matters**: A proper touch sequence should have exactly one ending event. Multiple ending events suggest a problem with touch detection.

**Example**: The touch sensor incorrectly detected a finger lift and then redetected the same touch.

**How it's detected**: The code counts the number of ending events in a sequence:

```python
# Count 'Ended' events
ended_count = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id) &
                 (df['touchPhase'] == 'Ended')].shape[0]
if ended_count > 1:
    seq_metrics.at[idx, 'flags'].append('multiple_end_events')
```

#### sequence_interrupted

**What it means**: A new sequence started before the current one ended.

**Why it matters**: Each finger should only be involved in one sequence at a time. If a new sequence starts before the current one ends, it indicates a problem with touch detection.

**Example**: A hardware issue causing "phantom" begin events.

**How it's detected**: The code checks if a new `Began` event occurs before an `Ended` event for the same finger:

```python
# Check for sequence interruptions
if prev_seq_id is not None and prev_seq_id != 0:
    prev_seq_ended = 'Ended' in df[(df['fingerId'] == finger_id) &
                                   (df['seqId'] == prev_seq_id)]['touchPhase'].values
    if not prev_seq_ended:
        seq_metrics.at[idx, 'flags'].append('sequence_interrupted')
```

### Sequence Quality Flags

These flags indicate issues with the quality or characteristics of a touch sequence.

#### short_duration

**What it means**: The sequence lasted less than 10 milliseconds.

**Why it matters**: Very short touch sequences might indicate accidental touches or issues with touch detection. Normal intentional touches typically last longer than 10ms.

**Example**: A user accidentally brushed against the screen.

**How it's detected**: The code checks if the duration of the sequence is less than 0.01 seconds (10ms):

```python
# Check for short duration
if row['durationSec'] < 0.01 and not pd.isna(row['durationSec']):
    seq_metrics.at[idx, 'flags'].append('short_duration')
```

#### too_few_points

**What it means**: The sequence has fewer than 3 touch points.

**Why it matters**: A normal touch sequence typically includes multiple points as the finger moves slightly, even for a quick tap. Having very few points might indicate issues with touch detection sensitivity.

**Example**: A quick tap rather than a drag or swipe.

**How it's detected**: The code checks if the number of points in the sequence is less than 3:

```python
# Check for too few points
if row['nPoints'] < 3:
    seq_metrics.at[idx, 'flags'].append('too_few_points')
```

#### sequence_gap / time_gap

**What it means**: There was a significant time gap (>100ms) between events in a sequence.

**Why it matters**: Touch events in a sequence should occur in close succession. Large gaps could indicate performance issues or dropped frames.

**Example**: The application experienced performance issues or frame drops.

**How it's detected**: The code checks for gaps between consecutive events:

```python
# Check for time gaps
time_diffs = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['time'].diff()
if (time_diffs > 0.1).any():
    seq_metrics.at[idx, 'flags'].append('sequence_gap')
```

#### improper_sequence_order

**What it means**: Touch events don't follow the expected order (Began → Moved/Stationary → Ended).

**Why it matters**: Touch events should follow a logical progression. Out-of-order events suggest problems with event processing.

**Example**: Touch events were processed out of order due to threading or timing issues.

**How it's detected**: The code checks if the touchPhase values follow the expected pattern:

```python
# Check for improper sequence order
phases = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['touchPhase'].tolist()
if phases[0] == 'Began' and 'Ended' in phases:
    ended_idx = phases.index('Ended')
    if any(phase == 'Began' for phase in phases[1:ended_idx]):
        seq_metrics.at[idx, 'flags'].append('improper_sequence_order')
```

#### has_canceled

**What it means**: The sequence contains a `Canceled` touchPhase.

**Why it matters**: A `Canceled` event indicates that the touch was interrupted or canceled by the system. This could happen due to various reasons like system interruptions or conflicts with other touch events.

**Example**: The system canceled the touch sequence due to a gesture conflict.

**How it's detected**: The code checks if the sequence has any `Canceled` events:

```python
# Check for canceled touches
if row['hasCanceled']:
    seq_metrics.at[idx, 'flags'].append('has_canceled')
```

#### orphaned_events

**What it means**: Touch events occurred between sequences, not belonging to any sequence.

**Why it matters**: Every touch event should be part of a valid sequence. Orphaned events suggest problems with touch detection.

**Example**: The touch sensor detected movement after the finger was lifted.

**How it's detected**: The code identifies events that don't belong to any valid sequence:

```python
# Check for orphaned events
if df.loc[idx, 'seqId'] == 0 and not is_begin:
    df.loc[idx, 'orphaned_events'] = True
```

#### invalid_TouchPhase

**What it means**: Contains invalid touchPhase values.

**Why it matters**: TouchPhase values should be one of the expected values. Invalid values suggest data corruption or sensor issues.

**Example**: Corrupted data or sensor malfunction.

**How it's detected**: The code checks if touchPhase values are in the list of valid values:

```python
# Check for invalid touchPhase values
valid_phases = ['B', 'M', 'S', 'E', 'C']
if row['touchPhase'] not in valid_phases:
    df.loc[idx, 'invalid_TouchPhase'] = True
```

#### zero_distance

**What it means**: Sum of distance values equals 0 in a sequence with more than 1 point.

**Why it matters**: In a multi-point sequence, there should be some movement (distance > 0). Zero distance suggests sensor issues.

**Example**: Sensor reporting inconsistency.

**How it's detected**: The code calculates the total distance in a sequence:

```python
# Check for zero distance
total_distance = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['distance'].sum()
if total_distance == 0 and row['nPoints'] > 1:
    seq_metrics.at[idx, 'flags'].append('zero_distance')
```

### Tracing-Specific Flags

These flags are specific to Tracing data and indicate issues with tracing activities.

#### PHANTOM_MOVE

**What it means**: TouchPhase is 'M' (Moved) but coordinates don't change while distance increases.

**Why it matters**: In a move event, if the coordinates don't change but the distance increases, it suggests a sensor reporting inconsistency.

**Example**: Sensor reporting inconsistency during a tracing activity.

**How it's detected**: The code checks for move events where coordinates don't change but distance increases:

```python
# Check for phantom moves
if row['touchPhase'] == 'M' and prev_x == row['x'] and prev_y == row['y'] and row['distance'] > 0:
    df.loc[idx, 'PHANTOM_MOVE'] = True
```

#### OVERLAPPING_FINGERIDS

**What it means**: Multiple fingers with overlapping IDs detected.

**Why it matters**: Each finger should have a unique ID. Overlapping IDs suggest a touch sensor ID assignment issue.

**Example**: Touch sensor ID assignment issue when multiple fingers are used.

**How it's detected**: The code checks for fingers with overlapping IDs:

```python
# Check for overlapping finger IDs
active_fingers = set()
for idx, row in df.iterrows():
    if row['touchPhase'] == 'B':
        if row['fingerId'] in active_fingers:
            df.loc[idx, 'OVERLAPPING_FINGERIDS'] = True
        active_fingers.add(row['fingerId'])
    elif row['touchPhase'] == 'E':
        active_fingers.discard(row['fingerId'])
```

#### UNTERMINATED

**What it means**: Touch sequence was not properly terminated.

**Why it matters**: Every touch sequence should be properly terminated with an ending event. Unterminated sequences suggest hardware or software interruptions.

**Example**: Hardware or software interruption during a tracing activity.

**How it's detected**: The code checks for sequences without proper termination:

```python
# Check for unterminated sequences
if 'E' not in df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]['touchPhase'].values:
    df.loc[idx, 'UNTERMINATED'] = True
```

#### ORPHANED_FINGER

**What it means**: Finger events detected without proper sequence context.

**Why it matters**: Every finger event should be part of a valid sequence. Orphaned fingers suggest touch sensor detection anomalies.

**Example**: Touch sensor detection anomaly where a finger ID appears without proper begin/end events.

**How it's detected**: The code identifies finger events that don't belong to any valid sequence:

```python
# Check for orphaned fingers
if df.loc[idx, 'seqId'] == 0 and df.loc[idx, 'fingerId'] not in known_fingers:
    df.loc[idx, 'ORPHANED_FINGER'] = True
```

### Flag Precedence

When conflicts occur between flags, the system resolves them using a priority hierarchy. Higher priority flags take precedence over lower priority flags.

For example, if both `missing_Began` and `orphaned_events` are detected, only `orphaned_events` is shown as it explains why a beginning might be missing.

The priority order (from highest to lowest) is:
1. orphaned_events
2. sequence_interrupted
3. multiple_end_events
4. missing_Began / missing_B
5. missing_Ended / missing_E
6. improper_sequence_order
7. short_duration
8. too_few_points
9. has_canceled
10. sequence_gap / time_gap

