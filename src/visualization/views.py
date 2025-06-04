#!/usr/bin/env python3
"""
Touch Sequence Visualization Tool

This script provides interactive visualizations for touch sequence data from CSV files
in the flagged_data directory. It allows users to explore and analyze touch patterns,
with special focus on flagged sequences that may indicate issues.

Features:
- 2D visualization of touch sequences with color-coded touchPhase values
- Interactive filtering by flag types, sequence IDs, and more
- Comparative visualization between flagged and non-flagged sequences
- Temporal visualization with color gradients and animation options

Usage:
    python views.py

"""

# Standard library imports
import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path

# Define color codes for terminal output (without requiring colorama)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Create aliases similar to colorama's interface
class Fore:
    GREEN = Colors.GREEN
    RED = Colors.RED
    YELLOW = Colors.YELLOW
    BLUE = Colors.BLUE
    CYAN = Colors.CYAN
    MAGENTA = Colors.MAGENTA
    WHITE = Colors.WHITE

class Style:
    BRIGHT = Colors.BOLD
    NORMAL = ''
    RESET_ALL = Colors.END

# ASCII Art for the app header
HEADER = f"""
{Fore.CYAN}
 _    _(_)               _____                      _____  _
| |  | |_  _____      __/ ____|                    |  __ \\| |
| |  | | |/ _ \\ \\ /\\ / / |  __  __ _ _ __ ___   ___| |__) | | __ _ _   _
| |  | | |  __/\\ V  V /| | |_ |/ _` | '_ ` _ \\ / _ \\  ___/| |/ _` | | | |
| |__| | | \\___  \\ /\\ / | |__| | (_| | | | | |  __/ |    | | (_| | |_| |
 \\____/|_|\\____/  V  V   \\_____|\\__,_|_| |_| |_|\\___|_|    |_|\\__,_|\\__, |
                                                                      __/ |
                                                                     |___/
{Style.RESET_ALL}
{Fore.GREEN}Visualization Tool{Style.RESET_ALL}
"""

# Utility functions for colored output
def print_colored(message, color=Fore.WHITE, style=Style.NORMAL):
    """Print colored text."""
    print(f"{style}{color}{message}{Style.RESET_ALL}")

def print_success(message):
    """Print success message."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message."""
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Virtual environment auto-detection and activation
def is_venv_active():
    """Check if a virtual environment is already active."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def find_venv_path():
    """Find the virtual environment path in standard locations."""
    possible_venv_paths = ['./venv', './.venv', '../venv', '../.venv']
    for path in possible_venv_paths:
        activate_script = os.path.join(
            path,
            'Scripts' if sys.platform == 'win32' else 'bin',
            'activate'
        )
        if os.path.exists(activate_script):
            return path, activate_script
    return None, None

# Check and install dependencies
def check_and_install_dependencies():
    """Check if required packages are installed and install them if needed."""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']

    # Check if pip is available
    try:
        import pip
    except ImportError:
        print_error("pip is not installed. Please install pip first.")
        sys.exit(1)

    # Check each package
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is already installed")
        except ImportError:
            missing_packages.append(package)

    # Install missing packages
    if missing_packages:
        print_info(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print_info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print_success(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print_error(f"Failed to install {package}. Please install it manually.")
                sys.exit(1)
        print_success("All dependencies installed successfully!")

    # Now import the packages
    global pd, np, plt, mpatches, Line2D, animation, CheckButtons, Slider, Button, RadioButtons, sns, LinearSegmentedColormap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    from matplotlib.widgets import CheckButtons, Slider, Button, RadioButtons
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # Set style and backend (Mac compatibility)
    import platform
    import matplotlib

    # Print matplotlib version for debugging
    print(f"Matplotlib version: {matplotlib.__version__}")

    if platform.system() == 'Darwin':  # macOS
        try:
            # Try TkAgg backend first (works on most Macs)
            matplotlib.use('TkAgg')
        except ImportError:
            try:
                # If TkAgg fails, try Qt5Agg
                matplotlib.use('Qt5Agg')
            except ImportError:
                try:
                    # If Qt5Agg fails, try macosx backend
                    matplotlib.use('macosx')
                except ImportError:
                    # Fall back to default backend if all else fails
                    print("Warning: Could not set optimal backend for Mac. Using default backend.")

        print(f"Using matplotlib backend: {matplotlib.get_backend()}")

    # Use compatible style based on matplotlib version
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            # For older matplotlib versions
            plt.style.use('seaborn-whitegrid')
        except:
            # Fallback to default style
            print("Note: Using default style as seaborn styles not found")

    sns.set_context("notebook")

# Check and install dependencies
check_and_install_dependencies()

# Constants
FLAGGED_DATA_DIR = "data/processed/flagged"
TOUCH_PHASE_COLORS = {
    'Began': 'green',        # Green color
    'Moved': '#444444',      # Keep dark grey for less visual dominance
    'Stationary': 'purple',  # Purple color
    'Ended': 'orange',       # Orange color (corrected)
    'Canceled': 'red'        # Red color (corrected)
}
FLAG_TYPES = [
    'missing_Began',
    'missing_Ended',
    'short_duration',
    'too_few_points',
    'has_canceled'
]

# Styling constants for emphasizing flagged sequences
NORMAL_ALPHA = 0.2  # Increased transparency for normal sequences (reduced from 0.3)
FLAGGED_ALPHA = 0.8  # Opacity for flagged sequences (unchanged)
NORMAL_LINEWIDTH = 1.0  # Line width for normal sequences (unchanged)
FLAGGED_LINEWIDTH = 3.0  # Line width for flagged sequences (unchanged)
NORMAL_MARKER_SIZE = 30  # Marker size for normal sequences (restored to original)
FLAGGED_MARKER_SIZE = 50  # Marker size for flagged sequences (restored to original)

# Marker sizes for specific touch phases in flagged sequences
FLAGGED_PHASE_MARKER_SIZES = {
    'Began': 40,     # Increased size for better visibility
    'Moved': 15,     # Keep reduced size for better path visibility
    'Stationary': 40, # Increased size for better visibility
    'Ended': 40,     # Increased size for better visibility
    'Canceled': 40   # Increased size for better visibility
}

# Monochromatic color for normal sequences
NORMAL_COLOR = 'skyblue'  # Brighter but still subdued blue color for normal sequences
NORMAL_EDGE_COLOR = 'deepskyblue'  # Slightly darker edge color for better visibility with new color

# Markers for different touch phases
TOUCH_PHASE_MARKERS = {
    'Began': 'o',       # Circle for begin events
    'Moved': '.',       # Point for move events (less visual dominance)
    'Stationary': 's',  # Square for stationary events
    'Ended': 'D',       # Diamond for end events
    'Canceled': 'X',    # X for canceled events
    'B': 'o',           # Circle for B events (Tracing format)
    'M': '.',           # Point for M events (Tracing format)
    'S': 's',           # Square for S events (Tracing format)
    'E': 'D'            # Diamond for E events (Tracing format)
}


def load_data(csv_path):
    """
    Load and preprocess touch sequence data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        tuple: (df, seq_metrics) - The main dataframe and sequence metrics
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # The CSV has a structure with two tables side by side
    # First part is the touch data, second part is sequence metrics
    # Find the separator column (usually filled with spaces)
    separator_col = None
    for col in df.columns:
        if col.strip() == '':
            separator_col = col
            break

    if separator_col:
        # Split the dataframe into two parts
        touch_cols = list(df.columns[:df.columns.get_loc(separator_col)])
        seq_cols = list(df.columns[df.columns.get_loc(separator_col) + 1:])

        # Extract sequence metrics
        seq_metrics = df[seq_cols].copy()
        seq_metrics.columns = [col.strip() for col in seq_metrics.columns]

        # Clean up the main dataframe
        df = df[touch_cols].copy()
    else:
        # If no separator column is found, assume it's just the touch data
        seq_metrics = None

    # Convert flags to list
    if 'flags' in df.columns:
        df['flags'] = df['flags'].fillna('').apply(lambda x: x.split(',') if x else [])

    if seq_metrics is not None and 'flags' in seq_metrics.columns:
        seq_metrics['flags'] = seq_metrics['flags'].fillna('').apply(lambda x: x.split(',') if x else [])

    return df, seq_metrics


def _get_sequence_iterator(df):
    """
    Get an iterator for sequences based on available columns.

    Args:
        df (DataFrame): The touch sequence data

    Returns:
        tuple: (iterator_source, use_touchdata_id, is_composite_key)
            - iterator_source: numpy array of sequence identifiers
            - use_touchdata_id: bool indicating if using Touchdata_id format
            - is_composite_key: bool indicating if using (fingerId, seqId) format
    """
    # Check for new format first (Touchdata_id)
    if 'Touchdata_id' in df.columns:
        # Use Touchdata_id for sequence iteration
        unique_ids = df['Touchdata_id'].dropna().unique()
        return unique_ids, True, False

    # Check for old format (fingerId + seqId)
    elif 'fingerId' in df.columns and 'seqId' in df.columns:
        # Use (fingerId, seqId) pairs for sequence iteration
        seq_pairs = df[['fingerId', 'seqId']].drop_duplicates().values
        return seq_pairs, False, True

    else:
        # No valid sequence identifier columns found
        print_warning("No valid sequence identifier columns found (Touchdata_id or fingerId/seqId)")
        return np.array([]), False, False


def create_basic_visualization(df, seq_metrics=None, output_path=None, csv_path=None):
    """
    Create a basic 2D visualization of touch sequences.

    Args:
        df (DataFrame): The touch sequence data
        seq_metrics (DataFrame, optional): Sequence metrics data
        output_path (str, optional): Path to save the visualization
        csv_path (str, optional): Path to the CSV file for title information

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    iterator_source, use_touchdata_id, is_composite_key = _get_sequence_iterator(df)

    if iterator_source.size == 0 and not (use_touchdata_id or is_composite_key):
        # This case implies _get_sequence_iterator found no valid ID columns.
        print_warning("Suitable ID columns ('Touchdata_id' or 'fingerId'/'seqId') are missing or invalid for basic plot. Visualization may be empty or fail.")

    # Track which sequences have flags for the legend
    has_flagged_seq = False
    has_normal_seq = False

    # Plot each sequence
    for id_val in iterator_source:
        seq_data = pd.DataFrame() 
        if use_touchdata_id:
            if pd.isna(id_val): continue # Should be handled by dropna in helper, but as safeguard
            seq_data = df[df['Touchdata_id'] == id_val]
        elif is_composite_key:
            finger_id, seq_id = id_val
            if seq_id == 0: # Original check for old format
                continue
            seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]
        else: 
            # No valid iterator type, or iterator is empty and this loop shouldn't run.
            break 
            
        if seq_data.empty:
            continue
        
        # Check if sequence has flags
        has_flags = sequence_has_flags(seq_data)

        # Set styling based on flags
        if has_flags:
            line_style = '--'
            line_width = FLAGGED_LINEWIDTH
            alpha = FLAGGED_ALPHA
            marker_size = FLAGGED_MARKER_SIZE
            has_flagged_seq = True
            zorder_line = 3  # Higher zorder to draw on top
            zorder_marker = 4
        else:
            line_style = '-'
            line_width = NORMAL_LINEWIDTH
            alpha = NORMAL_ALPHA
            marker_size = NORMAL_MARKER_SIZE
            has_normal_seq = True
            zorder_line = 1
            zorder_marker = 2

        # Plot the sequence path with appropriate color
        path_color = None if has_flags else NORMAL_COLOR  # Use monochromatic color for normal sequences
        ax.plot(seq_data['x'], seq_data['y'],
                linestyle=line_style,
                linewidth=line_width,
                alpha=alpha,
                color=path_color,  # Will be None for flagged sequences (uses default color cycle)
                zorder=zorder_line)

        # Plot points for each touchPhase with different colors
        for phase in TOUCH_PHASE_COLORS:
            phase_data = seq_data[seq_data['touchPhase'] == phase]
            if not phase_data.empty:
                # Choose color based on whether it's a flagged or normal sequence
                if has_flags:
                    # Use multi-color scheme for flagged sequences
                    point_color = TOUCH_PHASE_COLORS[phase]
                    # Remove border for special touch events, keep border only for Moved points
                    if phase == 'Moved':
                        edge_color = 'black'
                    else:
                        edge_color = point_color  # Same as fill color to remove visible border
                    # Use phase-specific marker size for flagged sequences
                    phase_marker_size = FLAGGED_PHASE_MARKER_SIZES.get(phase, FLAGGED_MARKER_SIZE)
                else:
                    # Use monochromatic scheme for normal sequences
                    point_color = NORMAL_COLOR
                    edge_color = NORMAL_EDGE_COLOR
                    phase_marker_size = marker_size

                ax.scatter(phase_data['x'], phase_data['y'],
                          color=point_color,
                          edgecolors=edge_color,
                          s=phase_marker_size,
                          alpha=alpha,
                          label=f"{phase}" if has_flags and phase not in ax.get_legend_handles_labels()[1] else "",
                          zorder=zorder_marker)

    # Create legend elements
    legend_elements = []

    # Add touchPhase colors to legend only for flagged sequences
    if has_flagged_seq:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=color, markersize=8,
                  label=f"{phase} (Flagged Sequences)")
            for phase, color in TOUCH_PHASE_COLORS.items()
        ])

    # Add line style to legend
    if has_normal_seq:
        # Add monochromatic normal sequence to legend
        legend_elements.append(Line2D([0], [0], linestyle='-', color=NORMAL_COLOR,
                                     alpha=NORMAL_ALPHA, linewidth=NORMAL_LINEWIDTH,
                                     label='Normal Sequence (Monochromatic)'))
        # Add a marker for normal sequence points
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=NORMAL_COLOR, markeredgecolor=NORMAL_EDGE_COLOR,
                                     markersize=8, alpha=NORMAL_ALPHA,
                                     label='Normal Sequence Points'))

    if has_flagged_seq:
        # Add multi-colored flagged sequence to legend
        legend_elements.append(Line2D([0], [0], linestyle='--', color='red',
                                     alpha=FLAGGED_ALPHA, linewidth=FLAGGED_LINEWIDTH,
                                     label='Flagged Sequence (Multi-colored)'))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Include the filename in the title if available
    if csv_path:
        filename = get_base_filename(csv_path)
        ax.set_title(f'View GamePlay: [{filename}] (Flagged Sequences Highlighted)')
    else:
        ax.set_title('View GamePlay (Flagged Sequences Highlighted)')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_interactive_visualization(df, seq_metrics=None, csv_path=None):
    """
    Create an interactive visualization with filtering controls.
    Now creates an HTML-based interactive visualization instead of matplotlib widgets.

    Args:
        df (DataFrame): The touch sequence data
        seq_metrics (DataFrame, optional): Sequence metrics data
        csv_path (str, optional): Path to the CSV file for title information

    Returns:
        str: Path to the generated HTML file
    """
    try:
        # Import required modules
        import os
        import sys

        # Import the HTML visualization function
        try:
            from .html_interactive import create_html_interactive_visualization
        except ImportError:
            # Fallback for when running as main script
            sys.path.append(os.path.dirname(__file__))
            from html_interactive import create_html_interactive_visualization

        if not csv_path:
            print_warning("No CSV path provided, using default filename")
            base_name = "touch_data"
        else:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]

        output_path = f"interactive_viz_{base_name}.html"

        print_info("Creating HTML-based interactive visualization...")
        html_path = create_html_interactive_visualization(csv_path, output_path)

        if html_path:
            print_success(f"Interactive HTML visualization created: {html_path}")

            # Try to open in browser
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
                print_info("Opening visualization in your default web browser...")
            except Exception as e:
                print_warning(f"Could not auto-open browser: {e}")
                print_info(f"Please manually open: {os.path.abspath(html_path)}")

            return html_path
        else:
            print_error("Failed to create HTML visualization, falling back to matplotlib")
            return _create_interactive_visualization_fallback(df, seq_metrics, csv_path)

    except ImportError:
        print_warning("HTML visualization module not available, falling back to matplotlib widgets")
        try:
            return _create_interactive_visualization_with_widgets(df, seq_metrics, csv_path)
        except Exception as e:
            print_warning(f"Interactive widgets failed ({e}), falling back to basic visualization with filtering info")
            return _create_interactive_visualization_fallback(df, seq_metrics, csv_path)
    except Exception as e:
        print_error(f"HTML visualization failed ({e}), falling back to matplotlib")
        return _create_interactive_visualization_fallback(df, seq_metrics, csv_path)


def _create_interactive_visualization_with_widgets(df, seq_metrics=None, csv_path=None):
    """
    Create an interactive visualization with filtering controls (with widgets).
    """
    # Create figure with tight layout manager for better Mac compatibility
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # Use a consistent grid size of 10x5
    # Main plot area
    ax_main = plt.subplot2grid((10, 5), (0, 0), colspan=4, rowspan=8)

    # Controls area - adjusted for better Mac compatibility
    ax_flags = plt.subplot2grid((10, 5), (0, 4), rowspan=4)
    ax_filter = plt.subplot2grid((10, 5), (4, 4), rowspan=2)
    ax_toggle = plt.subplot2grid((10, 5), (6, 4), rowspan=1)
    ax_reset = plt.subplot2grid((10, 5), (7, 4), rowspan=1)

    # Stats area
    ax_stats = plt.subplot2grid((10, 5), (8, 0), colspan=5, rowspan=2)

    # Ensure figure is drawn once to initialize properly on Mac
    fig.canvas.draw()

    # The old seq_pairs initialization here is removed as it's not used before update_plot.
    # seq_pairs = df[['fingerId', 'seqId']].drop_duplicates().values # This line is removed.

    # Get all unique flag types in the data
    all_flags = set()
    for flags in df['flags']:
        flag_list = parse_flags_from_string(flags)
        all_flags.update(flag_list)
    all_flags = sorted(list(all_flags))

    # If no flags found, use the predefined FLAG_TYPES
    if not all_flags:
        all_flags = FLAG_TYPES

    # Initialize with all flags active
    active_flags = {flag: True for flag in all_flags}

    # Initialize with flagged sequences visible (changed from 'all' to 'flagged')
    show_mode = 'flagged'  # Options: 'all', 'flagged', 'non_flagged'

    # Create scatter plot collections for each touchPhase (not used but kept for API consistency)
    scatter_collections = {}
    line_collections = {}

    def update_plot():
        # Clear the main axis
        ax_main.clear()

        # Import collections for faster plotting
        from matplotlib.collections import LineCollection, PathCollection
        import numpy as np

        # Track which sequences have flags for the legend
        has_flagged_seq = False
        has_normal_seq = False

        # Count statistics
        total_seqs = 0
        flagged_seqs = 0
        flag_type_counts = {flag: 0 for flag in all_flags}

        # Prepare collections for batch plotting
        flagged_lines = []
        normal_lines = []

        # Prepare collections for points by phase
        flagged_points = {phase: [] for phase in TOUCH_PHASE_COLORS}
        normal_points = {phase: [] for phase in TOUCH_PHASE_COLORS}

        # Track which phases are used for legend
        used_phases = set()

        # Process each sequence to prepare data for batch plotting
        iterator_source_update, use_touchdata_id_update, is_composite_key_update = _get_sequence_iterator(df)

        if iterator_source_update.size == 0 and not (use_touchdata_id_update or is_composite_key_update) :
            print_warning("Suitable ID columns for interactive plot update are missing. Visualization may be empty.")

        for id_val in iterator_source_update: 
            seq_data = pd.DataFrame() 
            if use_touchdata_id_update:
                if pd.isna(id_val): continue
                seq_data = df[df['Touchdata_id'] == id_val]
            elif is_composite_key_update:
                finger_id, seq_id = id_val 
                if seq_id == 0: 
                    continue
                seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]
            else:
                break

            if seq_data.empty: 
                continue

            # Check if sequence has any of the active flags
            seq_flags = get_sequence_flags(seq_data)

            # Count flag types for statistics
            for flag in seq_flags:
                if flag in flag_type_counts:
                    flag_type_counts[flag] += 1

            has_active_flags = any(flag in seq_flags and active_flags[flag] for flag in all_flags if flag in active_flags)

            # Determine if we should show this sequence based on the show_mode
            show_seq = False
            if show_mode == 'all':
                show_seq = True
            elif show_mode == 'flagged' and has_active_flags:
                show_seq = True
            elif show_mode == 'non_flagged' and not has_active_flags:
                show_seq = True

            if show_seq:
                total_seqs += 1

                # Sort by time to ensure proper line drawing
                seq_data = seq_data.sort_values(by='time')

                # Set styling based on flags
                if has_active_flags:
                    has_flagged_seq = True
                    flagged_seqs += 1
                else:
                    has_normal_seq = True

                # Prepare line segments for this sequence
                if len(seq_data) > 1:
                    # Get x and y coordinates as numpy arrays
                    x = seq_data['x'].values
                    y = seq_data['y'].values

                    # Create line segments with proper error handling
                    points = np.column_stack([x, y])
                    if len(points) >= 2:
                        # Create segments properly - each segment should be a 2x2 array
                        segments = []
                        for i in range(len(points)-1):
                            segment = np.array([points[i], points[i+1]])
                            if segment.shape == (2, 2):  # Ensure proper shape
                                segments.append(segment)

                        if segments:  # Only add if we have valid segments
                            segments = np.array(segments)
                            # Add to appropriate collection
                            if has_active_flags:
                                flagged_lines.append(segments)
                            else:
                                normal_lines.append(segments)

                # Prepare points for each touchPhase
                for phase in TOUCH_PHASE_COLORS:
                    phase_data = seq_data[seq_data['touchPhase'] == phase]
                    if not phase_data.empty:
                        # Add to appropriate collection
                        if has_active_flags:
                            flagged_points[phase].append(phase_data[['x', 'y']].values)
                            used_phases.add(phase)
                        else:
                            normal_points[phase].append(phase_data[['x', 'y']].values)

        # Create and add line collections
        if normal_lines:
            # Flatten the list of arrays into a single array of segments
            all_normal_segments = np.vstack([segment for segments in normal_lines for segment in segments])

            normal_line_collection = LineCollection(
                all_normal_segments,
                color=NORMAL_COLOR,
                alpha=NORMAL_ALPHA,
                linewidth=NORMAL_LINEWIDTH,
                linestyle='-',
                zorder=1
            )
            ax_main.add_collection(normal_line_collection)

        if flagged_lines:
            # Flatten the list of arrays into a single array of segments
            all_flagged_segments = np.vstack([segment for segments in flagged_lines for segment in segments])

            flagged_line_collection = LineCollection(
                all_flagged_segments,
                color='red',
                alpha=FLAGGED_ALPHA,
                linewidth=FLAGGED_LINEWIDTH,
                linestyle='--',
                zorder=3
            )
            ax_main.add_collection(flagged_line_collection)

        # Create and add point collections for each phase
        for phase in TOUCH_PHASE_COLORS:
            # Process normal points
            if normal_points[phase] and len(normal_points[phase]) > 0:
                # Combine all points for this phase
                combined_points = np.vstack(normal_points[phase])

                # Create scatter collection
                ax_main.scatter(
                    combined_points[:, 0], combined_points[:, 1],
                    color=NORMAL_COLOR,
                    edgecolors=NORMAL_EDGE_COLOR,
                    s=NORMAL_MARKER_SIZE,
                    alpha=NORMAL_ALPHA,
                    marker=TOUCH_PHASE_MARKERS.get(phase, 'o'),
                    zorder=2
                )

            # Process flagged points
            if flagged_points[phase] and len(flagged_points[phase]) > 0:
                # Combine all points for this phase
                combined_points = np.vstack(flagged_points[phase])

                # Get color and marker properties
                point_color = TOUCH_PHASE_COLORS[phase]
                edge_color = 'black' if phase == 'Moved' else point_color
                phase_marker_size = FLAGGED_PHASE_MARKER_SIZES.get(phase, FLAGGED_MARKER_SIZE)

                # Create scatter collection
                ax_main.scatter(
                    combined_points[:, 0], combined_points[:, 1],
                    color=point_color,
                    edgecolors=edge_color,
                    s=phase_marker_size,
                    alpha=FLAGGED_ALPHA,
                    marker=TOUCH_PHASE_MARKERS.get(phase, 'o'),
                    zorder=4,
                    label=phase
                )

        # Create legend elements
        legend_elements = []

        # Add touchPhase colors to legend only for flagged sequences
        if has_flagged_seq:
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=8,
                      label=f"{phase} (Flagged Sequences)")
                for phase, color in TOUCH_PHASE_COLORS.items() if phase in used_phases
            ])

        # Add line style to legend
        if has_normal_seq:
            # Add monochromatic normal sequence to legend
            legend_elements.append(Line2D([0], [0], linestyle='-', color=NORMAL_COLOR,
                                         alpha=NORMAL_ALPHA, linewidth=NORMAL_LINEWIDTH,
                                         label='Normal Sequence (Monochromatic)'))
            # Add a marker for normal sequence points
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=NORMAL_COLOR, markeredgecolor=NORMAL_EDGE_COLOR,
                                         markersize=8, alpha=NORMAL_ALPHA,
                                         label='Normal Sequence Points'))

        if has_flagged_seq:
            # Add multi-colored flagged sequence to legend
            legend_elements.append(Line2D([0], [0], linestyle='--', color='red',
                                         alpha=FLAGGED_ALPHA, linewidth=FLAGGED_LINEWIDTH,
                                         label='Flagged Sequence (Multi-colored)'))

        # Add legend
        ax_main.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

        # Set axis limits based on data
        ax_main.autoscale()

        # Set labels and title
        ax_main.set_xlabel('X Coordinate')
        ax_main.set_ylabel('Y Coordinate')

        # Set title based on current mode and include filename if available
        if csv_path:
            filename = get_base_filename(csv_path)
            if show_mode == 'all':
                title = f'View GamePlay: [{filename}] (All Sequences)'
            elif show_mode == 'flagged':
                title = f'View GamePlay: [{filename}] (Flagged Sequences Only)'
            else:
                title = f'View GamePlay: [{filename}] (Non-Flagged Sequences Only)'
        else:
            if show_mode == 'all':
                title = 'View GamePlay (All Sequences)'
            elif show_mode == 'flagged':
                title = 'View GamePlay (Flagged Sequences Only)'
            else:
                title = 'View GamePlay (Non-Flagged Sequences Only)'
        ax_main.set_title(title)

        # Add grid
        ax_main.grid(True, linestyle='--', alpha=0.7)

        # Update stats
        ax_stats.clear()
        ax_stats.axis('off')

        # Calculate total sequences from the iterator
        total_sequences_available = len(iterator_source_update)

        # Calculate percentages safely
        flagged_percent = (flagged_seqs/total_sequences_available*100) if total_sequences_available > 0 else 0
        non_flagged_percent = ((total_sequences_available - flagged_seqs)/total_sequences_available*100) if total_sequences_available > 0 else 0

        # Create detailed statistics text
        stats_text = (f"Total Sequences: {total_sequences_available}\n"
                     f"Flagged Sequences: {flagged_seqs} ({flagged_percent:.1f}%)\n"
                     f"Non-Flagged Sequences: {total_sequences_available - flagged_seqs} ({non_flagged_percent:.1f}%)\n"
                     f"Currently Showing: {total_seqs} sequences\n\n"
                     f"Flag Counts: " + ", ".join([f"{flag}: {count}" for flag, count in flag_type_counts.items() if count > 0]))

        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)

        # Redraw the figure
        fig.canvas.draw_idle()

    # Create flag checkboxes with improved Mac compatibility
    try:
        check = CheckButtons(ax_flags, all_flags, [True] * len(all_flags))
    except Exception as e:
        print_error(f"Error creating checkboxes: {e}")
        # Fallback: create a simple text display instead
        ax_flags.text(0.5, 0.5, f"Flags: {', '.join(all_flags)}", ha='center', va='center')
        ax_flags.axis('off')
        check = None

    # Improve visibility of checkboxes - handle different matplotlib versions
    try:
        # For newer matplotlib versions
        if hasattr(check, 'rectangles'):
            for rect in check.rectangles:
                rect.set_linewidth(1.5)
        # For older matplotlib versions
        elif hasattr(check, 'rects'):
            for rect in check.rects:
                rect.set_linewidth(1.5)
        # For matplotlib 3.5+ versions
        elif hasattr(check, '_rectangles'):
            for rect in check._rectangles:
                rect.set_linewidth(1.5)
    except Exception as e:
        print(f"Note: Could not customize checkbox appearance: {e}")
        # Continue without customization

    def flag_clicked(label):
        active_flags[label] = not active_flags[label]
        # Force redraw for Mac
        plt.draw()
        update_plot()

    if check is not None:
        check.on_clicked(flag_clicked)

    # Create radio buttons for filter mode with improved Mac compatibility
    try:
        radio = RadioButtons(ax_filter, ('Flagged Only', 'All Sequences', 'Non-Flagged Only'))
    except Exception as e:
        print_error(f"Error creating radio buttons: {e}")
        # Fallback: create a simple text display instead
        ax_filter.text(0.5, 0.5, "Filter: Flagged Only", ha='center', va='center')
        ax_filter.axis('off')
        radio = None

    # Improve visibility of radio buttons - handle different matplotlib versions
    try:
        # For newer matplotlib versions
        if hasattr(radio, 'circles'):
            for circle in radio.circles:
                circle.set_linewidth(1.5)
        # For older matplotlib versions
        elif hasattr(radio, '_circles'):
            for circle in radio._circles:
                circle.set_linewidth(1.5)
    except Exception as e:
        print(f"Note: Could not customize radio button appearance: {e}")
        # Continue without customization

    def mode_clicked(label):
        nonlocal show_mode
        if label == 'All Sequences':
            show_mode = 'all'
        elif label == 'Flagged Only':
            show_mode = 'flagged'
        elif label == 'Non-Flagged Only':
            show_mode = 'non_flagged'
        # Force redraw for Mac
        plt.draw()
        update_plot()

    if radio is not None:
        radio.on_clicked(mode_clicked)

    # Create toggle button for quick switching between all and flagged
    toggle_button = Button(ax_toggle, 'Toggle All/Flagged', color='lightblue')
    # Improve button appearance
    toggle_button.label.set_fontweight('bold')

    def toggle(_):
        nonlocal show_mode
        if show_mode == 'flagged':
            show_mode = 'all'
            if radio is not None:
                try:
                    radio.set_active(1)  # Set radio to 'All Sequences'
                except Exception as e:
                    print(f"Note: Could not set radio button: {e}")
        else:
            show_mode = 'flagged'
            if radio is not None:
                try:
                    radio.set_active(0)  # Set radio to 'Flagged Only'
                except Exception as e:
                    print(f"Note: Could not set radio button: {e}")
        # Force redraw for Mac
        plt.draw()
        update_plot()

    toggle_button.on_clicked(toggle)

    # Create reset button with improved Mac compatibility
    reset_button = Button(ax_reset, 'Reset View')
    # Improve button appearance
    reset_button.label.set_fontweight('bold')

    def reset(_):  # Unused event parameter
        # Reset all flags to active
        for flag in active_flags:
            active_flags[flag] = True

        # Reset show mode to flagged (changed from 'all')
        nonlocal show_mode
        show_mode = 'flagged'

        # Update checkboxes - handle different matplotlib versions
        if check is not None:
            try:
                # Try different attribute names based on matplotlib version
                if hasattr(check, 'rectangles'):
                    for i, _ in enumerate(check.labels):
                        check.rectangles[i].set_visible(True)
                elif hasattr(check, 'rects'):
                    for i, _ in enumerate(check.labels):
                        check.rects[i].set_visible(True)
                elif hasattr(check, '_rectangles'):
                    for i, _ in enumerate(check.labels):
                        check._rectangles[i].set_visible(True)
            except Exception as e:
                print(f"Note: Could not update checkbox visibility: {e}")
                # Continue without updating checkboxes

        # Set radio button to 'Flagged Only'
        if radio is not None:
            try:
                radio.set_active(0)
            except Exception as e:
                print(f"Note: Could not set radio button: {e}")

        # Force redraw for Mac
        plt.draw()
        # Update plot
        update_plot()

    reset_button.on_clicked(reset)

    # Initial plot
    update_plot()

    # Adjust layout
    plt.tight_layout()

    return fig


def _create_interactive_visualization_fallback(df, seq_metrics=None, csv_path=None):
    """
    Create a fallback visualization when interactive widgets fail.
    This creates a basic visualization with filtering information displayed as text.
    """
    # Create figure
    fig, (ax_main, ax_info) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[4, 1])

    # Get sequence iterator
    iterator_source, use_touchdata_id, is_composite_key = _get_sequence_iterator(df)

    # Count sequences by flag status
    flagged_count = 0
    normal_count = 0
    flag_type_counts = {flag: 0 for flag in FLAG_TYPES}

    # Plot all sequences
    for id_val in iterator_source:
        seq_data = pd.DataFrame()
        if use_touchdata_id:
            if pd.isna(id_val):
                continue
            seq_data = df[df['Touchdata_id'] == id_val]
        elif is_composite_key:
            finger_id, seq_id = id_val
            if seq_id == 0:
                continue
            seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]
        else:
            break

        if seq_data.empty:
            continue

        # Get all flags for this sequence
        seq_flags = get_sequence_flags(seq_data)

        # Count flag types
        for flag in seq_flags:
            if flag in flag_type_counts:
                flag_type_counts[flag] += 1

        # Check if sequence has flags
        has_flags = len(seq_flags) > 0

        if has_flags:
            flagged_count += 1
            # Plot flagged sequences with enhanced visibility
            ax_main.plot(seq_data['x'], seq_data['y'],
                        linestyle='--',
                        linewidth=FLAGGED_LINEWIDTH,
                        alpha=FLAGGED_ALPHA,
                        zorder=3)

            # Plot points for each touchPhase with different colors
            for phase in TOUCH_PHASE_COLORS:
                phase_data = seq_data[seq_data['touchPhase'] == phase]
                if not phase_data.empty:
                    phase_marker_size = FLAGGED_PHASE_MARKER_SIZES.get(phase, FLAGGED_MARKER_SIZE)
                    point_color = TOUCH_PHASE_COLORS[phase]

                    if phase == 'Moved':
                        edge_color = 'black'
                    else:
                        edge_color = point_color

                    ax_main.scatter(phase_data['x'], phase_data['y'],
                                  color=point_color,
                                  edgecolors=edge_color,
                                  s=phase_marker_size,
                                  alpha=FLAGGED_ALPHA,
                                  label=f"{phase}" if phase not in ax_main.get_legend_handles_labels()[1] else "",
                                  zorder=4)
        else:
            normal_count += 1
            # Plot normal sequences with monochromatic color
            ax_main.plot(seq_data['x'], seq_data['y'],
                        linestyle='-',
                        linewidth=NORMAL_LINEWIDTH,
                        alpha=NORMAL_ALPHA,
                        color=NORMAL_COLOR,
                        zorder=1)

            # Plot points with monochromatic colors
            for phase in TOUCH_PHASE_COLORS:
                phase_data = seq_data[seq_data['touchPhase'] == phase]
                if not phase_data.empty:
                    ax_main.scatter(phase_data['x'], phase_data['y'],
                                  color=NORMAL_COLOR,
                                  edgecolors=NORMAL_EDGE_COLOR,
                                  s=NORMAL_MARKER_SIZE,
                                  alpha=NORMAL_ALPHA,
                                  zorder=2)

    # Set up main plot
    ax_main.set_xlabel('X Coordinate')
    ax_main.set_ylabel('Y Coordinate')
    ax_main.grid(True, linestyle='--', alpha=0.7)
    ax_main.legend(loc='upper right', framealpha=0.7)

    # Add title
    if csv_path:
        filename = get_base_filename(csv_path)
        ax_main.set_title(f'View GamePlay: [{filename}] - All Touch Sequences (Interactive Mode Unavailable)', fontsize=14)
    else:
        ax_main.set_title('View GamePlay: All Touch Sequences (Interactive Mode Unavailable)', fontsize=14)

    # Display filtering information in the bottom panel
    ax_info.axis('off')

    total_sequences = flagged_count + normal_count
    flagged_percent = (flagged_count/total_sequences*100) if total_sequences > 0 else 0

    info_text = (
        f"FILTERING INFORMATION (Interactive controls unavailable):\n\n"
        f"Total Sequences: {total_sequences}   |   "
        f"Flagged: {flagged_count} ({flagged_percent:.1f}%)   |   "
        f"Non-Flagged: {normal_count} ({100-flagged_percent:.1f}%)\n\n"
        f"Flag Types: " + ", ".join([f"{flag}: {count}" for flag, count in flag_type_counts.items() if count > 0])
    )

    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    return fig


def create_comparative_visualization(df, seq_metrics=None, csv_path=None):  # seq_metrics is kept for API consistency
    """
    Create a comparative visualization with flagged and non-flagged sequences side by side.

    Args:
        df (DataFrame): The touch sequence data
        seq_metrics (DataFrame, optional): Sequence metrics data (not used in this visualization)
        csv_path (str, optional): Path to the CSV file for title information

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure with GridSpec for more control over subplot sizes
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])

    # Create axes with 2:1 width ratio (flagged:non-flagged)
    ax_flagged = fig.add_subplot(gs[0, 0:2])  # Flagged sequences get 2/3 of the width
    ax_non_flagged = fig.add_subplot(gs[0, 2], sharey=ax_flagged)  # Non-flagged get 1/3

    # Stats area at the bottom
    ax_stats = fig.add_subplot(gs[1, :])

    # Get sequence iterator
    iterator_source, use_touchdata_id, is_composite_key = _get_sequence_iterator(df)

    # Prepare data for statistics
    flagged_seqs = []
    non_flagged_seqs = []

    # Track flag types
    flag_type_counts = {flag: 0 for flag in FLAG_TYPES}

    # Process each sequence
    for id_val in iterator_source:
        seq_data = pd.DataFrame()
        if use_touchdata_id:
            if pd.isna(id_val):
                continue
            seq_data = df[df['Touchdata_id'] == id_val]
        elif is_composite_key:
            finger_id, seq_id = id_val
            # Skip sequences with seqId = 0
            if seq_id == 0:
                continue
            seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)]
        else:
            break

        if seq_data.empty:
            continue

        # Get all flags for this sequence
        seq_flags = get_sequence_flags(seq_data)

        # Count flag types
        for flag in seq_flags:
            if flag in flag_type_counts:
                flag_type_counts[flag] += 1

        # Check if sequence has flags
        has_flags = len(seq_flags) > 0

        # Store sequence info for statistics
        if use_touchdata_id:
            seq_info = {
                'Touchdata_id': id_val,
                'points': len(seq_data),
                'duration': seq_data['time'].max() - seq_data['time'].min() if len(seq_data) > 1 else 0,
                'flags': list(seq_flags)
            }
        else:
            seq_info = {
                'fingerId': finger_id,
                'seqId': seq_id,
                'points': len(seq_data),
                'duration': seq_data['time'].max() - seq_data['time'].min() if len(seq_data) > 1 else 0,
                'flags': list(seq_flags)
            }

        if has_flags:
            flagged_seqs.append(seq_info)

            # Plot on the flagged axis with enhanced visibility
            ax_flagged.plot(seq_data['x'], seq_data['y'],
                    linestyle='--',
                    linewidth=FLAGGED_LINEWIDTH,
                    alpha=FLAGGED_ALPHA,
                    zorder=3)

            # Plot points for each touchPhase with different colors
            for phase in TOUCH_PHASE_COLORS:
                phase_data = seq_data[seq_data['touchPhase'] == phase]
                if not phase_data.empty:
                    # Use phase-specific marker size for flagged sequences
                    phase_marker_size = FLAGGED_PHASE_MARKER_SIZES.get(phase, FLAGGED_MARKER_SIZE)
                    # Get color for this phase
                    point_color = TOUCH_PHASE_COLORS[phase]

                    # Remove border for special touch events, keep border only for Moved points
                    if phase == 'Moved':
                        edge_color = 'black'
                    else:
                        edge_color = point_color  # Same as fill color to remove visible border

                    ax_flagged.scatter(phase_data['x'], phase_data['y'],
                              color=point_color,
                              edgecolors=edge_color,
                              s=phase_marker_size,
                              alpha=FLAGGED_ALPHA,
                              label=f"{phase}" if phase not in ax_flagged.get_legend_handles_labels()[1] else "",
                              zorder=4)
        else:
            non_flagged_seqs.append(seq_info)

            # Plot on the non-flagged axis with monochromatic color
            ax_non_flagged.plot(seq_data['x'], seq_data['y'],
                    linestyle='-',
                    linewidth=NORMAL_LINEWIDTH,
                    alpha=NORMAL_ALPHA,
                    color=NORMAL_COLOR,  # Use monochromatic color for normal sequences
                    zorder=1)

            # Plot points for each touchPhase with monochromatic colors
            for phase in TOUCH_PHASE_COLORS:
                phase_data = seq_data[seq_data['touchPhase'] == phase]
                if not phase_data.empty:
                    ax_non_flagged.scatter(phase_data['x'], phase_data['y'],
                              color=NORMAL_COLOR,  # Use monochromatic color for normal sequences
                              edgecolors=NORMAL_EDGE_COLOR,
                              s=NORMAL_MARKER_SIZE,
                              alpha=NORMAL_ALPHA,
                              label=f"{phase}" if phase not in ax_non_flagged.get_legend_handles_labels()[1] else "",
                              zorder=2)

    # Create legend elements

    # Touch phase colors for flagged sequences
    flagged_legend_elements = [
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor=color, markersize=8,
              label=f"{phase} (Flagged)")
        for phase, color in TOUCH_PHASE_COLORS.items()
    ]

    # Add line style for flagged sequences
    flagged_legend_elements.append(
        Line2D([0], [0], linestyle='--', color='red',
              alpha=FLAGGED_ALPHA, linewidth=FLAGGED_LINEWIDTH,
              label='Flagged Sequence (Multi-colored)')
    )

    # Monochromatic elements for normal sequences
    normal_legend_elements = [
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor=NORMAL_COLOR, markeredgecolor=NORMAL_EDGE_COLOR,
              markersize=8, alpha=NORMAL_ALPHA,
              label='Normal Sequence Points'),
        Line2D([0], [0], linestyle='-', color=NORMAL_COLOR,
              alpha=NORMAL_ALPHA, linewidth=NORMAL_LINEWIDTH,
              label='Normal Sequence (Monochromatic)')
    ]

    # Add legends with appropriate elements
    ax_flagged.legend(handles=flagged_legend_elements, loc='upper right', framealpha=0.7)
    ax_non_flagged.legend(handles=normal_legend_elements, loc='upper right', framealpha=0.7)

    # Calculate statistics
    flagged_count = len(flagged_seqs)
    non_flagged_count = len(non_flagged_seqs)

    flagged_avg_points = np.mean([seq['points'] for seq in flagged_seqs]) if flagged_seqs else 0
    non_flagged_avg_points = np.mean([seq['points'] for seq in non_flagged_seqs]) if non_flagged_seqs else 0

    flagged_avg_duration = np.mean([seq['duration'] for seq in flagged_seqs]) if flagged_seqs else 0
    non_flagged_avg_duration = np.mean([seq['duration'] for seq in non_flagged_seqs]) if non_flagged_seqs else 0

    # Calculate median values for more robust statistics
    flagged_median_points = np.median([seq['points'] for seq in flagged_seqs]) if flagged_seqs else 0
    non_flagged_median_points = np.median([seq['points'] for seq in non_flagged_seqs]) if non_flagged_seqs else 0

    flagged_median_duration = np.median([seq['duration'] for seq in flagged_seqs]) if flagged_seqs else 0
    non_flagged_median_duration = np.median([seq['duration'] for seq in non_flagged_seqs]) if non_flagged_seqs else 0

    # Set titles with statistics
    ax_flagged.set_title(f'Flagged Sequences (n={flagged_count})\n'
                        f'Avg Points: {flagged_avg_points:.1f}, '
                        f'Avg Duration: {flagged_avg_duration:.2f}s',
                        fontsize=12, color='darkred')

    ax_non_flagged.set_title(f'Non-Flagged Sequences (n={non_flagged_count})\n'
                            f'Avg Points: {non_flagged_avg_points:.1f}, '
                            f'Avg Duration: {non_flagged_avg_duration:.2f}s',
                            fontsize=10)

    # Set labels
    ax_flagged.set_xlabel('X Coordinate')
    ax_flagged.set_ylabel('Y Coordinate')
    ax_non_flagged.set_xlabel('X Coordinate')

    # Add grid
    ax_flagged.grid(True, linestyle='--', alpha=0.7)
    ax_non_flagged.grid(True, linestyle='--', alpha=0.7)

    # Ensure both plots have the same x limits
    x_min = min(ax_flagged.get_xlim()[0], ax_non_flagged.get_xlim()[0])
    x_max = max(ax_flagged.get_xlim()[1], ax_non_flagged.get_xlim()[1])
    ax_flagged.set_xlim(x_min, x_max)
    ax_non_flagged.set_xlim(x_min, x_max)

    # Add a super title with filename if available
    if csv_path:
        filename = get_base_filename(csv_path)
        plt.suptitle(f'View GamePlay: [{filename}] - Comparison of Flagged vs. Non-Flagged Sequences', fontsize=16)
    else:
        plt.suptitle('View GamePlay: Comparison of Flagged vs. Non-Flagged Sequences', fontsize=16)

    # Display detailed statistics in the bottom panel
    ax_stats.axis('off')  # Turn off axis

    # Create detailed statistics text
    stats_text = (
        f"SUMMARY STATISTICS:\n\n"
        f"Total Sequences: {flagged_count + non_flagged_count}   |   "
        f"Flagged: {flagged_count} ({flagged_count/(flagged_count + non_flagged_count)*100:.1f}%)   |   "
        f"Non-Flagged: {non_flagged_count} ({non_flagged_count/(flagged_count + non_flagged_count)*100:.1f}%)\n\n"

        f"FLAGGED SEQUENCES:\n"
        f"Mean Points: {flagged_avg_points:.1f}   |   "
        f"Median Points: {flagged_median_points:.1f}   |   "
        f"Mean Duration: {flagged_avg_duration:.2f}s   |   "
        f"Median Duration: {flagged_median_duration:.2f}s\n\n"

        f"FLAG TYPE COUNTS:\n"
    )

    # Add flag type counts
    flag_counts_text = ", ".join([f"{flag}: {count}" for flag, count in flag_type_counts.items() if count > 0])
    stats_text += flag_counts_text

    # Display the statistics
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)

    return fig


def create_temporal_visualization(df, seq_metrics=None, output_path=None, csv_path=None):  # seq_metrics is kept for API consistency
    """
    Create a temporal visualization with color gradients and animation options.

    Args:
        df (DataFrame): The touch sequence data
        seq_metrics (DataFrame, optional): Sequence metrics data (not used in this visualization)
        csv_path (str, optional): Path to the CSV file for title information

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get sequence iterator
    iterator_source, use_touchdata_id, is_composite_key = _get_sequence_iterator(df)

    # Create colormaps for temporal progression
    normal_cmap = plt.cm.Blues  # Muted colormap for normal sequences
    flagged_cmap = plt.cm.plasma  # Vibrant colormap for flagged sequences

    # Track min and max times for normalization
    min_time = df['time'].min()
    max_time = df['time'].max()
    time_range = max_time - min_time

    # Track which sequences have flags for the legend
    has_flagged_seq = False
    has_normal_seq = False

    # Count flagged and normal sequences
    flagged_count = 0
    normal_count = 0

    # Plot each sequence with color gradient
    for id_val in iterator_source:
        seq_data = pd.DataFrame()
        if use_touchdata_id:
            if pd.isna(id_val):
                continue
            seq_data = df[df['Touchdata_id'] == id_val].sort_values('time')
        elif is_composite_key:
            finger_id, seq_id = id_val
            # Skip sequences with seqId = 0
            if seq_id == 0:
                continue
            seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)].sort_values('time')
        else:
            break

        # Skip sequences with only one point
        if len(seq_data) <= 1:
            continue

        # Check if sequence has flags
        has_flags = sequence_has_flags(seq_data)

        # Set styling based on flags
        if has_flags:
            line_style = '--'
            line_width = FLAGGED_LINEWIDTH
            alpha = FLAGGED_ALPHA
            marker_size = FLAGGED_MARKER_SIZE
            cmap = flagged_cmap
            has_flagged_seq = True
            flagged_count += 1
            zorder_line = 3  # Higher zorder to draw on top
            zorder_marker = 4
        else:
            line_style = '-'
            line_width = NORMAL_LINEWIDTH
            alpha = NORMAL_ALPHA
            marker_size = NORMAL_MARKER_SIZE
            cmap = normal_cmap
            has_normal_seq = True
            normal_count += 1
            zorder_line = 1
            zorder_marker = 2

        # Get x and y coordinates
        x = seq_data['x'].values
        y = seq_data['y'].values

        # Normalize times to [0, 1] for color mapping
        times = seq_data['time'].values
        norm_times = (times - min_time) / time_range

        # Plot segments with color gradient
        for i in range(len(x) - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                   color=cmap(norm_times[i]),
                   linestyle=line_style,
                   linewidth=line_width,
                   alpha=alpha,
                   zorder=zorder_line)

        # Plot touchPhase points
        for phase in TOUCH_PHASE_COLORS:
            phase_data = seq_data[seq_data['touchPhase'] == phase]
            if not phase_data.empty:
                # Choose color based on whether it's a flagged or normal sequence
                if has_flags:
                    # Use multi-color scheme for flagged sequences
                    point_color = TOUCH_PHASE_COLORS[phase]
                    # Remove border for special touch events, keep border only for Moved points
                    if phase == 'Moved':
                        edge_color = 'black'
                    else:
                        edge_color = point_color  # Same as fill color to remove visible border
                    # Use phase-specific marker size for flagged sequences
                    phase_marker_size = FLAGGED_PHASE_MARKER_SIZES.get(phase, FLAGGED_MARKER_SIZE)
                else:
                    # Use monochromatic scheme for normal sequences
                    point_color = NORMAL_COLOR
                    edge_color = NORMAL_EDGE_COLOR
                    phase_marker_size = marker_size

                ax.scatter(phase_data['x'], phase_data['y'],
                          color=point_color,
                          edgecolors=edge_color,
                          s=phase_marker_size,
                          alpha=alpha,
                          label=f"{phase}" if has_flags and phase not in ax.get_legend_handles_labels()[1] else "",
                          zorder=zorder_marker)

    # Create legend elements
    legend_elements = []

    # Add touchPhase colors to legend only for flagged sequences
    if has_flagged_seq:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=color, markersize=8,
                  label=f"{phase} (Flagged Sequences)")
            for phase, color in TOUCH_PHASE_COLORS.items()
        ])

    # Add line style to legend
    if has_normal_seq:
        # Add monochromatic normal sequence to legend
        legend_elements.append(Line2D([0], [0], linestyle='-', color=NORMAL_COLOR,
                                     alpha=NORMAL_ALPHA, linewidth=NORMAL_LINEWIDTH,
                                     label='Normal Sequence (Monochromatic)'))
        # Add a marker for normal sequence points
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=NORMAL_COLOR, markeredgecolor=NORMAL_EDGE_COLOR,
                                     markersize=8, alpha=NORMAL_ALPHA,
                                     label='Normal Sequence Points'))

    if has_flagged_seq:
        # Add multi-colored flagged sequence to legend
        legend_elements.append(Line2D([0], [0], linestyle='--', color='purple',
                                     alpha=FLAGGED_ALPHA, linewidth=FLAGGED_LINEWIDTH,
                                     label='Flagged Sequence (Multi-colored)'))

    # Create colorbars for time progression
    # Create a figure-wide colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)  # Use a neutral colormap for the time scale
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Time Progression')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([
        f"{min_time:.1f}s",
        f"{min_time + time_range*0.25:.1f}s",
        f"{min_time + time_range*0.5:.1f}s",
        f"{min_time + time_range*0.75:.1f}s",
        f"{max_time:.1f}s"
    ])

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Set title with sequence counts and filename if available
    if csv_path:
        filename = get_base_filename(csv_path)
        title = f'View GamePlay: [{filename}] - Temporal Visualization\n'
    else:
        title = f'View GamePlay: Temporal Visualization\n'

    if has_flagged_seq and has_normal_seq:
        title += f'Flagged: {flagged_count} sequences (highlighted), Normal: {normal_count} sequences'
    elif has_flagged_seq:
        title += f'Showing {flagged_count} Flagged Sequences'
    elif has_normal_seq:
        title += f'Showing {normal_count} Normal Sequences'

    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for the colorbar
    plt.subplots_adjust(bottom=0.15)

    return fig


def get_base_filename(csv_path):
    """Extract the base filename from a CSV path (remove directory path and .csv extension)."""
    # Get just the filename without the directory
    filename = os.path.basename(csv_path)
    # Remove the .csv extension
    base_filename = os.path.splitext(filename)[0]
    return base_filename


def parse_flags_from_string(flags_str):
    """
    Parse flags from a string or other format, handling NaN and comma-separated values.

    Args:
        flags_str: The flags value (could be string, NaN, list, etc.)

    Returns:
        list: List of individual flag strings
    """
    # Handle different types of input
    if isinstance(flags_str, str):
        # Handle string cases
        if flags_str == '' or flags_str == 'nan':
            return []
        # Split comma-separated flags and clean them
        return [flag.strip() for flag in flags_str.split(',') if flag.strip()]
    elif isinstance(flags_str, list):
        # If it's already a list, filter out empty strings
        return [flag.strip() for flag in flags_str if flag and flag.strip()]
    elif flags_str is None:
        return []
    else:
        # For other types (including numpy arrays, pandas Series, etc.)
        try:
            # Check if it's a scalar NaN
            if pd.isna(flags_str):
                return []
        except (ValueError, TypeError):
            # If pd.isna fails (e.g., on lists), continue with other checks
            pass

        # If it's another iterable, try to use it
        try:
            return [str(flag).strip() for flag in flags_str if flag and str(flag).strip()]
        except (TypeError, AttributeError):
            # If it's not iterable or has issues, return empty list
            return []


def get_sequence_flags(seq_data):
    """
    Get all unique flags for a sequence, properly parsing string flags.

    Args:
        seq_data: DataFrame containing sequence data with 'flags' column

    Returns:
        set: Set of unique flags for this sequence
    """
    seq_flags = set()
    for flags_str in seq_data['flags']:
        flag_list = parse_flags_from_string(flags_str)
        seq_flags.update(flag_list)
    return seq_flags


def sequence_has_flags(seq_data):
    """
    Check if a sequence has any flags, properly parsing string flags.

    Args:
        seq_data: DataFrame containing sequence data with 'flags' column

    Returns:
        bool: True if sequence has any flags
    """
    for flags_str in seq_data['flags']:
        flag_list = parse_flags_from_string(flags_str)
        if flag_list:
            return True
    return False

def list_csv_files():
    """List all CSV files in the flagged_data directory."""
    if not os.path.exists(FLAGGED_DATA_DIR):
        print(f"Error: Directory '{FLAGGED_DATA_DIR}' not found.")
        return []

    csv_files = [f for f in os.listdir(FLAGGED_DATA_DIR) if f.endswith('.csv')]
    return csv_files

def interactive_file_selection():
    """Interactively select a CSV file from the flagged_data directory."""
    csv_files = list_csv_files()

    if not csv_files:
        print_warning(f"No CSV files found in '{FLAGGED_DATA_DIR}' directory.")
        return None

    print_colored("\n=== Available CSV Files ===", Fore.BLUE, Style.BRIGHT)
    for i, file in enumerate(csv_files, 1):
        print_colored(f"{i}. {file}", Fore.WHITE)

    while True:
        try:
            choice = input(f"\n{Fore.YELLOW}Enter the number of the file to visualize (or 'q' to quit): {Style.RESET_ALL}")
            if choice.lower() == 'q':
                return None

            choice = int(choice)
            if 1 <= choice <= len(csv_files):
                return csv_files[choice - 1]
            else:
                print_error(f"Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print_error("Please enter a valid number.")

def interactive_visualization_type():
    """Interactively select the visualization type."""
    viz_types = {
        '1': ('basic', 'Basic 2D Visualization - Simple view of all touch sequences'),
        '2': ('interactive', 'Interactive Visualization - Filter sequences by flag types'),
        '3': ('comparative', 'Comparative Visualization - Compare flagged vs non-flagged sequences'),
        '4': ('temporal', 'Temporal Visualization - View sequences with time progression')
    }

    print_colored("\n=== Visualization Types ===", Fore.BLUE, Style.BRIGHT)
    for key, (_, desc) in viz_types.items():
        print_colored(f"{key}. {desc}", Fore.WHITE)

    while True:
        choice = input(f"\n{Fore.YELLOW}Enter the number of the visualization type (or 'q' to quit): {Style.RESET_ALL}")
        if choice.lower() == 'q':
            return None

        if choice in viz_types:
            return viz_types[choice][0]
        else:
            print_error(f"Please enter a number between 1 and {len(viz_types)}.")

def interactive_output_option():
    """Interactively ask if the user wants to save the visualization."""
    while True:
        choice = input(f"\n{Fore.YELLOW}Do you want to save the visualization to a file? (y/n): {Style.RESET_ALL}")
        if choice.lower() == 'y':
            output_path = input(f"{Fore.YELLOW}Enter the output file path (e.g., 'visualization.png'): {Style.RESET_ALL}")
            return output_path
        elif choice.lower() == 'n':
            return None
        else:
            print_error("Please enter 'y' or 'n'.")

def display_menu():
    """Display the main menu."""
    clear_screen()
    print(HEADER)
    print_colored("Please select an option:", Fore.YELLOW, Style.BRIGHT)
    print_colored("1. Basic 2D Visualization - Simple view of all touch sequences", Fore.WHITE)
    print_colored("2. Interactive Visualization - Filter sequences by flag types", Fore.WHITE)
    print_colored("3. Comparative Visualization - Compare flagged vs non-flagged sequences", Fore.WHITE)
    print_colored("4. Temporal Visualization - View sequences with time progression", Fore.WHITE)
    print_colored("5. List available CSV files", Fore.WHITE)
    print_colored("6. Exit", Fore.WHITE)
    print()
    return input(f"{Fore.YELLOW}Enter your choice (1-6): {Style.RESET_ALL}")

def run_visualization(viz_type, csv_file=None, output_path=None):
    """Run a specific visualization type."""
    # If no CSV file is provided, ask for one
    if not csv_file:
        csv_file = interactive_file_selection()
        if not csv_file:
            return False

    # Ask about saving output (only for non-interactive visualizations)
    if viz_type != 'interactive' and not output_path:
        output_path = interactive_output_option()

    # Construct the full path to the CSV file
    csv_path = os.path.join(FLAGGED_DATA_DIR, csv_file)
    if not os.path.exists(csv_path):
        # Try with full path
        csv_path = csv_file
        if not os.path.exists(csv_path):
            print_error(f"CSV file not found at {csv_file} or {csv_path}")
            return False

    print_info(f"Loading data from {csv_path}...")

    try:
        # Load the data
        df, seq_metrics = load_data(csv_path)

        print_info(f"Creating {viz_type} visualization...")

        # CSV path will be passed to visualization functions for title information

        # Create the visualization based on the type
        if viz_type == 'basic':
            create_basic_visualization(df, seq_metrics, output_path, csv_path)
            plt.show()
        elif viz_type == 'interactive':
            result = create_interactive_visualization(df, seq_metrics, csv_path)
            # For HTML interactive visualization, we don't need plt.show()
            if isinstance(result, str) and result.endswith('.html'):
                print_info("Interactive HTML visualization created and opened in browser")
            else:
                # If it returned a matplotlib figure, show it
                plt.show()
        elif viz_type == 'comparative':
            create_comparative_visualization(df, seq_metrics, csv_path)
            plt.show()
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print_success(f"Visualization saved to {output_path}")
        elif viz_type == 'temporal':
            create_temporal_visualization(df, seq_metrics, csv_path)
            plt.show()
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print_success(f"Visualization saved to {output_path}")

        print_success("Visualization complete!")
        return True
    except Exception as e:
        print_error(f"Error creating visualization: {e}")
        return False

def list_available_files():
    """List all available CSV files."""
    csv_files = list_csv_files()
    if csv_files:
        print_colored("\n=== Available CSV Files ===", Fore.BLUE, Style.BRIGHT)
        for i, file in enumerate(csv_files, 1):
            print_colored(f"{i}. {file}", Fore.WHITE)
    else:
        print_warning(f"No CSV files found in '{FLAGGED_DATA_DIR}' directory.")
    return True

def main():
    """Main function to run the visualization tool."""
    # Check if arguments were provided
    if len(sys.argv) > 1:
        # Use command-line arguments
        parser = argparse.ArgumentParser(description='Touch Sequence Visualization Tool')
        parser.add_argument('csv_file', nargs='?', help='CSV file from flagged_data directory to visualize')
        parser.add_argument('--type', choices=['basic', 'interactive', 'comparative', 'temporal'],
                            default='interactive', help='Type of visualization to create')
        parser.add_argument('--output', help='Path to save the visualization (for non-interactive types)')
        parser.add_argument('--list', action='store_true', help='List available CSV files')

        args = parser.parse_args()

        # If --list flag is provided, just list the files and exit
        if args.list:
            list_available_files()
            return

        # Get CSV file path
        csv_file = args.csv_file if args.csv_file else interactive_file_selection()
        if not csv_file:
            return

        # Run the visualization
        run_visualization(args.type, csv_file, args.output)
    else:
        # Interactive menu-driven mode
        while True:
            choice = display_menu()

            if choice == '1':
                run_visualization('basic')
            elif choice == '2':
                run_visualization('interactive')
            elif choice == '3':
                run_visualization('comparative')
            elif choice == '4':
                run_visualization('temporal')
            elif choice == '5':
                list_available_files()
            elif choice == '6':
                print_colored("\nThank you for using the Touch Sequence Visualization Tool. Goodbye!", Fore.GREEN, Style.BRIGHT)
                break
            else:
                print_error("Invalid choice. Please try again.")

            input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")


# Check if we're already in a virtual environment
if __name__ == "__main__":
    if not is_venv_active():
        print_info("Virtual environment not active. Attempting to activate...")

        # Find the virtual environment
        venv_path, activate_script = find_venv_path()

        if venv_path and activate_script:
            print_info(f"Found virtual environment at {venv_path}")

            # Activate the virtual environment and restart the script
            try:
                # We can't directly source the activate script from Python
                # Instead, we'll create a temporary shell script that activates the venv and runs our script
                script_path = os.path.abspath(__file__)

                if sys.platform == 'win32':
                    # Windows approach
                    temp_script = 'temp_activate.bat'
                    with open(temp_script, 'w') as f:
                        f.write(f'@echo off\n')
                        f.write(f'call "{activate_script}"\n')
                        f.write(f'python "{script_path}"\n')

                    # Execute the batch file
                    os.system(temp_script)
                    # Clean up
                    os.remove(temp_script)
                else:
                    # Unix-like systems approach
                    temp_script = 'temp_activate.sh'
                    with open(temp_script, 'w') as f:
                        f.write(f'#!/bin/bash\n')
                        f.write(f'source "{activate_script}"\n')
                        f.write(f'python3 "{script_path}"\n')

                    # Make the script executable
                    os.chmod(temp_script, 0o755)
                    # Execute the shell script
                    os.system(f'./{temp_script}')
                    # Clean up
                    os.remove(temp_script)

                # Exit the original process
                sys.exit(0)
            except Exception as e:
                print_error(f"Error activating virtual environment: {e}")
                print_warning("Please activate the virtual environment manually and try again.")
        else:
            print_warning("No virtual environment found. Please create one with:")
            print_info("python -m venv venv")
            print_info("Then activate it and install required packages:")
            print_info("source venv/bin/activate  # On Unix/macOS")
            print_info("venv\\Scripts\\activate    # On Windows")
            print_info("pip install pandas numpy matplotlib seaborn")
    else:
        # If we're already in a virtual environment, run the main function
        main()
