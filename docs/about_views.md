# Touch Sequence Visualization Guide

This guide explains the visualization tools available in our application for analyzing touch sequence data. These visualizations help you understand how users interact with the screen and identify potential issues in touch recognition.

## What Are Touch Sequences?

Touch sequences are series of screen interactions that begin when a finger touches the screen and end when it lifts off. Each sequence contains multiple touch points with information about:
- Position (x,y coordinates)
- Touch phase (Began, Moved, Stationary, Ended, or Canceled)
- Timestamp
- Other interaction data

## Our Visualization Tools

The application offers four different ways to visualize touch sequence data:

## 1. Basic 2D Visualization

![Basic Visualization Example](https://via.placeholder.com/600x400?text=Basic+Visualization+Example)

### What it shows
- All touch sequences plotted on a 2D coordinate system
- Normal sequences in light blue
- Flagged sequences in vibrant colors
- Different markers for each touch phase (Began, Moved, Stationary, Ended, Canceled)

### How it works
This visualization takes raw touch data from CSV files and plots each sequence as a connected line. The visualization:
- Draws each touch point at its exact x,y screen position
- Connects points in chronological order
- Uses different colors and markers to distinguish touch phases
- Makes flagged sequences stand out with brighter colors and thicker lines

### Why it's accurate
- Each point is plotted at its exact recorded position
- No data manipulation occurs - what you see is the raw data
- The scale matches the device screen dimensions
- Touch phases are color-coded exactly as recorded in the data

### Insights you can gain
- Overall touch patterns across the screen
- Areas with high touch density
- Unusual touch patterns that might indicate issues
- Visual identification of flagged sequences

## 2. Interactive Visualization

![Interactive Visualization Example](https://via.placeholder.com/600x400?text=Interactive+Visualization+Example)

### What it shows
- Filterable view of touch sequences
- Controls to show/hide sequences by flag type
- Statistics about sequence counts and flags
- Option to toggle between all sequences and flagged-only view

### How it works
This visualization builds on the basic view but adds interactive controls that:
- Filter sequences based on flag types (missing_Began, missing_Ended, etc.)
- Calculate and display statistics about the visible sequences
- Allow toggling between different view modes
- Provide reset functionality to return to default settings

### Why it's accurate
- Uses the same plotting technique as the basic visualization
- Filtering doesn't alter the underlying data, only what's displayed
- Statistics are calculated directly from the source data
- All touch points maintain their exact positions and properties

### Insights you can gain
- Patterns specific to certain flag types
- Comparative analysis between different flag categories
- Statistical understanding of flag distribution
- Focused examination of problematic sequences

## 3. Comparative Visualization

![Comparative Visualization Example](https://via.placeholder.com/600x400?text=Comparative+Visualization+Example)

### What it shows
- Side-by-side comparison of flagged vs. non-flagged sequences
- More screen space dedicated to flagged sequences for better visibility
- Statistical comparison between the two groups
- Identical scale and coordinate systems for accurate comparison

### How it works
This visualization splits the screen into two panels:
- Left panel (larger): Shows only flagged sequences
- Right panel: Shows only normal sequences
- Both use the same coordinate system and scale
- Statistical information is calculated and displayed for each panel

### Why it's accurate
- Both panels use identical scaling and coordinate systems
- No data manipulation occurs during visualization
- Statistical calculations are performed on the raw data
- Color coding and markers remain consistent with other visualizations

### Insights you can gain
- Direct comparison between normal and problematic touch patterns
- Visual identification of differences in touch behavior
- Understanding of how flagged sequences differ from normal ones
- Statistical comparison between the two groups

## 4. Temporal Visualization

![Temporal Visualization Example](https://via.placeholder.com/600x400?text=Temporal+Visualization+Example)

### What it shows
- Touch sequences with color gradients showing time progression
- Direction and speed of finger movement
- Time-based color coding to show sequence flow
- Emphasis on flagged sequences with higher opacity and thicker lines

### How it works
This visualization adds a time dimension to the 2D plot by:
- Using color gradients along each path to show direction of movement
- Displaying a color bar to indicate the time scale
- Maintaining the same position accuracy as other visualizations
- Emphasizing flagged sequences with higher opacity and thicker lines

### Why it's accurate
- Time information comes directly from the timestamp data
- Color gradients are calculated based on the exact sequence duration
- Spatial positioning remains identical to other visualizations
- No temporal compression or expansion occurs

### Insights you can gain
- Speed of finger movement across the screen
- Pauses or hesitations in touch sequences
- Direction and flow of touch gestures
- Temporal patterns that might indicate user confusion or app issues

## Understanding Flag Types

Our system flags touch sequences that may indicate problems:

| Flag Type | Meaning | Why It Matters |
|-----------|---------|----------------|
| missing_Began | Sequence doesn't start with a "Began" touch phase | May indicate touch detection started late |
| missing_Ended | Sequence doesn't end with an "Ended" touch phase | May indicate incomplete touch detection |
| short_duration | Sequence lasted less than 10 milliseconds | May be too brief for intentional interaction |
| too_few_points | Sequence has fewer than 3 touch points | May indicate incomplete gesture capture |
| has_canceled | Sequence contains a "Canceled" touch phase | Indicates the system canceled the touch |

## How to Use the Visualization Tool

1. Run the application with: `python3 views.py`
2. Select a CSV file containing touch sequence data
3. Choose one of the four visualization types
4. For non-interactive visualizations, decide whether to save the output
5. Explore the visualization to gain insights about touch patterns

For advanced usage, command-line options are available:
```bash
python3 views.py [csv_file] [--type TYPE] [--output OUTPUT_PATH] [--list]
```

## Troubleshooting

If you encounter issues with the visualizations:

- For display problems, try resizing the window
- For performance issues, close other applications
- For interactive control issues, try clicking and waiting a moment
- For Mac-specific issues, ensure you have the required packages installed

## Need Help?

If you need assistance interpreting the visualizations or have questions about the data, please contact our support team.
