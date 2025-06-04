#!/usr/bin/env python3
"""
HTML-based Interactive Visualization Generator
Creates interactive touch sequence visualizations using HTML, CSS, and JavaScript
"""

import pandas as pd
import json
import os
from datetime import datetime

def load_and_process_data(csv_path):
    """Load and process CSV data for HTML visualization."""
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Process flags column - handle both string and list formats
    if 'flags' in df.columns:
        def parse_flags(flags_val):
            if pd.isna(flags_val) or flags_val == '' or flags_val == 'nan':
                return []
            if isinstance(flags_val, str):
                if flags_val.startswith('[') and flags_val.endswith(']'):
                    # Handle string representation of list
                    try:
                        return eval(flags_val)
                    except:
                        return [flag.strip() for flag in flags_val.split(',') if flag.strip()]
                else:
                    return [flag.strip() for flag in flags_val.split(',') if flag.strip()]
            elif isinstance(flags_val, list):
                return [flag.strip() for flag in flags_val if flag and flag.strip()]
            else:
                return []
        
        df['flags_parsed'] = df['flags'].apply(parse_flags)
    else:
        df['flags_parsed'] = [[] for _ in range(len(df))]
    
    return df

def prepare_visualization_data(df):
    """Prepare data for JavaScript visualization."""
    
    # Get all unique flags
    all_flags = set()
    for flags_list in df['flags_parsed']:
        all_flags.update(flags_list)
    all_flags = sorted(list(all_flags))
    
    # Group data by sequences
    sequences = []
    
    # Check if we have Touchdata_id or need to use fingerId+seqId
    if 'Touchdata_id' in df.columns:
        # Use Touchdata_id for grouping
        for touchdata_id in df['Touchdata_id'].dropna().unique():
            seq_data = df[df['Touchdata_id'] == touchdata_id].copy()
            if seq_data.empty:
                continue
                
            # Get flags for this sequence
            seq_flags = set()
            for flags_list in seq_data['flags_parsed']:
                seq_flags.update(flags_list)
            
            sequence = {
                'id': str(touchdata_id),
                'type': 'touchdata_id',
                'points': seq_data[['x', 'y', 'time', 'touchPhase']].to_dict('records'),
                'flags': list(seq_flags),
                'has_flags': len(seq_flags) > 0
            }
            sequences.append(sequence)
    
    elif 'fingerId' in df.columns and 'seqId' in df.columns:
        # Use fingerId + seqId for grouping
        for (finger_id, seq_id) in df[['fingerId', 'seqId']].drop_duplicates().values:
            if seq_id == 0:  # Skip invalid sequences
                continue
                
            seq_data = df[(df['fingerId'] == finger_id) & (df['seqId'] == seq_id)].copy()
            if seq_data.empty:
                continue
                
            # Get flags for this sequence
            seq_flags = set()
            for flags_list in seq_data['flags_parsed']:
                seq_flags.update(flags_list)
            
            sequence = {
                'id': f"{finger_id}_{seq_id}",
                'type': 'finger_seq',
                'fingerId': int(finger_id),
                'seqId': int(seq_id),
                'points': seq_data[['x', 'y', 'time', 'touchPhase']].to_dict('records'),
                'flags': list(seq_flags),
                'has_flags': len(seq_flags) > 0
            }
            sequences.append(sequence)
    
    # Calculate statistics
    flagged_sequences = [seq for seq in sequences if seq['has_flags']]
    normal_sequences = [seq for seq in sequences if not seq['has_flags']]
    
    # Count flag occurrences
    flag_counts = {}
    for flag in all_flags:
        flag_counts[flag] = sum(1 for seq in sequences if flag in seq['flags'])
    
    return {
        'sequences': sequences,
        'all_flags': all_flags,
        'flag_counts': flag_counts,
        'stats': {
            'total_sequences': len(sequences),
            'flagged_sequences': len(flagged_sequences),
            'normal_sequences': len(normal_sequences),
            'flagged_percentage': round(len(flagged_sequences) / len(sequences) * 100, 1) if sequences else 0
        }
    }

def generate_html_visualization(data, csv_filename, output_path):
    """Generate the HTML file with interactive visualization."""
    
    # Get base filename for title
    base_filename = os.path.splitext(os.path.basename(csv_filename))[0]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Touch Sequence Visualization - {base_filename}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        
        .controls {{
            padding: 20px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .control-group {{
            display: inline-block;
            margin-right: 30px;
            vertical-align: top;
        }}
        
        .control-group h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .checkbox-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .checkbox-item {{
            display: flex;
            align-items: center;
            background: white;
            padding: 8px 12px;
            border-radius: 20px;
            border: 2px solid #e9ecef;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }}
        
        .checkbox-item:hover {{
            border-color: #667eea;
            transform: translateY(-1px);
        }}
        
        .checkbox-item.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .radio-group {{
            display: flex;
            gap: 10px;
        }}
        
        .radio-item {{
            padding: 8px 16px;
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }}
        
        .radio-item:hover {{
            border-color: #667eea;
        }}
        
        .radio-item.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .visualization {{
            padding: 20px;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        #chart {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }}
        
        .sequence-line {{
            fill: none;
            stroke-width: 2;
            opacity: 0.7;
        }}
        
        .sequence-point {{
            stroke-width: 1;
            opacity: 0.8;
        }}
        
        .flagged {{
            stroke-dasharray: 5,5;
        }}
        
        .normal {{
            stroke: #6c757d;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Interactive Touch Sequence Visualization</h1>
            <p>File: {base_filename} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h3>🏷️ Flag Filters</h3>
                <div class="checkbox-group" id="flag-checkboxes">
                    <!-- Flag checkboxes will be generated by JavaScript -->
                </div>
            </div>
            
            <div class="control-group">
                <h3>👁️ View Mode</h3>
                <div class="radio-group" id="view-mode">
                    <div class="radio-item active" data-mode="all">All Sequences</div>
                    <div class="radio-item" data-mode="flagged">Flagged Only</div>
                    <div class="radio-item" data-mode="normal">Normal Only</div>
                </div>
            </div>
        </div>
        
        <div class="visualization">
            <div class="stats" id="stats">
                <!-- Stats will be updated by JavaScript -->
            </div>
            
            <svg id="chart" width="100%" height="600"></svg>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    
    <script>
        // Data from Python
        const data = {json.dumps(data, indent=2)};
        
        // Color schemes
        const touchPhaseColors = {{
            'Began': '#e74c3c',
            'Moved': '#3498db', 
            'Ended': '#2ecc71',
            'Canceled': '#f39c12',
            'Stationary': '#9b59b6',
            'B': '#e74c3c',
            'M': '#3498db',
            'E': '#2ecc71',
            'S': '#9b59b6'
        }};
        
        const flagColors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
        ];
        
        // State
        let activeFlags = new Set(data.all_flags);
        let viewMode = 'all';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            initializeControls();
            updateVisualization();
        }});
        
        function initializeControls() {{
            // Initialize flag checkboxes
            const flagContainer = document.getElementById('flag-checkboxes');
            data.all_flags.forEach((flag, index) => {{
                const checkbox = document.createElement('div');
                checkbox.className = 'checkbox-item active';
                checkbox.textContent = `${{flag}} (${{data.flag_counts[flag]}})`;
                checkbox.addEventListener('click', () => toggleFlag(flag, checkbox));
                flagContainer.appendChild(checkbox);
            }});
            
            // Initialize view mode radio buttons
            document.querySelectorAll('.radio-item').forEach(item => {{
                item.addEventListener('click', () => {{
                    document.querySelectorAll('.radio-item').forEach(r => r.classList.remove('active'));
                    item.classList.add('active');
                    viewMode = item.dataset.mode;
                    updateVisualization();
                }});
            }});
        }}
        
        function toggleFlag(flag, element) {{
            if (activeFlags.has(flag)) {{
                activeFlags.delete(flag);
                element.classList.remove('active');
            }} else {{
                activeFlags.add(flag);
                element.classList.add('active');
            }}
            updateVisualization();
        }}
        
        function updateVisualization() {{
            // Filter sequences based on current settings
            const filteredSequences = data.sequences.filter(seq => {{
                // View mode filter
                if (viewMode === 'flagged' && !seq.has_flags) return false;
                if (viewMode === 'normal' && seq.has_flags) return false;
                
                // Flag filter (for flagged sequences)
                if (seq.has_flags) {{
                    return seq.flags.some(flag => activeFlags.has(flag));
                }}
                
                return true;
            }});
            
            updateStats(filteredSequences);
            drawChart(filteredSequences);
        }}
        
        function updateStats(sequences) {{
            const flaggedCount = sequences.filter(s => s.has_flags).length;
            const normalCount = sequences.filter(s => !s.has_flags).length;
            const totalCount = sequences.length;
            
            document.getElementById('stats').innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${{totalCount}}</div>
                    <div class="stat-label">Total Sequences</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{flaggedCount}}</div>
                    <div class="stat-label">Flagged</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{normalCount}}</div>
                    <div class="stat-label">Normal</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{totalCount > 0 ? Math.round(flaggedCount/totalCount*100) : 0}}%</div>
                    <div class="stat-label">Flagged %</div>
                </div>
            `;
        }}
        
        function drawChart(sequences) {{
            const svg = d3.select('#chart');
            svg.selectAll('*').remove();
            
            const margin = {{top: 20, right: 20, bottom: 40, left: 40}};
            const width = parseInt(svg.style('width')) - margin.left - margin.right;
            const height = 600 - margin.top - margin.bottom;
            
            const g = svg.append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
            
            if (sequences.length === 0) {{
                g.append('text')
                    .attr('x', width/2)
                    .attr('y', height/2)
                    .attr('text-anchor', 'middle')
                    .style('font-size', '18px')
                    .style('fill', '#6c757d')
                    .text('No sequences match current filters');
                return;
            }}
            
            // Get all points for scaling
            const allPoints = sequences.flatMap(seq => seq.points);
            const xExtent = d3.extent(allPoints, d => d.x);
            const yExtent = d3.extent(allPoints, d => d.y);
            
            const xScale = d3.scaleLinear()
                .domain(xExtent)
                .range([0, width]);
                
            const yScale = d3.scaleLinear()
                .domain(yExtent)
                .range([height, 0]);
            
            // Add axes
            g.append('g')
                .attr('transform', `translate(0,${{height}})`)
                .call(d3.axisBottom(xScale));
                
            g.append('g')
                .call(d3.axisLeft(yScale));
            
            // Add axis labels
            g.append('text')
                .attr('x', width/2)
                .attr('y', height + 35)
                .attr('text-anchor', 'middle')
                .style('font-size', '12px')
                .text('X Coordinate');
                
            g.append('text')
                .attr('transform', 'rotate(-90)')
                .attr('x', -height/2)
                .attr('y', -25)
                .attr('text-anchor', 'middle')
                .style('font-size', '12px')
                .text('Y Coordinate');
            
            // Draw sequences
            sequences.forEach((seq, seqIndex) => {{
                const line = d3.line()
                    .x(d => xScale(d.x))
                    .y(d => yScale(d.y));
                
                // Draw line
                const pathColor = seq.has_flags ? flagColors[seqIndex % flagColors.length] : '#6c757d';
                g.append('path')
                    .datum(seq.points)
                    .attr('class', seq.has_flags ? 'sequence-line flagged' : 'sequence-line normal')
                    .attr('d', line)
                    .style('stroke', pathColor);
                
                // Draw points
                seq.points.forEach(point => {{
                    const pointColor = touchPhaseColors[point.touchPhase] || '#333';
                    g.append('circle')
                        .attr('class', 'sequence-point')
                        .attr('cx', xScale(point.x))
                        .attr('cy', yScale(point.y))
                        .attr('r', seq.has_flags ? 4 : 3)
                        .style('fill', pointColor)
                        .style('stroke', seq.has_flags ? pathColor : '#333')
                        .on('mouseover', function(event) {{
                            showTooltip(event, seq, point);
                        }})
                        .on('mouseout', hideTooltip);
                }});
            }});
        }}
        
        function showTooltip(event, sequence, point) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
            
            const flagsText = sequence.flags.length > 0 ? sequence.flags.join(', ') : 'None';
            tooltip.innerHTML = `
                <strong>Sequence:</strong> ${{sequence.id}}<br>
                <strong>Point:</strong> (${{point.x.toFixed(1)}}, ${{point.y.toFixed(1)}})<br>
                <strong>Phase:</strong> ${{point.touchPhase}}<br>
                <strong>Time:</strong> ${{point.time}}<br>
                <strong>Flags:</strong> ${{flagsText}}
            `;
        }}
        
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}
    </script>
</body>
</html>"""
    
    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML visualization saved to: {output_path}")
    return output_path

def create_html_interactive_visualization(csv_path, output_path=None):
    """Main function to create HTML interactive visualization."""
    
    if not output_path:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"interactive_viz_{base_name}.html"
    
    try:
        # Load and process data
        df = load_and_process_data(csv_path)
        
        # Prepare visualization data
        viz_data = prepare_visualization_data(df)
        
        # Generate HTML file
        html_path = generate_html_visualization(viz_data, csv_path, output_path)
        
        return html_path
        
    except Exception as e:
        print(f"❌ Error creating HTML visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python html_interactive_viz.py <csv_file> [output_file]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = create_html_interactive_visualization(csv_file, output_file)
    if result:
        print(f"\n🎉 Interactive visualization created successfully!")
        print(f"📂 Open this file in your web browser: {result}")
