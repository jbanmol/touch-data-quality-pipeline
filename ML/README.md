# ML-Based Touch Data Cleaning System

This directory contains a comprehensive machine learning system for cleaning and enhancing Coloring touchdata while preserving all original data integrity.

## 🎯 Key Features

- **Data Preservation**: Never modifies original coordinates or timing data
- **Rich Metadata**: Adds quality scores, behavioral classifications, and usage recommendations
- **Transfer Learning**: Pre-trains on synthetic data, fine-tunes on Coloring patterns
- **Anomaly Detection**: Identifies outliers without removing them
- **Behavioral Analysis**: Classifies interaction patterns and user intent
- **Quality Assessment**: Provides confidence scores for different analysis types

## 📁 File Structure

```
ML/
├── README.md                    # This file
├── setup_ml_environment.py     # Environment setup script
├── example_usage.py            # Usage examples and demonstrations
├── ml_clean_coloring_data.py   # Command-line interface
├── ml_cleaning_pipeline.py     # Main ML cleaning pipeline
├── metadata_enhancer.py        # Quality assessment and behavioral analysis
├── feature_engineering.py      # Feature extraction for ML models
├── transfer_learning_model.py  # Transfer learning implementation
├── cleaning.py                 # Enhanced version of original cleaning
├── models/                     # Directory for trained models
├── examples/                   # Example outputs
└── enhanced_data/              # Enhanced JSON outputs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies and setup environment
python ML/setup_ml_environment.py
```

### 2. Run Example

```bash
# See the system in action
python ML/example_usage.py
```

### 3. Process Your Data

```bash
# Process all JSON files in raw_JSONs directory
python ML/ml_clean_coloring_data.py --input raw_JSONs --output enhanced_data

# Process a single file
python ML/ml_clean_coloring_data.py --file raw_JSONs/Coloring_2022-01-12.json --output enhanced_file.json

# Generate report only (no output files)
python ML/ml_clean_coloring_data.py --input raw_JSONs --report-only
```

## 🧠 How It Works

### 1. Feature Engineering
- **Temporal Features**: Time differences, sequence positions, rhythm consistency
- **Spatial Features**: Movement distances, velocities, direction changes
- **Behavioral Features**: Touch phase transitions, completion rates, zone changes
- **Quality Features**: Sequence completeness, pattern validity, outlier indicators

### 2. Quality Assessment
- **Sequence Completeness**: Validates touch sequence patterns (Began → Moved/Stationary → Ended)
- **Temporal Consistency**: Analyzes timing patterns and identifies irregularities
- **Spatial Consistency**: Evaluates movement smoothness and coherence
- **Overall Quality Score**: Combines multiple metrics into a single 0-1 score

### 3. Behavioral Classification
- **Interaction Styles**: Deliberate, interrupted, irregular, incomplete
- **Movement Types**: Stationary, smooth, erratic, variable
- **Pattern Recognition**: Tap, drag, hold, complex interactions
- **Intent Confidence**: Likelihood that the interaction was intentional

### 4. Anomaly Detection
- **Outlier Identification**: Uses Isolation Forest to detect unusual data points
- **Confidence Scoring**: Provides confidence levels for anomaly classifications
- **Context Preservation**: Flags anomalies without removing them

### 5. Transfer Learning
- **Pre-training**: Learns general touch patterns from synthetic data
- **Fine-tuning**: Adapts to Coloring-specific interaction patterns
- **Pattern Recognition**: Identifies common and unusual interaction sequences

## 📊 Output Format

The enhanced JSON maintains the original structure while adding ML metadata:

```json
{
  "message": "gameData",
  "json": {
    "dataSet": "Coloring",
    "touchData": {
      "1": [
        {
          "x": 800.0,
          "y": 504.0,
          "time": 3192.546,
          "touchPhase": "Began",
          "ml_quality_score": 0.92,
          "quality_tier": "high",
          "behavioral_pattern": "deliberate",
          "anomaly_score": 0.05,
          "usage_recommendations": ["timing_analysis", "spatial_analysis"]
        }
      ]
    }
  },
  "ml_metadata": {
    "processing_timestamp": "2024-01-15T10:30:00",
    "total_sequences": 5,
    "quality_distribution": {"high": 3, "medium": 2, "low": 0},
    "behavioral_patterns": {"deliberate": 4, "interrupted": 1}
  }
}
```

## 🎯 Usage Recommendations

The system provides specific recommendations for how to use each data point:

- **`timing_analysis`**: High-quality temporal data suitable for timing studies
- **`spatial_analysis`**: Clean spatial data good for movement analysis
- **`completion_tracking`**: Reliable completion percentage data
- **`user_intent_analysis`**: Clear intentional interactions
- **`error_analysis`**: Interrupted or erratic patterns worth studying
- **`outlier_investigation`**: Anomalous data points requiring special attention
- **`primary_analysis`**: Highest quality data for main research questions
- **`secondary_analysis`**: Good quality data for supporting analysis
- **`exploratory_analysis`**: Lower quality data suitable for exploration

## 🔧 Configuration

Edit `ML/ml_config.json` to customize:

```json
{
  "ml_settings": {
    "enable_transfer_learning": true,
    "pretrain_epochs": 30,
    "batch_size": 16
  },
  "quality_assessment": {
    "quality_thresholds": {
      "high": 0.8,
      "medium": 0.5,
      "low": 0.0
    }
  }
}
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_ml_cleaning.py -v
```

## 📈 Benefits

1. **Scientific Integrity**: Preserves all original data for reproducible research
2. **Enhanced Analysis**: Rich metadata enables more sophisticated analysis
3. **Quality Guidance**: Helps researchers choose appropriate data for specific studies
4. **Anomaly Insights**: Identifies interesting edge cases without losing them
5. **Behavioral Understanding**: Provides insights into user interaction patterns
6. **Automated Assessment**: Reduces manual data quality evaluation time

## 🤝 Integration

The ML cleaning system integrates seamlessly with the existing pipeline:

```python
from ML.cleaning import clean_data_with_enhanced_ml

# Use in your existing code
enhanced_df = clean_data_with_enhanced_ml(df, data_type='Coloring')
```

## 📝 Notes

- All original coordinate data (x, y) is preserved unchanged
- All original timing data is preserved unchanged
- All original touch phases are preserved unchanged
- ML enhancements are added as additional metadata columns
- The system gracefully falls back to basic cleaning if ML components are unavailable
- Transfer learning models are automatically trained on first use
- Processing is optimized for batch operations on multiple files

## 🆘 Troubleshooting

**Issue**: ML components not loading
**Solution**: Run `python ML/setup_ml_environment.py` to install dependencies

**Issue**: Out of memory during processing
**Solution**: Process files individually or reduce batch size in configuration

**Issue**: Slow processing
**Solution**: Disable transfer learning in configuration for faster processing

**Issue**: Missing models directory
**Solution**: The system automatically creates required directories on first run
