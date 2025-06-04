# 🎮 Unified Touch Data Processing System

## Overview

The main `app.py` file now serves as a **unified entry point** for the entire touch data processing ecosystem. This central launcher provides easy access to all functionality without needing to know which specific script to run.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the unified system
python app.py
```

## Main Menu Structure

### 📊 DATA PROCESSING
- **Option 1**: Complete pipeline (JSON→CSV→Processing→Export)
- **Option 2**: Convert JSON files to CSV only
- **Option 3**: Process CSV files (flagging & validation) only

### 📈 VISUALIZATION TOOLS
- **Option 4**: Launch full visualization interface (all tools)
- **Option 5**: Basic 2D visualization (matplotlib)
- **Option 6**: Interactive HTML visualization (web-based)

### 🧠 ML PIPELINE
- **Option 7**: ML data cleaning & enhancement
- **Option 8**: Feature engineering

### 📤 EXPORT
- **Option 9**: Export to Google Sheets

### 🧪 TESTING & VALIDATION
- **Option 10**: Integration tests
- **Option 11**: Validation tests
- **Option 12**: Performance tests
- **Option 13**: Run all tests

### 🔧 UTILITIES
- **Option 14**: System status (shows available data & dependencies)
- **Option 15**: Help (detailed information about all features)
- **Option 16**: Exit

## Key Features

### ✅ Preserved Functionality
- **100% backward compatibility** - all existing functionality preserved
- **Independent operation** - individual scripts still work as before
- **Same user experience** - familiar interfaces maintained

### ✅ Enhanced User Experience
- **Unified entry point** - single command to access everything
- **Clear categorization** - organized by function type
- **Robust error handling** - graceful failure recovery
- **Colored output** - consistent visual feedback
- **Input validation** - prevents invalid selections

### ✅ Smart Integration
- **Dependency checking** - verifies required packages
- **Directory management** - creates folders as needed
- **Status monitoring** - shows system health
- **Help system** - comprehensive guidance

## Usage Examples

### First-Time Setup
1. Run `python app.py`
2. Choose option 14 (System Status) to check dependencies
3. Choose option 15 (Help) to understand available features

### Complete Data Processing Workflow
1. Place JSON files in `data/raw/json/`
2. Run option 1 (Complete pipeline)
3. Follow prompts for Google Sheets export

### Visualization Workflow
1. Ensure processed data exists (run processing first)
2. Choose option 4 for full visualization interface
3. Or choose options 5-6 for specific visualization types

### ML Enhancement Workflow
1. Ensure CSV data exists
2. Choose option 7 for ML cleaning
3. Choose option 8 for feature engineering

## Directory Structure

```
data/
├── raw/
│   ├── json/          # Place raw JSON files here
│   └── csv/           # Converted CSV files
├── processed/
│   ├── flagged/       # Processed files with flags
│   ├── ml_enhanced/   # ML-enhanced data
│   └── features/      # Feature-engineered data
└── outputs/
    ├── reports/       # Processing reports
    └── visualizations/ # Generated visualizations

ML/                    # Machine learning components
tests/                 # Test scripts
config/                # Configuration files
```

## Dependencies

### Core Dependencies (Required)
- pandas
- numpy
- gspread (for Google Sheets export)

### Optional Dependencies
- scikit-learn (for ML features)
- matplotlib (for visualizations)
- seaborn (for enhanced visualizations)

## Error Handling

The system includes robust error handling:
- **Missing dependencies**: Graceful degradation with informative messages
- **Missing directories**: Automatic creation when possible
- **Invalid input**: Clear error messages and retry prompts
- **Interrupted operations**: Return to main menu option

## Tips for Best Results

1. **Check system status first** (option 14) to verify setup
2. **Run data processing before visualization** - visualizations need processed data
3. **Use help system** (option 15) for detailed feature explanations
4. **Test with small datasets first** before processing large batches
5. **Keep virtual environment activated** for best compatibility

## Troubleshooting

### Common Issues
- **"Virtual environment not active"**: Run `source venv/bin/activate` first
- **"No CSV files found"**: Run data processing (options 2-3) first
- **"ML dependencies not available"**: Install with `pip install scikit-learn`
- **"Google Sheets error"**: Ensure `config/credentials.json` is set up

### Getting Help
- Use option 15 in the main menu for comprehensive help
- Use option 14 to check system status and dependencies
- Individual scripts still work independently if needed

## Legacy Compatibility

All existing command-line interfaces remain functional:
- `python src/visualization/views.py` - Direct visualization access
- `python cli/app.py` - Legacy CLI interface
- Individual test scripts in `tests/` directory
- ML scripts in `ML/` directory

The unified launcher simply provides a more convenient way to access everything in one place.
