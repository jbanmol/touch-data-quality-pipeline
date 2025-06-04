#!/usr/bin/env python3
"""
Unified Touch Data Processing System - Central Launcher

This script provides a comprehensive command-line interface for the entire touch data processing ecosystem,
including data conversion, processing, visualization, ML pipelines, testing, and export functionality.

Features:
- Data Processing (JSON to CSV conversion, CSV flagging/validation)
- Visualization Tools (Basic 2D, Interactive HTML, Comparative, Temporal)
- ML Pipeline Operations (Data cleaning, metadata enhancement, feature engineering)
- Export Functions (Google Sheets upload)
- Testing and Validation utilities

Usage:
    python app.py

Author: Augment Agent
"""

import os
import sys
import subprocess
import re

# Virtual environment detection
def is_venv_active():
    """Check if a virtual environment is already active."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_required_packages():
    """Check if all required packages are installed."""
    # Default required packages if requirements.txt can't be read
    default_packages = [
        'pandas',
        'numpy',
        'gspread',
        'gspread_formatting',
        'oauth2client'
    ]

    # Read packages from requirements.txt
    required_packages = []
    try:
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                for line in f:
                    # Skip empty lines and comments
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract just the package name without version
                        package_name = line.split('>=')[0].split('==')[0].strip()
                        required_packages.append(package_name)
        else:
            print("Warning: requirements.txt not found, using default package list")
            required_packages = default_packages
    except Exception as e:
        print(f"Error reading requirements.txt: {e}")
        required_packages = default_packages

    # Check which packages are missing
    missing_packages = []

    # Map package names to their import names
    import_name_map = {
        'python-dateutil': 'dateutil',
        'scikit-learn': 'sklearn',
        'gspread_formatting': 'gspread_formatting',
        'oauth2client': 'oauth2client'
    }

    for package in required_packages:
        try:
            # Get the correct import name
            import_name = import_name_map.get(package, package)
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    return missing_packages

def install_missing_packages(packages):
    """Install missing packages using pip."""
    if not packages:
        return True

    print("Installing missing packages...")

    # Always try to install from requirements.txt first if it exists
    if os.path.exists('requirements.txt'):
        try:
            print("Installing packages from requirements.txt...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("Successfully installed all packages from requirements.txt")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing from requirements.txt: {e}")
            print("Trying individual package installation...")

    # If requirements.txt doesn't exist or failed, install packages individually
    success = True
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            success = False

    return success

# Check if we're already in a virtual environment
if not is_venv_active():
    print("Virtual environment not active.")
    print("Please activate the virtual environment before running this script:")
    print("  source venv/bin/activate  # On Unix/macOS")
    print("  venv\\Scripts\\activate    # On Windows")
    print("\nIf you haven't created a virtual environment yet, create one with:")
    print("  python -m venv venv")
    print("\nAfter activation, install required packages with:")
    print("  pip install -r requirements.txt")
    print("\nThen run this script again.")
    sys.exit(1)

# If we get here, we're running in an activated virtual environment
# Now import the rest of the modules
import datetime
import logging
import subprocess
import importlib.util
from src.utils import json_converter
from src.core import data_processor
# Import the Google Sheets uploader
from src.export import google_sheets

# Check for missing packages and ensure they're installed
def ensure_requirements():
    """Check for required packages and install them if missing."""
    print("Checking for required packages...")

    # Create requirements.txt if it doesn't exist
    if not os.path.exists('requirements.txt'):
        print("Creating requirements.txt file...")
        default_packages = [
            'pandas>=2.0.0',
            'numpy>=1.22.0',
            'gspread>=5.0.0',
            'gspread_formatting>=1.0.0',
            'oauth2client>=4.1.0',
            'python-dateutil>=2.8.0',
            'pytz>=2020.1'
        ]
        try:
            with open('requirements.txt', 'w') as f:
                f.write('\n'.join(default_packages) + '\n')
            print("Created requirements.txt file")
        except Exception as e:
            print(f"Warning: Could not create requirements.txt: {e}")

    # Check for missing packages
    missing_packages = check_required_packages()

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")

        # Try to install missing packages
        if not install_missing_packages(missing_packages):
            print("\nError: Could not install all required packages.")
            print("Please run the following command manually:")
            print("  pip install -r requirements.txt")
            sys.exit(1)

        # Verify installation was successful
        still_missing = check_required_packages()
        if still_missing:
            print(f"\nError: The following packages could not be installed: {', '.join(still_missing)}")
            print("Please install them manually with:")
            print("  pip install -r requirements.txt")
            sys.exit(1)

        print("All required packages installed successfully.")
    else:
        print("All required packages are already installed.")

    return True

# Ensure all requirements are met
ensure_requirements()

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ASCII Art for the app header
HEADER = f"""
{Fore.GREEN}
 _____                 _       _____        _          _____                           _
|_   _|               | |     |  __ \\      | |        |  __ \\                         (_)
  | | ___  _   _  ___| |__   | |  | | __ _| |_ __ _  | |__) | __ ___   ___ ___  ___ _ _ __   __ _
  | |/ _ \\| | | |/ __| '_ \\  | |  | |/ _` | __/ _` | |  ___/ '__/ _ \\ / __/ _ \\/ __| | '_ \\ / _` |
  | | (_) | |_| | (__| | | | | |__| | (_| | || (_| | | |   | | | (_) | (_|  __/\\__ \\ | | | | (_| |
  \\_/\\___/ \\__,_|\\___|_| |_| |_____/ \\__,_|\\__\\__,_| |_|   |_|  \\___/ \\___\\___||___/_|_| |_|\\__, |
                                                                                             __/ |
                                                                                            |___/
{Style.RESET_ALL}
{Fore.CYAN}🎮 Unified Touch Data Processing System 🎮{Style.RESET_ALL}
{Fore.MAGENTA}Complete toolkit for touch data analysis, visualization, and ML processing{Style.RESET_ALL}
"""

# Folder paths
RAW_JSON_FOLDER = 'data/raw/json'
RAW_CSV_FOLDER = 'data/raw/csv'
FLAGGED_DATA_FOLDER = 'data/processed/flagged'

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

def print_progress(message, progress, total):
    """Print progress bar."""
    percent = int(progress / total * 100)
    bar_length = 40
    filled_length = int(bar_length * progress // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"\r{Fore.BLUE}{message}: {Fore.YELLOW}|{bar}| {percent}% ({progress}/{total}){Style.RESET_ALL}", end='')
    if progress == total:
        print()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_user_input(prompt, validator=None, default=None):
    """
    Get validated user input.

    Args:
        prompt (str): The prompt to display to the user
        validator (function, optional): A function that validates the input
        default (str, optional): Default value if user enters nothing

    Returns:
        str: The validated user input
    """
    while True:
        if default:
            display_prompt = f"{prompt} [{default}]: "
        else:
            display_prompt = f"{prompt}: "

        print(f"{Fore.YELLOW}{Style.BRIGHT}{display_prompt}{Style.RESET_ALL}", end='')

        user_input = input()

        if not user_input and default:
            return default

        if not validator or validator(user_input):
            return user_input

        print_error("Invalid input. Please try again.")

def check_folders():
    """Check if required folders exist and create them if needed."""
    folders = [RAW_JSON_FOLDER, RAW_CSV_FOLDER, FLAGGED_DATA_FOLDER]
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                print_success(f"Created folder: {folder}")
            except Exception as e:
                print_error(f"Failed to create folder {folder}: {e}")
                return False
    return True

def convert_json_to_csv():
    """Run the JSON to CSV conversion process."""
    print_info("Starting JSON to CSV conversion...")

    # Check if there are any JSON files
    if not os.path.exists(RAW_JSON_FOLDER):
        print_error(f"Folder {RAW_JSON_FOLDER} does not exist!")
        return False

    json_files = [f for f in os.listdir(RAW_JSON_FOLDER) if f.endswith('.json')]
    if not json_files:
        print_warning(f"No JSON files found in {RAW_JSON_FOLDER}")
        return False

    print_info(f"Found {len(json_files)} JSON files to convert")

    # Run the conversion
    try:
        successful = json_converter.convert_json_to_csv(RAW_JSON_FOLDER, RAW_CSV_FOLDER)
        if successful:
            print_success(f"Successfully converted {successful} JSON files to CSV")
            return True
        else:
            print_warning("No files were successfully converted")
            return False
    except Exception as e:
        print_error(f"Error during conversion: {e}")
        return False

def process_csv_files():
    """Run the CSV data processing pipeline."""
    print_info("Starting CSV data processing...")

    # Check if there are any CSV files
    if not os.path.exists(RAW_CSV_FOLDER):
        print_error(f"Folder {RAW_CSV_FOLDER} does not exist!")
        return False

    csv_files = [f for f in os.listdir(RAW_CSV_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print_warning(f"No CSV files found in {RAW_CSV_FOLDER}")
        return False

    print_info(f"Found {len(csv_files)} CSV files to process")

    # Run the processing
    try:
        successful = data_processor.batch_process_csv_files(RAW_CSV_FOLDER, FLAGGED_DATA_FOLDER)
        if successful:
            print_success(f"Successfully processed {successful} CSV files")
            return True
        else:
            print_warning("No files were successfully processed")
            return False
    except Exception as e:
        print_error(f"Error during processing: {e}")
        return False


def export_to_sheets():
    """Export processed data to Google Sheets."""
    print_info("Preparing to export data to Google Sheets...")

    # Check if there are any processed CSV files
    if not os.path.exists(FLAGGED_DATA_FOLDER):
        print_error(f"Folder {FLAGGED_DATA_FOLDER} does not exist!")
        return False

    csv_files = [f for f in os.listdir(FLAGGED_DATA_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print_warning(f"No CSV files found in {FLAGGED_DATA_FOLDER}")
        return False

    print_info(f"Found {len(csv_files)} CSV files to export")

    # Get spreadsheet name
    default_name = f"Kidaura Data Export - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    spreadsheet_name = get_user_input("Enter spreadsheet name", default=default_name)

    # Get email address
    email = None
    share_option = get_user_input("Share the spreadsheet with your email? (y/n)",
                                 validator=lambda x: x.lower() in ['y', 'yes', 'n', 'no'],
                                 default='y')

    if share_option.lower() in ['y', 'yes']:
        email = get_user_input("Enter your email address", validator=validate_email)

    # Run the export
    try:
        print_info("Exporting data to Google Sheets...")
        print_info("This may take a few moments. Please be patient...")

        # Start the export process
        google_sheets.upload_all_csvs(FLAGGED_DATA_FOLDER, spreadsheet_name, email)

        print_success("Data successfully exported to Google Sheets")
        return True
    except Exception as e:
        print_error(f"Error during export: {e}")
        return False

def run_complete_pipeline():
    """Run the complete data processing pipeline."""
    print_colored("Running the complete data processing pipeline...", Fore.MAGENTA, Style.BRIGHT)

    # Step 1: Convert JSON to CSV
    print_colored("\n=== STEP 1: JSON to CSV Conversion ===", Fore.BLUE, Style.BRIGHT)
    if not convert_json_to_csv():
        if input(f"{Fore.YELLOW}Continue to next step anyway? (y/n): {Style.RESET_ALL}").lower() != 'y':
            return

    # Step 2: Process CSV data
    print_colored("\n=== STEP 2: CSV Data Processing ===", Fore.BLUE, Style.BRIGHT)
    if not process_csv_files():
        if input(f"{Fore.YELLOW}Continue to next step anyway? (y/n): {Style.RESET_ALL}").lower() != 'y':
            return

    # Step 3: Export to Google Sheets
    print_colored("\n=== STEP 3: Export to Google Sheets ===", Fore.BLUE, Style.BRIGHT)
    export_to_sheets()

    print_colored("\n🎉 Complete pipeline execution finished! 🎉", Fore.GREEN, Style.BRIGHT)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def run_visualization_tools():
    """Launch the visualization tools interface."""
    print_colored("\n=== Touch Data Visualization Tools ===", Fore.BLUE, Style.BRIGHT)
    print_info("Launching visualization interface...")

    try:
        # Import and run the visualization tool
        import subprocess
        result = subprocess.run([sys.executable, 'src/visualization/views.py'],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print_success("Visualization tool completed successfully")
            return True
        else:
            print_warning("Visualization tool exited with warnings")
            return False
    except Exception as e:
        print_error(f"Error launching visualization tools: {e}")
        return False

def run_basic_visualization():
    """Run basic 2D visualization."""
    print_info("Starting basic 2D visualization...")

    # Check if there are processed files
    if not os.path.exists(FLAGGED_DATA_FOLDER):
        print_error(f"No processed data found in {FLAGGED_DATA_FOLDER}")
        print_info("Please run data processing first (option 2 or 3)")
        return False

    csv_files = [f for f in os.listdir(FLAGGED_DATA_FOLDER) if f.endswith('.csv') and f != 'summary.csv']
    if not csv_files:
        print_warning("No CSV files found for visualization")
        return False

    try:
        # Import visualization functions
        sys.path.append('src/visualization')
        from src.visualization.views import run_visualization

        print_info(f"Found {len(csv_files)} files available for visualization")
        result = run_visualization('basic')
        return result
    except Exception as e:
        print_error(f"Error running basic visualization: {e}")
        return False

def run_interactive_visualization():
    """Run interactive HTML visualization."""
    print_info("Starting interactive HTML visualization...")

    # Check if there are processed files
    if not os.path.exists(FLAGGED_DATA_FOLDER):
        print_error(f"No processed data found in {FLAGGED_DATA_FOLDER}")
        print_info("Please run data processing first (option 2 or 3)")
        return False

    try:
        # Import visualization functions
        sys.path.append('src/visualization')
        from src.visualization.views import run_visualization

        result = run_visualization('interactive')
        return result
    except Exception as e:
        print_error(f"Error running interactive visualization: {e}")
        return False

# ============================================================================
# ML PIPELINE FUNCTIONS
# ============================================================================

def check_ml_dependencies():
    """Check if ML dependencies are available."""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        return True
    except ImportError as e:
        print_warning(f"ML dependencies not fully available: {e}")
        print_info("Some ML features may not work. Install with: pip install scikit-learn")
        return False

def run_ml_cleaning_pipeline():
    """Run the ML-based data cleaning pipeline."""
    print_colored("\n=== ML Data Cleaning Pipeline ===", Fore.BLUE, Style.BRIGHT)
    print_info("Starting ML-based data cleaning and enhancement...")

    # Check ML dependencies
    if not check_ml_dependencies():
        print_warning("ML dependencies not available. Using basic cleaning instead.")
        return False

    # Check for input data
    if not os.path.exists(RAW_CSV_FOLDER):
        print_error(f"No raw CSV data found in {RAW_CSV_FOLDER}")
        print_info("Please convert JSON files first (option 2)")
        return False

    csv_files = [f for f in os.listdir(RAW_CSV_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print_warning(f"No CSV files found in {RAW_CSV_FOLDER}")
        return False

    try:
        # Try to use the enhanced ML cleaning function
        from ML.cleaning import clean_data_with_enhanced_ml

        print_info(f"Found {len(csv_files)} CSV files for ML processing")

        # Process files
        output_dir = "data/processed/ml_enhanced"
        os.makedirs(output_dir, exist_ok=True)

        successful = 0
        for csv_file in csv_files:
            input_path = os.path.join(RAW_CSV_FOLDER, csv_file)
            output_path = os.path.join(output_dir, f"ml_enhanced_{csv_file}")

            print_info(f"Processing {csv_file} with ML pipeline...")

            # Load and process the CSV file
            import pandas as pd
            df = pd.read_csv(input_path)

            # Detect data type
            data_type = 'Coloring' if 'color' in df.columns and 'completionPerc' in df.columns else 'Tracing'

            # Apply ML enhancement
            enhanced_df = clean_data_with_enhanced_ml(df, data_type)
            enhanced_df.to_csv(output_path, index=False)

            successful += 1
            print_success(f"Enhanced {csv_file} -> {output_path}")

        print_success(f"ML pipeline completed successfully. Enhanced {successful} files.")
        print_info(f"Enhanced files saved to: {output_dir}")
        return True

    except ImportError as e:
        print_error(f"ML pipeline not available: {e}")
        print_info("ML components may not be properly installed")
        return False
    except Exception as e:
        print_error(f"Error in ML pipeline: {e}")
        return False

def run_feature_engineering():
    """Run feature engineering on processed data."""
    print_colored("\n=== Feature Engineering ===", Fore.BLUE, Style.BRIGHT)
    print_info("Running feature engineering on processed data...")

    if not check_ml_dependencies():
        return False

    # Check for processed data
    if not os.path.exists(FLAGGED_DATA_FOLDER):
        print_error(f"No processed data found in {FLAGGED_DATA_FOLDER}")
        print_info("Please run data processing first")
        return False

    try:
        from ML.feature_engineering import TouchFeatureEngineer

        feature_engineer = TouchFeatureEngineer()
        csv_files = [f for f in os.listdir(FLAGGED_DATA_FOLDER) if f.endswith('.csv') and f != 'summary.csv']

        if not csv_files:
            print_warning("No CSV files found for feature engineering")
            return False

        output_dir = "data/processed/features"
        os.makedirs(output_dir, exist_ok=True)

        successful = 0
        for csv_file in csv_files:
            input_path = os.path.join(FLAGGED_DATA_FOLDER, csv_file)
            output_path = os.path.join(output_dir, f"features_{csv_file}")

            print_info(f"Extracting features from {csv_file}...")

            import pandas as pd
            df = pd.read_csv(input_path)

            # Use the correct method name for feature extraction
            enhanced_df = feature_engineer.extract_all_features(df)
            enhanced_df.to_csv(output_path, index=False)

            successful += 1
            print_success(f"Features extracted: {csv_file} -> {output_path}")

        print_success(f"Feature engineering completed. Processed {successful} files.")
        print_info(f"Feature files saved to: {output_dir}")
        return True

    except Exception as e:
        print_error(f"Error in feature engineering: {e}")
        return False

# ============================================================================
# TESTING AND VALIDATION FUNCTIONS
# ============================================================================

def run_integration_tests():
    """Run integration tests to validate the system."""
    print_colored("\n=== Integration Tests ===", Fore.BLUE, Style.BRIGHT)
    print_info("Running integration tests...")

    try:
        # Run the main integration test
        result = subprocess.run([sys.executable, 'tests/integration_test.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print_success("Integration tests passed!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print_error("Integration tests failed!")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            return False

    except Exception as e:
        print_error(f"Error running integration tests: {e}")
        return False

def run_performance_tests():
    """Run performance tests."""
    print_colored("\n=== Performance Tests ===", Fore.BLUE, Style.BRIGHT)
    print_info("Running performance tests...")

    try:
        result = subprocess.run([sys.executable, 'tests/performance_test.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print_success("Performance tests completed!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print_warning("Performance tests completed with warnings")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            return False

    except Exception as e:
        print_error(f"Error running performance tests: {e}")
        return False

def run_validation_tests():
    """Run data validation tests."""
    print_colored("\n=== Data Validation Tests ===", Fore.BLUE, Style.BRIGHT)
    print_info("Running data validation tests...")

    validation_tests = [
        'tests/test_coloring_validation.py',
        'tests/test_touchdata_id_validation.py',
        'tests/test_export_validation.py',
        'tests/test_column_ordering.py'
    ]

    passed = 0
    failed = 0

    for test_file in validation_tests:
        if os.path.exists(test_file):
            print_info(f"Running {os.path.basename(test_file)}...")
            try:
                result = subprocess.run([sys.executable, test_file],
                                      capture_output=True, text=True)

                if result.returncode == 0:
                    print_success(f"✓ {os.path.basename(test_file)} passed")
                    passed += 1
                else:
                    print_error(f"✗ {os.path.basename(test_file)} failed")
                    failed += 1
                    if result.stderr:
                        print(f"  Error: {result.stderr.strip()}")

            except Exception as e:
                print_error(f"✗ {os.path.basename(test_file)} error: {e}")
                failed += 1
        else:
            print_warning(f"Test file not found: {test_file}")

    print_colored(f"\nValidation Results: {passed} passed, {failed} failed",
                 Fore.GREEN if failed == 0 else Fore.YELLOW, Style.BRIGHT)
    return failed == 0

def run_all_tests():
    """Run all available tests."""
    print_colored("\n=== Running All Tests ===", Fore.MAGENTA, Style.BRIGHT)

    results = []

    # Integration tests
    print_colored("\n1. Integration Tests", Fore.CYAN, Style.BRIGHT)
    results.append(("Integration", run_integration_tests()))

    # Validation tests
    print_colored("\n2. Validation Tests", Fore.CYAN, Style.BRIGHT)
    results.append(("Validation", run_validation_tests()))

    # Performance tests
    print_colored("\n3. Performance Tests", Fore.CYAN, Style.BRIGHT)
    results.append(("Performance", run_performance_tests()))

    # Summary
    print_colored("\n=== Test Summary ===", Fore.MAGENTA, Style.BRIGHT)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_type, result in results:
        status = "PASSED" if result else "FAILED"
        color = Fore.GREEN if result else Fore.RED
        print_colored(f"{test_type}: {status}", color)

    print_colored(f"\nOverall: {passed}/{total} test suites passed",
                 Fore.GREEN if passed == total else Fore.YELLOW, Style.BRIGHT)

    return passed == total

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_system_status():
    """Show the current system status and available data."""
    print_colored("\n=== System Status ===", Fore.BLUE, Style.BRIGHT)

    # Check directories and file counts
    directories = {
        "Raw JSON Files": RAW_JSON_FOLDER,
        "Raw CSV Files": RAW_CSV_FOLDER,
        "Processed/Flagged Data": FLAGGED_DATA_FOLDER,
        "ML Enhanced Data": "data/processed/ml_enhanced",
        "Feature Data": "data/processed/features",
        "Visualizations": "data/outputs/visualizations"
    }

    for name, path in directories.items():
        if os.path.exists(path):
            if name.endswith("Data") or name.endswith("Files"):
                files = [f for f in os.listdir(path) if f.endswith(('.json', '.csv'))]
                count = len(files)
                print_success(f"{name}: {count} files in {path}")
            else:
                files = os.listdir(path)
                count = len(files)
                print_success(f"{name}: {count} items in {path}")
        else:
            print_warning(f"{name}: Directory {path} does not exist")

    # Check dependencies
    print_colored("\n=== Dependencies Status ===", Fore.BLUE, Style.BRIGHT)

    core_deps = ['pandas', 'numpy', 'gspread']
    ml_deps = ['sklearn', 'matplotlib', 'seaborn']

    print_info("Core Dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            print_success(f"✓ {dep}")
        except ImportError:
            print_error(f"✗ {dep}")

    print_info("ML/Visualization Dependencies:")
    for dep in ml_deps:
        try:
            __import__(dep)
            print_success(f"✓ {dep}")
        except ImportError:
            print_warning(f"⚠ {dep} (optional)")

def show_help():
    """Show detailed help information."""
    print_colored("\n=== Touch Data Processing System Help ===", Fore.BLUE, Style.BRIGHT)

    help_text = """
🔄 DATA PROCESSING:
   • JSON to CSV Conversion: Converts raw JSON touch data to CSV format
   • CSV Processing: Applies flagging, validation, and quality checks
   • Complete Pipeline: Runs conversion + processing + export in sequence

📊 VISUALIZATION TOOLS:
   • Basic 2D: Simple matplotlib visualization of touch sequences
   • Interactive HTML: Web-based interactive visualization with filtering
   • Comparative: Side-by-side comparison of flagged vs normal sequences
   • Temporal: Time-based visualization with animation options

🧠 ML PIPELINE:
   • ML Cleaning: Advanced data cleaning with machine learning
   • Feature Engineering: Extract statistical and behavioral features
   • Metadata Enhancement: Add quality scores and behavioral classifications

📤 EXPORT FUNCTIONS:
   • Google Sheets: Upload processed data to Google Sheets with sharing

🧪 TESTING & VALIDATION:
   • Integration Tests: End-to-end system testing
   • Performance Tests: Speed and memory usage analysis
   • Validation Tests: Data integrity and format validation

💡 TIPS:
   • Always run data processing before visualization
   • ML features require scikit-learn installation
   • Google Sheets export requires credentials.json setup
   • Use system status to check available data and dependencies
    """

    print(help_text)

    print_colored("📁 DIRECTORY STRUCTURE:", Fore.CYAN, Style.BRIGHT)
    structure = """
   data/raw/json/          - Place raw JSON files here
   data/raw/csv/           - Converted CSV files
   data/processed/flagged/ - Processed files with flags
   data/outputs/           - Generated reports and visualizations
   ML/                     - Machine learning components
   tests/                  - Test scripts
   config/                 - Configuration files
    """
    print(structure)

def display_menu():
    """Display the main menu."""
    print(HEADER)
    print_colored("=== MAIN MENU ===", Fore.YELLOW, Style.BRIGHT)
    print()

    # Data Processing Section
    print_colored("📊 DATA PROCESSING:", Fore.CYAN, Style.BRIGHT)
    print_colored("  1. Run complete pipeline (JSON→CSV→Processing→Export)", Fore.WHITE)
    print_colored("  2. Convert JSON files to CSV", Fore.WHITE)
    print_colored("  3. Process CSV files (flagging & validation)", Fore.WHITE)
    print()

    # Visualization Section
    print_colored("📈 VISUALIZATION TOOLS:", Fore.CYAN, Style.BRIGHT)
    print_colored("  4. Launch visualization interface (all tools)", Fore.WHITE)
    print_colored("  5. Basic 2D visualization", Fore.WHITE)
    print_colored("  6. Interactive HTML visualization", Fore.WHITE)
    print()

    # ML Pipeline Section
    print_colored("🧠 ML PIPELINE:", Fore.CYAN, Style.BRIGHT)
    print_colored("  7. ML data cleaning & enhancement", Fore.WHITE)
    print_colored("  8. Feature engineering", Fore.WHITE)
    print()

    # Export Section
    print_colored("📤 EXPORT:", Fore.CYAN, Style.BRIGHT)
    print_colored("  9. Export to Google Sheets", Fore.WHITE)
    print()

    # Testing Section
    print_colored("🧪 TESTING & VALIDATION:", Fore.CYAN, Style.BRIGHT)
    print_colored(" 10. Run integration tests", Fore.WHITE)
    print_colored(" 11. Run validation tests", Fore.WHITE)
    print_colored(" 12. Run performance tests", Fore.WHITE)
    print_colored(" 13. Run all tests", Fore.WHITE)
    print()

    # Utilities Section
    print_colored("🔧 UTILITIES:", Fore.CYAN, Style.BRIGHT)
    print_colored(" 14. Show system status", Fore.WHITE)
    print_colored(" 15. Show help", Fore.WHITE)
    print_colored(" 16. Exit", Fore.WHITE)
    print()

    return input(f"{Fore.YELLOW}Enter your choice (1-16): {Style.RESET_ALL}")

def main():
    """Main function to run the interactive CLI."""
    # Check and create required folders
    if not check_folders():
        print_error("Failed to set up required folders. Exiting...")
        return

    while True:
        clear_screen()
        choice = display_menu()

        try:
            # Data Processing
            if choice == '1':
                run_complete_pipeline()
            elif choice == '2':
                convert_json_to_csv()
            elif choice == '3':
                process_csv_files()

            # Visualization Tools
            elif choice == '4':
                run_visualization_tools()
            elif choice == '5':
                run_basic_visualization()
            elif choice == '6':
                run_interactive_visualization()

            # ML Pipeline
            elif choice == '7':
                run_ml_cleaning_pipeline()
            elif choice == '8':
                run_feature_engineering()

            # Export
            elif choice == '9':
                export_to_sheets()

            # Testing & Validation
            elif choice == '10':
                run_integration_tests()
            elif choice == '11':
                run_validation_tests()
            elif choice == '12':
                run_performance_tests()
            elif choice == '13':
                run_all_tests()

            # Utilities
            elif choice == '14':
                show_system_status()
            elif choice == '15':
                show_help()
            elif choice == '16':
                print_colored("\n🎉 Thank you for using the Touch Data Processing System! 🎉", Fore.GREEN, Style.BRIGHT)
                print_colored("Goodbye!", Fore.GREEN, Style.BRIGHT)
                break
            else:
                print_error("Invalid choice. Please enter a number between 1-16.")

        except KeyboardInterrupt:
            print_colored("\n\nOperation interrupted by user.", Fore.YELLOW)
            if input(f"{Fore.YELLOW}Return to main menu? (y/n): {Style.RESET_ALL}").lower() != 'y':
                break
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            print_info("Please try again or contact support if the issue persists.")

        if choice != '16':  # Don't pause if exiting
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
