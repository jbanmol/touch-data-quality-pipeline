#!/usr/bin/env python3
"""
Interactive Data Processing CLI

This script provides an interactive command-line interface for the entire data processing workflow,
including JSON to CSV conversion, data processing, and Google Sheets export.

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
import json_to_csv_converter
import process_csv_data
# Import the Google Sheets uploader
import upload_to_sheets

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
 _  ___     _                        _____        _          _____                           _
| |/ (_)   | |                      |  __ \\      | |        |  __ \\                         (_)
| ' / _  __| | __ _ _   _ _ __ __ _ | |  | | __ _| |_ __ _  | |__) | __ ___   ___ ___  ___ _ _ __   __ _
|  < | |/ _` |/ _` | | | | '__/ _` || |  | |/ _` | __/ _` | |  ___/ '__/ _ \\ / __/ _ \\/ __| | '_ \\ / _` |
| . \\| | (_| | (_| | |_| | | | (_| || |__| | (_| | || (_| | | |   | | | (_) | (_|  __/\\__ \\ | | | | (_| |
|_|\\_\\_|\\__,_|\\__,_|\\__,_|_|  \\__,_||_____/ \\__,_|\\__\\__,_| |_|   |_|  \\___/ \\___\\___||___/_|_| |_|\\__, |
                                                                                                     __/ |
                                                                                                    |___/
{Style.RESET_ALL}
{Fore.CYAN}Interactive Data Processing CLI{Style.RESET_ALL}
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
        successful = json_to_csv_converter.convert_json_to_csv(RAW_JSON_FOLDER, RAW_CSV_FOLDER)
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
        successful = process_csv_data.batch_process_csv_files(RAW_CSV_FOLDER, FLAGGED_DATA_FOLDER)
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
        upload_to_sheets.upload_all_csvs(FLAGGED_DATA_FOLDER, spreadsheet_name, email)

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

def display_menu():
    """Display the main menu."""
    print(HEADER)
    print_colored("Please select an option:", Fore.YELLOW, Style.BRIGHT)
    print_colored("1. Run complete pipeline (conversion, processing, and export)", Fore.WHITE)
    print_colored("2. Convert JSON files to CSV only", Fore.WHITE)
    print_colored("3. Process CSV files only", Fore.WHITE)
    print_colored("4. Export processed data to Google Sheets only", Fore.WHITE)
    print_colored("5. Exit", Fore.WHITE)
    print()
    return input(f"{Fore.YELLOW}Enter your choice (1-5): {Style.RESET_ALL}")

def main():
    """Main function to run the interactive CLI."""
    # Check and create required folders
    if not check_folders():
        print_error("Failed to set up required folders. Exiting...")
        return

    while True:
        clear_screen()
        choice = display_menu()

        if choice == '1':
            run_complete_pipeline()
        elif choice == '2':
            convert_json_to_csv()
        elif choice == '3':
            process_csv_files()
        elif choice == '4':
            export_to_sheets()
        elif choice == '5':
            print_colored("Thank you for using Kidaura Data Processing CLI. Goodbye!", Fore.GREEN, Style.BRIGHT)
            break
        else:
            print_error("Invalid choice. Please try again.")

        input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
