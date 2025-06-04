#!/usr/bin/env python3
"""
Google Sheets Uploader

This script uploads all CSV files from the flagged_data folder to a Google Spreadsheet.
It creates a new spreadsheet with the summary.csv as the first sheet, followed by all other CSV files.

Usage:
    python upload_to_sheets.py [spreadsheet_name] [your_email@example.com]

Arguments:
    spreadsheet_name: Optional. Name for the Google Spreadsheet.
                     Default: "Flagged Data Export - YYYY-MM-DD"
    your_email@example.com: Optional. Your email address to share the spreadsheet with.
                           Default: "jbanmol9@gmail.com" (automatically shared with edit access)
                           This allows you to access and modify the spreadsheet directly in your Google Drive.

Author: Augment Agent
"""

import os
import sys
import csv
import time
import random
import gspread
from gspread_formatting import (
    CellFormat, Color, TextFormat, format_cell_range, set_frozen,
    get_conditional_format_rules, ConditionalFormatRule, GridRange,
    BooleanRule, BooleanCondition, set_data_validation_for_cell_range,
    DataValidationRule
)
from oauth2client.service_account import ServiceAccountCredentials
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def retry_with_backoff(func, *args, max_retries=5, initial_delay=0.5, **kwargs):
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            if "Quota exceeded" in str(e) and retries < max_retries:
                # Add jitter to avoid synchronized retries
                sleep_time = delay + (random.randint(0, 1000) / 1000.0)
                logger.warning(f"API quota exceeded. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                retries += 1
                delay *= 2  # Exponential backoff
            else:
                raise
        except Exception as e:
            # For other exceptions, don't retry
            raise

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def authenticate_google_apis(credentials_file):
    """
    Authenticate with Google APIs using service account credentials.

    Args:
        credentials_file (str): Path to the service account credentials JSON file

    Returns:
        gspread.Client: Authenticated gspread client
    """
    try:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_file, SCOPES)
        client = gspread.authorize(credentials)
        logger.info("Successfully authenticated with Google APIs")
        return client
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)

def create_spreadsheet(client, title):
    """
    Create a new Google Spreadsheet.

    Args:
        client (gspread.Client): Authenticated gspread client
        title (str): Title of the new spreadsheet

    Returns:
        gspread.Spreadsheet: The created spreadsheet
    """
    try:
        spreadsheet = client.create(title)
        logger.info(f"Created spreadsheet: {title}")
        logger.info(f"Spreadsheet URL: {spreadsheet.url}")
        return spreadsheet
    except Exception as e:
        logger.error(f"Failed to create spreadsheet: {e}")
        sys.exit(1)

def share_spreadsheet(spreadsheet, email, role='writer'):
    """
    Share the spreadsheet with a specific email address.

    Args:
        spreadsheet (gspread.Spreadsheet): The spreadsheet to share
        email (str): Email address to share with
        role (str): Permission level ('reader', 'writer', 'owner'). Default is 'writer' for edit access.
    """
    try:
        spreadsheet.share(email, perm_type='user', role=role)
        logger.info(f"Shared spreadsheet with {email} (role: {role})")
    except Exception as e:
        logger.error(f"Failed to share spreadsheet: {e}")

def read_csv_file(file_path):
    """
    Read a CSV file and return its contents as a list of rows.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        list: List of rows from the CSV file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return []

def sanitize_sheet_name(name):
    """
    Sanitize sheet name to comply with Google Sheets requirements.

    Args:
        name (str): Original sheet name

    Returns:
        str: Sanitized sheet name
    """
    # Remove the .csv extension if present
    if name.endswith('.csv'):
        name = name[:-4]

    # Replace invalid characters
    sanitized = name.replace(':', '-').replace('/', '-').replace('\\', '-')

    # Limit to 100 characters (Google Sheets limit)
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    return sanitized

def consolidate_ranges(ranges):
    """
    Consolidate adjacent ranges into larger ranges to reduce API calls.

    Args:
        ranges (list): List of ranges in format "start:end"

    Returns:
        list: Consolidated list of ranges
    """
    if not ranges:
        return []

    # Parse ranges into start and end values
    parsed_ranges = []
    for r in ranges:
        start, end = map(int, r.split(':'))
        parsed_ranges.append((start, end))

    # Sort by start value
    parsed_ranges.sort()

    # Consolidate adjacent ranges
    consolidated = []
    current_start, current_end = parsed_ranges[0]

    for start, end in parsed_ranges[1:]:
        if start <= current_end + 1:
            # Ranges overlap or are adjacent, extend the current range
            current_end = max(current_end, end)
        else:
            # Ranges are not adjacent, add the current range and start a new one
            consolidated.append(f"{current_start}:{current_end}")
            current_start, current_end = start, end

    # Add the last range
    consolidated.append(f"{current_start}:{current_end}")

    return consolidated

def format_sheet(_, worksheet, is_summary=False):
    """
    Apply formatting to a worksheet with modern and sleek styling.
    Optimized to reduce API calls by consolidating ranges and using batch operations.

    Args:
        _ (gspread.Spreadsheet): The spreadsheet containing the worksheet (not used but kept for API consistency)
        worksheet (gspread.Worksheet): The worksheet to format
        is_summary (bool): Whether this is the summary sheet
    """
    try:
        # Get the number of rows and columns
        rows = worksheet.row_count
        cols = worksheet.col_count

        logger.info(f"Formatting worksheet with {rows} rows and {cols} columns")

        # Light brown header format with green text (tree colors)
        header_format = CellFormat(
            backgroundColor=Color(0.76, 0.6, 0.42),  # Light brown
            textFormat=TextFormat(bold=True, foregroundColor=Color(0.13, 0.55, 0.13)),  # Forest green text
            horizontalAlignment='CENTER',
            borders={
                "bottom": {"style": "SOLID_MEDIUM"},
                "top": {"style": "SOLID_MEDIUM"},
                "left": {"style": "SOLID_MEDIUM"},
                "right": {"style": "SOLID_MEDIUM"}
            }
        )

        # Apply header formatting - use retry with backoff to handle rate limiting
        retry_with_backoff(format_cell_range, worksheet, '1:1', header_format)

        # Freeze the header row(s) - for summary sheet, this will be overridden later with rows=3
        if not is_summary:
            retry_with_backoff(set_frozen, worksheet, rows=1)
        # For summary sheets, we'll freeze rows in the enhance_summary_sheet function

        # Apply alternating row colors for better readability
        even_row_format = CellFormat(
            backgroundColor=Color(0.95, 0.98, 0.9),  # Very light green
            borders={
                "bottom": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "left": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "right": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)}
            }
        )

        odd_row_format = CellFormat(
            backgroundColor=Color(1, 1, 1),  # White
            borders={
                "bottom": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "left": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "right": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)}
            }
        )

        # Generate row ranges for even and odd rows
        even_rows = [f'{i}:{i}' for i in range(2, rows + 1) if i % 2 == 0]
        odd_rows = [f'{i}:{i}' for i in range(2, rows + 1) if i % 2 != 0]

        # Consolidate adjacent ranges to reduce API calls
        consolidated_even_rows = consolidate_ranges(even_rows)
        consolidated_odd_rows = consolidate_ranges(odd_rows)

        logger.info(f"Optimized formatting: reduced from {len(even_rows) + len(odd_rows)} to {len(consolidated_even_rows) + len(consolidated_odd_rows)} API calls")

        # Apply formatting to consolidated ranges
        for row_range in consolidated_even_rows:
            retry_with_backoff(format_cell_range, worksheet, row_range, even_row_format)

        for row_range in consolidated_odd_rows:
            retry_with_backoff(format_cell_range, worksheet, row_range, odd_row_format)

        logger.info("Applied alternating row formatting with optimized API calls")

        # Format flag columns if this is not a summary sheet
        if not is_summary:
            # Get all values to find flag columns
            try:
                all_values = worksheet.get_all_values()
                header_row = all_values[0] if all_values else []

                # Find all flag-related columns
                flag_col_indices = []
                for i, header in enumerate(header_row):
                    if header.lower() == 'flags' or 'flag' in header.lower():
                        flag_col_indices.append(i + 1)  # Convert to 1-based index

                if flag_col_indices:
                    # Format for flag columns
                    flag_format = CellFormat(
                        backgroundColor=Color(1.0, 0.95, 0.95),  # Light red
                        textFormat=TextFormat(bold=True, foregroundColor=Color(0.7, 0.0, 0.0)),  # Dark red text
                        horizontalAlignment='LEFT',
                        wrapStrategy='WRAP'  # Enable text wrapping for multiple flags
                    )

                    # Apply formatting to all flag columns in a single batch if possible
                    for flag_col_index in flag_col_indices:
                        flag_col_letter = chr(64 + flag_col_index)
                        flag_range = f"{flag_col_letter}2:{flag_col_letter}{rows}"
                        retry_with_backoff(format_cell_range, worksheet, flag_range, flag_format)

                    logger.info(f"Formatted {len(flag_col_indices)} flag columns with optimized formatting")
            except Exception as e:
                logger.error(f"Error formatting flag columns: {e}")

        # Auto-resize columns in batches for better performance
        try:
            # Group columns into batches of 5 for auto-resizing
            batch_size = 5
            for i in range(0, cols, batch_size):
                end_col = min(i + batch_size - 1, cols - 1)
                retry_with_backoff(worksheet.columns_auto_resize, i, end_col)
                logger.info(f"Auto-resized columns {i} to {end_col}")
        except Exception as e:
            logger.error(f"Error auto-resizing columns: {e}")

        logger.info("Completed worksheet formatting with optimized API calls")

        # Add conditional formatting for the summary sheet
        if is_summary and worksheet.title == 'Summary':
            # Get the column index for flagged_percentage
            header_row = worksheet.row_values(1)
            try:
                # Find the flagged_percentage column index (not used directly but needed for the check)
                _ = header_row.index('flagged_percentage')

                # Add conditional formatting to highlight rows with flagged_percentage < 90% in red
                low_percentage_format = CellFormat(
                    backgroundColor=Color(1, 0.8, 0.8)  # Light red
                )

                # Create conditional formatting rule
                rules = get_conditional_format_rules(worksheet)

                # Clear existing rules
                rules.clear()

                # Add rule for rows with flagged_percentage < 90
                rule = ConditionalFormatRule(
                    ranges=[GridRange.from_a1_range(f'A2:{chr(64 + cols)}{rows}', worksheet)],
                    booleanRule=BooleanRule(
                        condition=BooleanCondition('NUMBER_LESS', ['90']),
                        format=low_percentage_format
                    )
                )

                rules.append(rule)
                rules.save()

                logger.info("Added conditional formatting for flagged percentage")
            except ValueError:
                logger.warning("Could not find flagged_percentage column in summary sheet")

        # Auto-resize columns
        worksheet.columns_auto_resize(0, worksheet.col_count - 1)

        logger.info(f"Applied enhanced formatting to sheet: {worksheet.title}")
    except Exception as e:
        logger.error(f"Error applying formatting to sheet {worksheet.title}: {e}")

def upload_csv_to_sheet(spreadsheet, csv_path, sheet_name=None):
    """
    Upload a CSV file to a sheet in the spreadsheet.

    Args:
        spreadsheet (gspread.Spreadsheet): The target spreadsheet
        csv_path (str): Path to the CSV file
        sheet_name (str, optional): Name for the sheet. If None, uses the CSV filename

    Returns:
        gspread.Worksheet: The created or updated worksheet
    """
    # Use the CSV filename if no sheet name is provided
    if sheet_name is None:
        sheet_name = os.path.basename(csv_path)

    # Sanitize the sheet name
    sheet_name = sanitize_sheet_name(sheet_name)

    # Check if this is the summary sheet
    is_summary = (sheet_name == 'Summary')

    try:
        # Read CSV data
        data = read_csv_file(csv_path)
        if not data:
            logger.warning(f"No data found in {csv_path}")
            return None

        # Check if the sheet already exists
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            # Clear existing content
            worksheet.clear()
            logger.info(f"Cleared existing sheet: {sheet_name}")
        except gspread.exceptions.WorksheetNotFound:
            # Create a new worksheet
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(data), cols=len(data[0]))
            logger.info(f"Created new sheet: {sheet_name}")

        # Upload data
        worksheet.update(data)
        logger.info(f"Uploaded data to sheet: {sheet_name}")

        # Apply formatting
        format_sheet(spreadsheet, worksheet, is_summary)

        # Add tag-based system for flags column if it exists
        if not is_summary:
            header_row = worksheet.row_values(1)
            try:
                flags_col_index = header_row.index('flags') + 1
                flags_col_letter = chr(64 + flags_col_index)  # Convert to column letter (A, B, C, etc.)

                # Format the flags column to make tags more visible
                flags_format = CellFormat(
                    textFormat=TextFormat(bold=True),
                    horizontalAlignment='LEFT',
                    wrapStrategy='WRAP'  # Enable text wrapping for multiple tags
                )

                # Apply formatting to the flags column (excluding header)
                retry_with_backoff(
                    format_cell_range,
                    worksheet,
                    f'{flags_col_letter}2:{flags_col_letter}{worksheet.row_count}',
                    flags_format
                )

                # Add a note to the header cell explaining the tag system
                header_note = (
                    "Multiple flags can be added to each cell, separated by commas.\n\n"
                    "# TOUCH DATA FLAG DEFINITIONS\n\n"
                    "## SEQUENCE COMPLETENESS FLAGS\n"
                    "- missing_Began: The sequence is missing its starting point. Example: Touch began outside the active sensing area.\n"
                    "- missing_Ended: The sequence never properly ended. Example: User lifted their finger outside the active sensing area.\n"
                    "- missing_B: First event in Tracing sequence is not 'B'. Example: Touch began outside the active sensing area.\n"
                    "- missing_E: Tracing sequence has no 'E' event. Example: User lifted their finger outside the active sensing area.\n"
                    "- multiple_end_events: Sequence has more than one ending event. Example: Touch sensor incorrectly detected finger lift and then redetected the same touch.\n"
                    "- sequence_interrupted: A new sequence started before the current one ended. Example: Hardware issue causing 'phantom' begin events.\n\n"

                    "## SEQUENCE QUALITY FLAGS\n"
                    "- short_duration: Touch sequence was extremely brief (less than 10ms). Example: User accidentally brushed against the screen.\n"
                    "- too_few_points: Sequence has fewer than 3 touch events. Example: Quick tap rather than a drag or swipe.\n"
                    "- sequence_gap/time_gap: Significant time gap (>100ms) between events. Example: Application experienced performance issues or frame drops.\n"
                    "- improper_sequence_order: Touch events don't follow the expected order. Example: Events processed out of order due to threading or timing issues.\n"
                    "- has_canceled: Sequence contains 'Canceled' events. Example: System canceled the touch sequence due to a gesture conflict.\n"
                    "- orphaned_events: Touch events occurred between sequences. Example: Touch sensor detected movement after finger was lifted.\n"
                    "- invalid_TouchPhase: Contains invalid touchPhase values. Example: Corrupted data or sensor malfunction.\n"
                    "- zero_distance: Sum of distance values equals 0 in sequence with >1 point. Example: Sensor reporting inconsistency.\n\n"

                    "## TRACING-SPECIFIC FLAGS\n"
                    "- PHANTOM_MOVE: TouchPhase is 'M' but coordinates don't change while distance increases. Example: Sensor reporting inconsistency.\n"
                    "- OVERLAPPING_FINGERIDS: Multiple fingers with overlapping IDs detected. Example: Touch sensor ID assignment issue.\n"
                    "- UNTERMINATED: Touch sequence was not properly terminated. Example: Hardware or software interruption.\n"
                    "- ORPHANED_FINGER: Finger events detected without proper sequence context. Example: Touch sensor detection anomaly.\n\n"

                    "## FLAG PRECEDENCE\n"
                    "When conflicts occur between flags, the system resolves them using a priority hierarchy. Higher priority flags take precedence over lower priority flags.\n"
                    "Example: If both 'missing_Began' and 'orphaned_events' are detected, only 'orphaned_events' is shown as it explains why a beginning might be missing."
                )

                # Add the note to the header cell
                try:
                    worksheet.update_note(f'{flags_col_letter}1', header_note)
                    logger.info(f"Added flags column note to {flags_col_letter}1")
                except Exception as e:
                    logger.warning(f"Could not add note to flags column header: {e}")

                # Make the flags column wider to accommodate multiple tags
                try:
                    retry_with_backoff(
                        worksheet.columns_auto_resize,
                        flags_col_index - 1,  # Convert to 0-based index
                        flags_col_index - 1
                    )
                    logger.info(f"Resized flags column {flags_col_letter}")
                except Exception as e:
                    logger.warning(f"Could not resize flags column: {e}")

                logger.info(f"Added tag-based system for flags column {flags_col_letter}")
            except ValueError:
                logger.info("No flags column found in this sheet")

        return worksheet
    except Exception as e:
        logger.error(f"Error uploading {csv_path} to sheet {sheet_name}: {e}")
        return None

def enhance_summary_sheet(_, worksheet):
    """
    Apply additional enhancements to the summary sheet to make it more visually appealing.

    Args:
        _ (gspread.Spreadsheet): The spreadsheet containing the worksheet (not used but kept for API consistency)
        worksheet (gspread.Worksheet): The summary worksheet to enhance
    """
    try:
        # Get the number of rows and columns
        rows = worksheet.row_count
        cols = worksheet.col_count

        # Add a title and metadata above the first column (filename column)
        # First, get the current header row to identify the filename column
        header_row = worksheet.row_values(1)

        # Get current date and time for metadata
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Insert a row for metadata (processing date/time)
        worksheet.insert_row(["Processed: " + current_time] + [""] * (len(header_row) - 1), 1)

        # Insert a row for the title
        worksheet.insert_row(["Data Processing Summary"] + [""] * (len(header_row) - 1), 1)

        # No need to merge cells - title will only appear in the first column

        # Format the title with tree-inspired colors
        title_format = CellFormat(
            backgroundColor=Color(0.76, 0.6, 0.42),  # Light brown
            textFormat=TextFormat(bold=True, foregroundColor=Color(0.13, 0.55, 0.13), fontSize=14),  # Forest green text, larger font
            horizontalAlignment='LEFT',  # Left-align for better readability
            verticalAlignment='MIDDLE',
            wrapStrategy='WRAP'  # Enable text wrapping
        )
        retry_with_backoff(format_cell_range, worksheet, 'A1:A1', title_format)

        # Format the metadata row with a subtle style
        metadata_format = CellFormat(
            backgroundColor=Color(0.95, 0.95, 0.95),  # Light gray
            textFormat=TextFormat(italic=True, foregroundColor=Color(0.4, 0.4, 0.4), fontSize=10),  # Gray text, smaller font
            horizontalAlignment='LEFT',
            verticalAlignment='MIDDLE'
        )
        retry_with_backoff(format_cell_range, worksheet, 'A2:A2', metadata_format)

        # Make the filename column wider to accommodate both the heading and filenames
        try:
            # Set a fixed width for the first column that's wide enough for both heading and filenames
            worksheet.set_column_width(0, 400)  # 400 pixels should be enough for most filenames
            logger.info("Adjusted width of filename column")
        except Exception as e:
            logger.warning(f"Could not adjust width of filename column: {e}")

        # Format the header row (now row 3) with tree-inspired colors
        header_format = CellFormat(
            backgroundColor=Color(0.76, 0.6, 0.42),  # Light brown
            textFormat=TextFormat(bold=True, foregroundColor=Color(0.13, 0.55, 0.13)),  # Forest green text
            horizontalAlignment='CENTER',
            borders={
                "bottom": {"style": "SOLID_MEDIUM"},
                "top": {"style": "SOLID_MEDIUM"},
                "left": {"style": "SOLID_MEDIUM"},
                "right": {"style": "SOLID_MEDIUM"}
            }
        )
        retry_with_backoff(format_cell_range, worksheet, '3:3', header_format)

        # Freeze the header rows (title, metadata, and actual header)
        retry_with_backoff(set_frozen, worksheet, rows=3)

        # Apply alternating row colors with tree-inspired theme
        even_row_format = CellFormat(
            backgroundColor=Color(0.95, 0.98, 0.9),  # Very light green
            borders={
                "bottom": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "left": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "right": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)}
            }
        )

        odd_row_format = CellFormat(
            backgroundColor=Color(1, 1, 1),  # White
            borders={
                "bottom": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "left": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)},
                "right": {"style": "SOLID", "color": Color(0.8, 0.8, 0.8)}
            }
        )

        # Apply to all data rows - use larger batches to reduce API calls
        batch_size = 25  # Increased batch size

        # Format even rows in batches
        even_rows = []
        for i in range(4, rows + 1):  # Start from row 4 (after title, metadata, and header)
            if i % 2 == 0 and i <= rows:
                even_rows.append(f'{i}:{i}')

        # Format odd rows in batches
        odd_rows = []
        for i in range(4, rows + 1):  # Start from row 4 (after title, metadata, and header)
            if i % 2 != 0 and i <= rows:
                odd_rows.append(f'{i}:{i}')

        # Apply formatting in batches
        for i in range(0, len(even_rows), batch_size):
            batch = even_rows[i:i+batch_size]
            if batch:
                for row_range in batch:
                    retry_with_backoff(format_cell_range, worksheet, row_range, even_row_format)

        for i in range(0, len(odd_rows), batch_size):
            batch = odd_rows[i:i+batch_size]
            if batch:
                for row_range in batch:
                    retry_with_backoff(format_cell_range, worksheet, row_range, odd_row_format)

        # No need for delay between batches as we're already using retry with backoff

        # Add conditional formatting for flagged_percentage column
        header_row = worksheet.row_values(3)  # Header is now in row 3
        try:
            flagged_percentage_col = header_row.index('flagged_percentage') + 1
            flagged_percentage_col_letter = chr(64 + flagged_percentage_col)

            # Add conditional formatting to highlight rows with flagged_percentage < 90% in red
            rules = retry_with_backoff(get_conditional_format_rules, worksheet)
            rules.clear()

            # Rule for rows with high flagged percentage (greater than 10%)
            # Use SEARCH function to check if the value contains "%" and is greater than 10
            high_percentage_rule = ConditionalFormatRule(
                ranges=[GridRange.from_a1_range(f'A4:{chr(64 + cols)}{rows}', worksheet)],  # Start from row 4 (data rows)
                booleanRule=BooleanRule(
                    condition=BooleanCondition('CUSTOM_FORMULA',
                        [f'=AND(SEARCH("%",${flagged_percentage_col_letter}4)>0,VALUE(LEFT(${flagged_percentage_col_letter}4,SEARCH("%",${flagged_percentage_col_letter}4)-1))>10)']),
                    format=CellFormat(backgroundColor=Color(1, 0.8, 0.8))  # Light red
                )
            )

            # Rule for rows with low flagged percentage (10% or less)
            low_percentage_rule = ConditionalFormatRule(
                ranges=[GridRange.from_a1_range(f'A4:{chr(64 + cols)}{rows}', worksheet)],  # Start from row 4 (data rows)
                booleanRule=BooleanRule(
                    condition=BooleanCondition('CUSTOM_FORMULA',
                        [f'=AND(SEARCH("%",${flagged_percentage_col_letter}4)>0,VALUE(LEFT(${flagged_percentage_col_letter}4,SEARCH("%",${flagged_percentage_col_letter}4)-1))<=10)']),
                    format=CellFormat(backgroundColor=Color(0.9, 1, 0.9))  # Light green
                )
            )

            rules.append(low_percentage_rule)
            rules.append(high_percentage_rule)
            retry_with_backoff(lambda r: r.save(), rules)

            # Format the percentage column to show as percentage with 2 decimal places
            percentage_format = CellFormat(
                numberFormat={'type': 'PERCENT', 'pattern': '0.00%'}
            )

            # Apply percentage formatting to the flagged_percentage column (excluding header)
            retry_with_backoff(
                format_cell_range,
                worksheet,
                f'{flagged_percentage_col_letter}4:{flagged_percentage_col_letter}{rows}',  # Start from row 4 (data rows)
                percentage_format
            )

            # Get the current values
            percentage_values = retry_with_backoff(
                worksheet.get,
                f'{flagged_percentage_col_letter}4:{flagged_percentage_col_letter}{rows}'  # Start from row 4 (data rows)
            )

            # Process the values to ensure they display correctly as percentages
            for i, row in enumerate(percentage_values):
                if row and row[0]:
                    try:
                        # Check if it's already a percentage string
                        if '%' in str(row[0]):
                            # Extract the numeric part and convert to decimal for percentage formatting
                            numeric_part = str(row[0]).replace('%', '').strip()
                            value = float(numeric_part) / 100
                            percentage_values[i] = [value]
                        else:
                            # If no percent sign, try to convert to float and divide by 100
                            value = float(row[0]) / 100
                            percentage_values[i] = [value]
                    except ValueError:
                        # If conversion fails, ensure it has a percent sign
                        if not str(row[0]).endswith('%'):
                            percentage_values[i] = [f"{row[0]}%"]
                    except Exception:
                        # If all else fails, leave it as is
                        pass

            # Update the cells with the converted values
            if percentage_values:
                retry_with_backoff(
                    worksheet.update,
                    f'{flagged_percentage_col_letter}4:{flagged_percentage_col_letter}{rows}',  # Start from row 4 (data rows)
                    percentage_values
                )

            # Make the percentage column wider to accommodate the percent sign
            try:
                # Set a fixed width for the percentage column
                worksheet.set_column_width(flagged_percentage_col - 1, 150)  # 150 pixels for percentage column
                logger.info(f"Adjusted width of percentage column {flagged_percentage_col_letter}")
            except Exception as e:
                logger.warning(f"Could not adjust width of percentage column: {e}")

            logger.info("Enhanced summary sheet with conditional formatting and percentage formatting")
        except ValueError:
            logger.warning("Could not find flagged_percentage column in summary sheet")

        # Auto-resize columns with retry
        try:
            retry_with_backoff(worksheet.columns_auto_resize, 0, worksheet.col_count - 1)
        except Exception as e:
            logger.warning(f"Could not auto-resize columns: {e}")

    except Exception as e:
        logger.error(f"Error enhancing summary sheet: {e}")

def create_documentation_sheet(spreadsheet):
    """
    Create a documentation sheet from README.md content.

    This function reads the README.md file and converts its content to a format
    suitable for Google Sheets, preserving the markdown structure while making
    it readable in a spreadsheet format.

    Args:
        spreadsheet: The Google Spreadsheet to add the documentation to

    Returns:
        The created worksheet or None if there was an error
    """
    try:
        # Check if README.md exists
        if not os.path.exists('README.md'):
            logger.warning("README.md not found, skipping documentation sheet")
            return None

        # Read README.md content
        with open('README.md', 'r') as f:
            readme_content = f.read()

        # Process the content into rows for the sheet
        rows = []

        # Add title row
        rows.append(["Touch Data Analysis Tool Documentation"])
        rows.append([])  # Empty row for spacing

        # Process the content line by line
        current_section = ""
        in_code_block = False
        code_block_content = []

        for line in readme_content.split('\n'):
            # Handle headings
            if line.startswith('# '):
                rows.append([line[2:]])
                rows.append([])  # Empty row for spacing
            elif line.startswith('## '):
                current_section = line[3:]
                rows.append([current_section])
                rows.append([])  # Empty row for spacing
            elif line.startswith('### '):
                rows.append([line[4:]])
            elif line.startswith('#### '):
                rows.append([line[5:]])
            # Handle code blocks
            elif line.startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    # Add the code block content
                    for code_line in code_block_content:
                        rows.append([code_line])
                    code_block_content = []
                    rows.append([])  # Empty row after code block
                else:
                    # Start of code block
                    in_code_block = True
                    rows.append(["Code Example:"])
            elif in_code_block:
                # Inside code block
                code_block_content.append(line)
            # Handle list items
            elif line.strip().startswith('- '):
                rows.append(["• " + line.strip()[2:]])
            elif line.strip().startswith('1. ') or line.strip().startswith('2. ') or line.strip().startswith('3. '):
                # Numbered list item
                rows.append([line.strip()])
            # Handle bold text
            elif '**' in line:
                # Replace markdown bold with actual bold (will be formatted later)
                line = line.replace('**', '')
                rows.append([line])
            # Handle regular text
            elif line.strip():
                rows.append([line])
            # Handle empty lines
            elif not in_code_block:
                rows.append([])

        # Create a new sheet for documentation
        try:
            # Check if Documentation sheet already exists
            try:
                doc_sheet = spreadsheet.worksheet('Documentation')
                # Clear existing content
                doc_sheet.clear()
                logger.info("Cleared existing Documentation sheet")
            except gspread.exceptions.WorksheetNotFound:
                # Create a new worksheet
                doc_sheet = spreadsheet.add_worksheet(title='Documentation', rows=len(rows), cols=1)
                logger.info("Created new Documentation sheet")

            # Update the sheet with the content
            doc_sheet.update('A1', rows)

            # Format the documentation sheet
            format_documentation_sheet(doc_sheet)

            logger.info("Successfully created Documentation sheet")
            return doc_sheet

        except Exception as e:
            logger.error(f"Error creating Documentation sheet: {e}")
            return None

    except Exception as e:
        logger.error(f"Error processing README.md: {e}")
        return None

def format_documentation_sheet(worksheet):
    """
    Apply formatting to the documentation sheet to make it readable.

    Args:
        worksheet: The documentation worksheet to format
    """
    try:
        # Format the title (first row)
        title_format = CellFormat(
            textFormat=TextFormat(bold=True, fontSize=16),
            horizontalAlignment='CENTER'
        )
        retry_with_backoff(format_cell_range, worksheet, 'A1:A1', title_format)

        # Format section headings (rows that contain only one word or short phrase)
        all_values = worksheet.get_all_values()

        # Format headings
        heading_format = CellFormat(
            backgroundColor=Color(0.76, 0.6, 0.42),  # Light brown
            textFormat=TextFormat(bold=True, fontSize=14, foregroundColor=Color(1, 1, 1))  # White text
        )

        subheading_format = CellFormat(
            textFormat=TextFormat(bold=True, fontSize=12),
            backgroundColor=Color(0.85, 0.75, 0.6)  # Lighter brown
        )

        code_format = CellFormat(
            textFormat=TextFormat(fontFamily="Courier New"),
            backgroundColor=Color(0.95, 0.98, 0.95)  # Very light green background
        )

        # Apply formatting based on content
        for i, row in enumerate(all_values):
            if not row or not row[0]:
                continue

            cell_range = f'A{i+1}:A{i+1}'

            # Format main headings (rows that start with a section name)
            if row[0] in ["Overview", "Data Processing Pipeline", "Sequence Segmentation", "Flag Rules"]:
                retry_with_backoff(format_cell_range, worksheet, cell_range, heading_format)

            # Format subheadings
            elif row[0].startswith("1. ") or row[0].startswith("2. ") or row[0].startswith("3. ") or row[0].startswith("4. ") or row[0].startswith("5. "):
                retry_with_backoff(format_cell_range, worksheet, cell_range, subheading_format)

            # Format code examples
            elif row[0] == "Code Example:":
                retry_with_backoff(format_cell_range, worksheet, cell_range, code_format)

                # Format the next lines as code until an empty line
                code_start = i + 1
                code_end = code_start
                while code_end < len(all_values) and all_values[code_end] and all_values[code_end][0]:
                    code_end += 1

                if code_end > code_start:
                    retry_with_backoff(format_cell_range, worksheet, f'A{code_start+1}:A{code_end}', code_format)

        # Auto-resize column
        worksheet.columns_auto_resize(0, 0)

        logger.info("Applied formatting to Documentation sheet")
    except Exception as e:
        logger.error(f"Error formatting Documentation sheet: {e}")

def upload_all_csvs(flagged_data_folder, spreadsheet_name=None, email=None):
    """
    Upload all CSV files from the flagged_data folder to a Google Spreadsheet.

    Args:
        flagged_data_folder (str): Path to the folder containing CSV files
        spreadsheet_name (str, optional): Name for the spreadsheet. If None, uses a default name
        email (str, optional): Email address to share the spreadsheet with.
                              If None, defaults to "jbanmol9@gmail.com" with edit access
    """
    # Use default spreadsheet name if none provided
    if not spreadsheet_name:
        spreadsheet_name = f"Flagged Data Export - {time.strftime('%Y-%m-%d')}"

    # Use default email if none provided
    if not email:
        email = "jbanmol9@gmail.com"
        logger.info(f"No email provided, using default email: {email}")

    # Authenticate with Google APIs
    credentials_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'credentials.json')
    client = authenticate_google_apis(credentials_path)

    # Create a new spreadsheet
    spreadsheet = create_spreadsheet(client, spreadsheet_name)

    # Share the spreadsheet with the specified email (default or provided)
    share_spreadsheet(spreadsheet, email)

    # Get list of CSV files
    try:
        csv_files = [f for f in os.listdir(flagged_data_folder) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {flagged_data_folder}")
    except Exception as e:
        logger.error(f"Failed to read files from {flagged_data_folder}: {e}")
        return

    if not csv_files:
        logger.warning(f"No CSV files found in {flagged_data_folder}")
        return

    # First, check if summary.csv exists and upload it first
    summary_path = os.path.join(flagged_data_folder, 'summary.csv')
    if os.path.exists(summary_path):
        # Upload summary.csv to the first sheet (rename the default Sheet1)
        default_sheet = spreadsheet.sheet1
        default_sheet.update_title('Summary')
        summary_sheet = upload_csv_to_sheet(spreadsheet, summary_path, 'Summary')

        # Apply additional enhancements to the summary sheet
        if summary_sheet:
            enhance_summary_sheet(spreadsheet, summary_sheet)

        # Remove summary.csv from the list to avoid duplication
        if 'summary.csv' in csv_files:
            csv_files.remove('summary.csv')

    # Add documentation sheet after summary sheet
    create_documentation_sheet(spreadsheet)

    # Upload all other CSV files with increased delay between uploads
    for i, csv_file in enumerate(csv_files):
        csv_path = os.path.join(flagged_data_folder, csv_file)
        logger.info(f"Uploading file {i+1} of {len(csv_files)}: {csv_file}")

        # Try to upload the file
        result = upload_csv_to_sheet(spreadsheet, csv_path)

        # If successful, log it
        if result:
            logger.info(f"Successfully uploaded {csv_file}")

        # Sleep briefly between uploads to avoid rate limiting
        # Use a minimal fixed delay to maintain API stability
        delay = 0.3  # Fixed minimal delay
        logger.info(f"Waiting {delay:.1f} seconds before next upload...")
        time.sleep(delay)

    logger.info(f"Successfully uploaded {len(csv_files) + ('summary.csv' in os.listdir(flagged_data_folder))} CSV files to {spreadsheet_name}")
    logger.info(f"Spreadsheet URL: {spreadsheet.url}")

def main():
    """Main function to run the uploader."""
    logger.info("Starting Google Sheets uploader")

    # Get spreadsheet name and email from command line if provided
    spreadsheet_name = None
    email = None

    if len(sys.argv) > 1:
        spreadsheet_name = sys.argv[1]

    if len(sys.argv) > 2:
        email = sys.argv[2]

    # Upload all CSV files from the flagged_data folder
    upload_all_csvs('flagged_data', spreadsheet_name, email)

    logger.info("Upload completed")

if __name__ == "__main__":
    main()
