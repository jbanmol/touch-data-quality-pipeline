#!/usr/bin/env python3
"""
Test script to verify that the default email sharing functionality works correctly.
This script tests that the upload_all_csvs function uses the default email when none is provided.
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the current directory to the path so we can import upload_to_sheets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_default_email_sharing():
    """Test that default email is used when none is provided."""
    print("Testing Default Email Sharing Functionality")
    print("=" * 50)

    # Mock the dependencies
    with patch('upload_to_sheets.authenticate_google_apis') as mock_auth, \
         patch('upload_to_sheets.create_spreadsheet') as mock_create, \
         patch('upload_to_sheets.share_spreadsheet') as mock_share, \
         patch('upload_to_sheets.os.listdir') as mock_listdir, \
         patch('upload_to_sheets.os.path.exists') as mock_exists, \
         patch('upload_to_sheets.upload_csv_to_sheet') as mock_upload, \
         patch('upload_to_sheets.create_documentation_sheet') as mock_doc:

        # Set up mocks
        mock_client = Mock()
        mock_auth.return_value = mock_client

        mock_spreadsheet = Mock()
        mock_spreadsheet.url = "https://docs.google.com/spreadsheets/test"
        mock_create.return_value = mock_spreadsheet

        mock_listdir.return_value = ['test_file.csv']
        mock_exists.return_value = False  # No summary.csv
        mock_upload.return_value = Mock()

        # Import the function after setting up mocks
        try:
            from upload_to_sheets import upload_all_csvs
            print("✓ Successfully imported upload_all_csvs function")
        except ImportError as e:
            print(f"✗ Error importing function: {e}")
            return False

        # Test 1: No email provided (should use default)
        print("\nTest 1: No email provided (should use default)")
        print("-" * 40)

        upload_all_csvs('test_folder')

        # Verify that share_spreadsheet was called with the default email
        mock_share.assert_called_once()
        call_args = mock_share.call_args
        shared_email = call_args[0][1]  # Second argument (email)

        if shared_email == "jbanmol9@gmail.com":
            print("✓ Default email 'jbanmol9@gmail.com' was used")
            test1_passed = True
        else:
            print(f"✗ Wrong email used: {shared_email}")
            test1_passed = False

        # Reset mocks for next test
        mock_share.reset_mock()

        # Test 2: Custom email provided (should use custom email)
        print("\nTest 2: Custom email provided (should use custom email)")
        print("-" * 40)

        custom_email = "custom@example.com"
        upload_all_csvs('test_folder', email=custom_email)

        # Verify that share_spreadsheet was called with the custom email
        mock_share.assert_called_once()
        call_args = mock_share.call_args
        shared_email = call_args[0][1]  # Second argument (email)

        if shared_email == custom_email:
            print(f"✓ Custom email '{custom_email}' was used")
            test2_passed = True
        else:
            print(f"✗ Wrong email used: {shared_email}, expected: {custom_email}")
            test2_passed = False

        # Reset mocks for next test
        mock_share.reset_mock()

        # Test 3: Empty string email provided (should use default)
        print("\nTest 3: Empty string email provided (should use default)")
        print("-" * 40)

        upload_all_csvs('test_folder', email="")

        # Verify that share_spreadsheet was called with the default email
        mock_share.assert_called_once()
        call_args = mock_share.call_args
        shared_email = call_args[0][1]  # Second argument (email)

        if shared_email == "jbanmol9@gmail.com":
            print("✓ Default email 'jbanmol9@gmail.com' was used for empty string")
            test3_passed = True
        else:
            print(f"✗ Wrong email used: {shared_email}")
            test3_passed = False

        return test1_passed and test2_passed and test3_passed

def test_share_spreadsheet_permissions():
    """Test that the share_spreadsheet function uses the correct permissions."""
    print("\nTesting Share Spreadsheet Permissions")
    print("=" * 40)

    # Mock the spreadsheet object
    mock_spreadsheet = Mock()

    try:
        from upload_to_sheets import share_spreadsheet
        print("✓ Successfully imported share_spreadsheet function")
    except ImportError as e:
        print(f"✗ Error importing function: {e}")
        return False

    # Test default role (should be 'writer')
    share_spreadsheet(mock_spreadsheet, "test@example.com")

    # Verify the share method was called with correct parameters
    mock_spreadsheet.share.assert_called_once_with(
        "test@example.com",
        perm_type='user',
        role='writer'
    )

    print("✓ Default role 'writer' is used for edit access")

    # Reset mock
    mock_spreadsheet.reset_mock()

    # Test custom role
    share_spreadsheet(mock_spreadsheet, "test@example.com", role='writer')

    # Verify the share method was called with custom role
    mock_spreadsheet.share.assert_called_once_with(
        "test@example.com",
        perm_type='user',
        role='writer'
    )

    print("✓ Custom role 'writer' can be specified")

    return True

def main():
    """Run all tests."""
    print("Testing Default Email Sharing and Permissions")
    print("=" * 60)

    email_test_passed = test_default_email_sharing()
    permissions_test_passed = test_share_spreadsheet_permissions()

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Default email sharing: {'PASSED' if email_test_passed else 'FAILED'}")
    print(f"Share permissions:     {'PASSED' if permissions_test_passed else 'FAILED'}")

    all_passed = email_test_passed and permissions_test_passed
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\n✅ Default email sharing functionality is working correctly")
        print("✅ Spreadsheets will be automatically shared with 'jbanmol9@gmail.com'")
        print("✅ Edit access permissions are properly configured")
    else:
        print("\n❌ Issues detected with default email sharing functionality")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
