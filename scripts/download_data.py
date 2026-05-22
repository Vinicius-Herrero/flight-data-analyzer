#!/usr/bin/env python3
"""
Download flight data from Google Drive.

This script downloads the flight datasets from a shared Google Drive folder
and extracts them to the database/ directory.
"""

import os
import gdown
from pathlib import Path

# Google Drive folder ID
FOLDER_ID = "1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj"
GOOGLE_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

# Expected data files
DATA_FILES = [
    "airlines.csv",
    "airports.csv",
    "flights.csv"
]

def create_database_folder():
    """Create the database folder if it doesn't exist."""
    db_path = Path("database")
    db_path.mkdir(exist_ok=True)
    return db_path

def download_files():
    """Download files from Google Drive folder."""
    db_path = create_database_folder()

    print("=" * 80)
    print("DOWNLOADING FLIGHT DATA FROM GOOGLE DRIVE")
    print("=" * 80)
    print(f"Folder URL: {GOOGLE_DRIVE_FOLDER_URL}")
    print()

    try:
        # Download folder from Google Drive
        gdown.download_folder(
            url=GOOGLE_DRIVE_FOLDER_URL,
            output=str(db_path),
            quiet=False,
            use_cookies=False
        )

        print()
        print("=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)

        # Verify downloaded files
        missing_files = []
        for file in DATA_FILES:
            file_path = db_path / file
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"✓ {file} ({file_size:.2f} MB)")
            else:
                missing_files.append(file)
                print(f"✗ {file} (NOT FOUND)")

        if missing_files:
            print(f"\n⚠️  Warning: Missing files: {', '.join(missing_files)}")
            return False

        print("\n✓ All data files successfully downloaded!")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading files: {e}")
        print(f"\nMake sure the Google Drive folder is accessible:")
        print(f"  {GOOGLE_DRIVE_FOLDER_URL}")
        return False

if __name__ == "__main__":
    success = download_files()
    exit(0 if success else 1)
