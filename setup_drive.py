# setup_drive.py
"""Run this once to authenticate with Google Drive.
After running, token.json is saved and all future scripts
authenticate silently without opening a browser."""

from drive_storage import get_drive_service, get_or_create_folder, list_drive_files

if __name__ == "__main__":
    print("Authenticating with Google Drive...")
    service   = get_drive_service()
    folder_id = get_or_create_folder(service)
    print(f"Authentication successful. Drive folder ID: {folder_id}")
    list_drive_files()
    print("\nSetup complete. You can now run train.py and predict.py.")
