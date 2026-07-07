# drive_integration.md — Google Drive Setup and Helper

## One-time setup steps the user must do manually before running any code

### Step 1 — Enable the Drive API

Go to console.cloud.google.com → create a project called "bfsi-sentiment" →
APIs and Services → Enable APIs → search "Google Drive API" → Enable.

### Step 2 — Create OAuth credentials

APIs and Services → Credentials → Create Credentials → OAuth client ID →
Application type: Desktop app → Name: bfsi-sentiment-slm → Create.

Download the JSON file → save it as `credentials.json` at the project root.

### Step 3 — First authentication run

Run `python setup_drive.py` — this opens a browser asking for Google account
permission. Click Allow. The script saves `token.json` locally. Every subsequent
run uses token.json silently without opening a browser.

## Files that must be in .gitignore

```
credentials.json
token.json
```

## Drive folder structure created automatically

```
Google Drive/
└── bfsi-sentiment-model/
    ├── training_data.csv
    ├── lora_adapters_adapter_config.json
    ├── lora_adapters_adapter_model.bin
    ├── sentiment_head.pt
    └── impact_head.pt
```

## drive_storage.py — complete implementation

```python
# drive_storage.py
import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES             = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_FOLDER_NAME  = "bfsi-sentiment-model"
CREDENTIALS_FILE   = "credentials.json"
TOKEN_FILE         = "token.json"


def get_drive_service():
    """Returns an authenticated Drive API service.
    Opens browser on first run, uses token.json silently thereafter."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"credentials.json not found at project root.\n"
                    f"Follow the steps in drive_integration.md to create it."
                )
            flow  = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service) -> str:
    """Gets the Drive folder ID, creating it if it doesn't exist."""
    query = (
        f"name='{DRIVE_FOLDER_NAME}' and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )
    results  = service.files().list(q=query, fields="files(id, name)").execute()
    existing = results.get("files", [])

    if existing:
        return existing[0]["id"]

    metadata  = {"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"}
    folder    = service.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    print(f"Created Drive folder: {DRIVE_FOLDER_NAME}")
    return folder_id


def upload_file(local_path: str, drive_filename: str):
    """Uploads or updates a file in the Drive folder."""
    if not os.path.exists(local_path):
        print(f"  [Drive] Skipping {local_path} — file does not exist")
        return

    service   = get_drive_service()
    folder_id = get_or_create_folder(service)

    query    = (f"name='{drive_filename}' and '{folder_id}' in parents "
                f"and trashed=false")
    results  = service.files().list(q=query, fields="files(id)").execute()
    existing = results.get("files", [])
    media    = MediaFileUpload(local_path, resumable=True)

    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        print(f"  [Drive] Updated: {drive_filename}")
    else:
        metadata = {"name": drive_filename, "parents": [folder_id]}
        service.files().create(body=metadata, media_body=media, fields="id").execute()
        print(f"  [Drive] Uploaded: {drive_filename}")


def download_file(drive_filename: str, local_path: str) -> bool:
    """Downloads a file from the Drive folder to a local path."""
    service   = get_drive_service()
    folder_id = get_or_create_folder(service)

    query   = (f"name='{drive_filename}' and '{folder_id}' in parents "
               f"and trashed=false")
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files   = results.get("files", [])

    if not files:
        print(f"  [Drive] Not found: {drive_filename}")
        return False

    os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else ".", exist_ok=True)
    request = service.files().get_media(fileId=files[0]["id"])

    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  [Drive] Downloading {drive_filename}: {int(status.progress()*100)}%")

    print(f"  [Drive] Saved to: {local_path}")
    return True


def upload_folder(local_folder: str, drive_prefix: str = ""):
    """Uploads all files in a local folder, flattening paths with underscores."""
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_path     = os.path.join(root, filename)
            relative       = os.path.relpath(local_path, local_folder)
            prefix         = f"{drive_prefix}_" if drive_prefix else ""
            drive_filename = (prefix + relative).replace("\\", "_").replace("/", "_")
            upload_file(local_path, drive_filename)


def upload_model_to_drive():
    """Uploads all trained model files to Drive. Called at end of training."""
    print("Uploading model to Google Drive...")
    upload_folder("output/lora_adapters", drive_prefix="lora_adapters")
    upload_file("output/sentiment_head.pt", "sentiment_head.pt")
    upload_file("output/impact_head.pt",    "impact_head.pt")
    print("Model upload complete")


def download_model_from_drive(output_dir: str = "output"):
    """Downloads all trained model files from Drive. Called before inference."""
    print("Downloading model from Google Drive...")
    os.makedirs(f"{output_dir}/lora_adapters", exist_ok=True)
    download_file("lora_adapters_adapter_config.json",
                  f"{output_dir}/lora_adapters/adapter_config.json")
    download_file("lora_adapters_adapter_model.bin",
                  f"{output_dir}/lora_adapters/adapter_model.bin")
    download_file("sentiment_head.pt", f"{output_dir}/sentiment_head.pt")
    download_file("impact_head.pt",    f"{output_dir}/impact_head.pt")
    print("Model download complete")


def list_drive_files():
    """Lists all files in the Drive folder."""
    service   = get_drive_service()
    folder_id = get_or_create_folder(service)
    results   = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(name, size)"
    ).execute()
    files = results.get("files", [])
    print(f"\nFiles in Google Drive '{DRIVE_FOLDER_NAME}':")
    for f in files:
        size_kb = int(f.get("size", 0)) // 1024
        print(f"  {f['name']} ({size_kb} KB)")
```

## setup_drive.py — one-time authentication script

```python
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
```
