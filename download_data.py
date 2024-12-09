from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io
import os
import zipfile

'''
Change the following variables to match your needs:
- SERVICE_ACCOUNT_FILE: path to the service account file
- SCOPES: list of scopes needed for the Google Drive API
- FILE_ID: Google Drive ID of the file to download
- DATA_ROOT: path to the directory where the downloaded file will be saved
- FILE_NAME: how you want the downloaded file to be named
Alternatively, you can download a zip file from the following link:
https://drive.google.com/file/d/1tZFHkD0ohgFZjlp-CYNW4GYkA3Dvmx1P/view?usp=drive_link
'''

SERVICE_ACCOUNT_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
FILE_ID = '1tZFHkD0ohgFZjlp-CYNW4GYkA3Dvmx1P'
DATA_ROOT = './raw_data'
FILE_NAME = 'compressed.zip'

def download_all() -> None:
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')

    download_path = os.path.join(DATA_ROOT, FILE_NAME)

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    request = service.files().get_media(fileId=FILE_ID)

    with io.FileIO(download_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

    print(f"File '{FILE_NAME}' downloaded successfully!")

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall('raw_data')

    print(f"File '{FILE_NAME}' extracted successfully!")
    os.remove(download_path)

if __name__ == "__main__":
   download_all()