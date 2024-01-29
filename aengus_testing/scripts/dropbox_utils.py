import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError


import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()
ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')

# Initialize Dropbox Client
def initialize_dropbox_client(access_token):
    return dropbox.Dropbox(access_token)

# Upload File to Dropbox
def upload_file(dbx, local_file_path, dropbox_path):
    try:
        with open(local_file_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_path, mode=WriteMode('overwrite'))
        print(f"File uploaded successfully to {dropbox_path}")
    except ApiError as e:
        print(f"API Error: {e}")
    except FileNotFoundError:
        print("Local file not found")

# Download File from Dropbox
def download_file(dbx, dropbox_file_path, local_path):
    try:
        dbx.files_download_to_file(local_path, dropbox_file_path)
        print(f"File downloaded successfully to {local_path}")
    except ApiError as e:
        print(f"API Error: {e}")

# List Files in Dropbox Folder
def list_files_in_folder(dbx, folder_path):
    try:
        response = dbx.files_list_folder(folder_path)
        for file in response.entries:
            print(file.name)
    except ApiError as e:
        print(f"API Error: {e}")







# ----------------------------------------------------------------
# Example Usage

dbx = initialize_dropbox_client(ACCESS_TOKEN) 

# Replace with your file paths
upload_file(dbx, 'scripts/finetuning.py', '/finetuning.py')
download_file(dbx, '/finetuning.py', './test123.py')
list_files_in_folder(dbx, '')
