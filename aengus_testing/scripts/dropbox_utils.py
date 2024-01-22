import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError

# DROPBOX_ACCESS_TOKEN = 'xxxxxxxxxxxxx'

# def dropbox_connect():
#     """Create a connection to Dropbox."""

#     try:
#         dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
#     except AuthError as e:
#         print('Error connecting to Dropbox with access token: ' + str(e))
#     return dbx


# def dropbox_list_files(path):
#     """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
#     """

#     dbx = dropbox_connect()

#     try:
#         files = dbx.files_list_folder(path).entries
#         files_list = []
#         for file in files:
#             if isinstance(file, dropbox.files.FileMetadata):
#                 metadata = {
#                     'name': file.name,
#                     'path_display': file.path_display,
#                     'client_modified': file.client_modified,
#                     'server_modified': file.server_modified
#                 }
#                 files_list.append(metadata)

#         df = pd.DataFrame.from_records(files_list)
#         return df.sort_values(by='server_modified', ascending=False)

#     except Exception as e:
#         print('Error getting list of files from Dropbox: ' + str(e))


# def dropbox_download_file(dropbox_file_path, local_file_path):
#     """Download a file from Dropbox to the local machine."""

#     try:
#         dbx = dropbox_connect()

#         with open(local_file_path, 'wb') as f:
#             metadata, result = dbx.files_download(path=dropbox_file_path)
#             f.write(result.content)
#     except Exception as e:
#         print('Error downloading file from Dropbox: ' + str(e))


# def dropbox_upload_file(local_path, local_file, dropbox_file_path):
#     """Upload a file from the local machine to a path in the Dropbox app directory.

#     Args:
#         local_path (str): The path to the local file.
#         local_file (str): The name of the local file.
#         dropbox_file_path (str): The path to the file in the Dropbox app directory.

#     Example:
#         dropbox_upload_file('.', 'test.csv', '/stuff/test.csv')

#     Returns:
#         meta: The Dropbox file metadata.
#     """

#     try:
#         dbx = dropbox_connect()

#         local_file_path = pathlib.Path(local_path) / local_file

#         with local_file_path.open("rb") as f:
#             meta = dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

#             return meta
#     except Exception as e:
#         print('Error uploading file to Dropbox: ' + str(e))


# def dropbox_get_link(dropbox_file_path):
#     """Get a shared link for a Dropbox file path.

#     Args:
#         dropbox_file_path (str): The path to the file in the Dropbox app directory.

#     Returns:
#         link: The shared link.
#     """

#     try:
#         dbx = dropbox_connect()
#         shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_file_path)
#         shared_link = shared_link_metadata.url
#         return shared_link.replace('?dl=0', '?dl=1')
#     except dropbox.exceptions.ApiError as exception:
#         if exception.error.is_shared_link_already_exists():
#             shared_link_metadata = dbx.sharing_get_shared_links(dropbox_file_path)
#             shared_link = shared_link_metadata.links[0].url
#             return shared_link.replace('?dl=0', '?dl=1')


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

# Example Usage

dbx = initialize_dropbox_client(ACCESS_TOKEN)

# Replace with your file paths
upload_file(dbx, 'scripts/finetuning.py', '/finetuning.py')
download_file(dbx, '/finetuning.py', './test123.py')
list_files_in_folder(dbx, '')
