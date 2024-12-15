from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
import os

# Define the repository and file type to download
REPO_ID = "P-H-B-D-a16z/CommaVQbins"
FILE_EXTENSION = ".bin"
DOWNLOAD_DIR = "./nanogpt/"  # Specify download directory

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Function to download a single file
def download_file(file_name):
    try:
        # Directly download to DOWNLOAD_DIR without using cache_dir
        file_path = hf_hub_download(repo_id=REPO_ID, filename=file_name, local_dir=DOWNLOAD_DIR, repo_type="dataset", local_dir_use_symlinks=False)
        print(f"Downloaded {file_name} to {file_path}")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")

# List the specific files you want to download
bin_files = [f"{x}.bin" for x in range(41)]

# Download files in parallel
with ThreadPoolExecutor() as executor:
    executor.map(download_file, bin_files)

print("All downloads complete.")
