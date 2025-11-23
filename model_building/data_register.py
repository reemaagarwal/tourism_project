import os
import sys
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Hugging Face dataset repo
repo_id = "Reemaagarwal/visit-with-us-tourism"
repo_type = "dataset"

# github folder to upload
local_data_folder = "data"

# Get token from environment
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Initialize HF API
api = HfApi(token=token)

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload local data folder (contains tourism.csv)
if not os.path.isdir(local_data_folder):
    print(f"Local data folder does not exist: {local_data_folder}")
    sys.exit(1)

print(f"Uploading '{local_data_folder}' to dataset repo '{repo_id}'...")
api.upload_folder(
    folder_path=local_data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",  # upload to root of dataset repo
    token=token
)

print("Data registration completed successfully.")
