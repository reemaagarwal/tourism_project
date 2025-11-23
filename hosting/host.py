import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Hugging Face Space repo
SPACE_REPO_ID = "Reemaagarwal/visit-with-us-tourism-space"  
SPACE_REPO_TYPE = "space"

LOCAL_DEPLOYMENT_FOLDER = "deployment"


def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")
    return token


def main():
    token = get_hf_token()
    api = HfApi(token=token)

    # 1. Ensure Space exists
    try:
        api.repo_info(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE)
        print(f"Space '{SPACE_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{SPACE_REPO_ID}' not found. Creating new Space...")
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type=SPACE_REPO_TYPE,
            space_sdk="streamlit", 
            private=False,
            token=token,
        )
        print(f"Space '{SPACE_REPO_ID}' created.")

    # 2. Check local deployment folder
    print("Current working directory:", os.getcwd())
    print("Deployment folder exists?", os.path.isdir(LOCAL_DEPLOYMENT_FOLDER))
    if not os.path.isdir(LOCAL_DEPLOYMENT_FOLDER):
        raise FileNotFoundError(f"Deployment folder not found: {LOCAL_DEPLOYMENT_FOLDER}")

    # 3. Upload all files in deployment folder to the Space
    print(f"Uploading '{LOCAL_DEPLOYMENT_FOLDER}' to Space '{SPACE_REPO_ID}'...")
    api.upload_folder(
        folder_path=LOCAL_DEPLOYMENT_FOLDER,
        repo_id=SPACE_REPO_ID,
        repo_type=SPACE_REPO_TYPE,
        path_in_repo="",   # root of Space
        token=token,
    )

    print("✅ Deployment files uploaded successfully to Hugging Face Space.")


if __name__ == "__main__":
    main()
