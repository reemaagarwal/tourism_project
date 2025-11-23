# this cell is to create prep file (above cells were for local running only
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Hugging Face dataset repo details (same as data_register.py)
repo_id = "Reemaagarwal/visit-with-us-tourism"
repo_type = "dataset"

# Get HF token from environment (CI + local if exported)
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN is missing. CI runner is not receiving the token.")

# Initialize API client
api = HfApi(token=token)

# ---- 1. Load dataset directly from Hugging Face data space ----
# This uses the same hf://datasets/... convention as the trainer code
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from:", DATASET_PATH)

# ---- 2. Data cleaning: drop unnecessary columns ----
# Unnamed: 0 is a junk index column; CustomerID is an identifier
for col in ["Unnamed: 0", "CustomerID"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped column: {col}")

target_col = "ProdTaken"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataframe.")

# ---- 3. Split into X (features) and y (target) ----
X = df.drop(columns=[target_col])
y = df[target_col]

# ---- 4. Perform train-test split (stratified) ----
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # keeps class balance of ProdTaken
)

print("Train shapes:", Xtrain.shape, ytrain.shape)
print("Test shapes :", Xtest.shape, ytest.shape)

# ---- 5. Save splits as CSVs (same pattern as trainer) ----
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally.")

# ---- 6. Upload split files back to the same HF dataset repo ----
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # upload just filename at root
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    print(f"Uploaded {file_path} to dataset repo: {repo_id}")

print("Data preparation (train/test splits) uploaded to Hugging Face dataset space.")
