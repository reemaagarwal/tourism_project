# this is for github script, above cells are for local run
import os
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# CONFIG
# -----------------------------
DATASET_REPO_ID = "Reemaagarwal/visit-with-us-tourism"      # same as data_register.py / prep.py
DATASET_REPO_TYPE = "dataset"

MODEL_REPO_ID = "Reemaagarwal/visit-with-us-tourism-model"  # model repo on HF
MODEL_REPO_TYPE = "model"

# HF dataset file paths (created by prep.py and uploaded to dataset repo)
XTRAIN_PATH = f"hf://datasets/{DATASET_REPO_ID}/Xtrain.csv"
XTEST_PATH  = f"hf://datasets/{DATASET_REPO_ID}/Xtest.csv"
YTRAIN_PATH = f"hf://datasets/{DATASET_REPO_ID}/ytrain.csv"
YTEST_PATH  = f"hf://datasets/{DATASET_REPO_ID}/ytest.csv"

TARGET_COL = "ProdTaken"  # only for reference; ytrain/ytest already extracted as this column


def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")
    return token


def main():
    # -----------------------------
    # 1. Load train/test splits from HF dataset repo
    # -----------------------------
    print("📥 Loading train/test splits from Hugging Face dataset repo...")
    print("Xtrain path:", XTRAIN_PATH)
    print("Xtest path :", XTEST_PATH)
    print("ytrain path:", YTRAIN_PATH)
    print("ytest path :", YTEST_PATH)

    X_train = pd.read_csv(XTRAIN_PATH)
    X_test  = pd.read_csv(XTEST_PATH)
    y_train = pd.read_csv(YTRAIN_PATH)
    y_test  = pd.read_csv(YTEST_PATH)

    # y_* are single-column dataframes; convert to 1D Series
    if y_train.shape[1] != 1 or y_test.shape[1] != 1:
        print("❌ ytrain/ytest should have exactly 1 column (the target).")
        sys.exit(1)

    y_train = y_train.iloc[:, 0]
    y_test  = y_test.iloc[:, 0]

    print("Shapes:")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_test :", y_test.shape)

    # -----------------------------
    # 2. Define preprocessing and model (Random Forest)
    # -----------------------------
    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_model),
        ]
    )

    # -----------------------------
    # 3. Hyperparameter tuning with GridSearchCV
    # -----------------------------
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    print("🔍 Running GridSearchCV...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("🏆 Best params:", grid_search.best_params_)
    print("📊 Best CV accuracy:", grid_search.best_score_)

    # -----------------------------
    # 4. Log tuned parameters (simple experiment tracking)
    # -----------------------------
    from pathlib import Path

    results_dir = Path("tourism_project/model_building")
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    log_columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    experiment_log = cv_results[log_columns].sort_values("rank_test_score")

    log_path = results_dir / "experiment_results.csv"
    experiment_log.to_csv(log_path, index=False)
    print(f"📝 Saved experiment results to: {log_path}")

    # -----------------------------
    # 5. Evaluate on test set
    # -----------------------------
    print("📈 Evaluating best model on test set...")
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # -----------------------------
    # 6. Save best model locally
    # -----------------------------
    model_path = results_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"💾 Best model saved at: {model_path}")

    # -----------------------------
    # 7. Upload best model to Hugging Face Model Hub
    # -----------------------------
    hf_token = get_hf_token()
    api = HfApi(token=hf_token)

    # Ensure model repo exists
    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type=MODEL_REPO_TYPE)
        print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO_ID}' not found. Creating new model repo...")
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type=MODEL_REPO_TYPE,
            private=False,
            token=hf_token,
        )
        print(f"Model repo '{MODEL_REPO_ID}' created.")

    print(f"🔼 Uploading model file to HF model repo: {MODEL_REPO_ID}")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type=MODEL_REPO_TYPE,
        token=hf_token,
    )

    print(f"✅ Best model uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
