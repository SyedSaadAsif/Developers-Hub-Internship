# =============================================================================
# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
# =============================================================================
# Objective:
#   Build a reusable, production-ready ML pipeline to predict customer churn
#   using the Telco Customer Churn dataset.
#
# Dataset:   Telco Customer Churn
#            https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#            (Place the downloaded CSV as 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
#             in the same directory, or adjust DATA_PATH below.)
#
# Pipeline:  Preprocessing → Model → GridSearchCV → Export with joblib
# Models:    Logistic Regression, Random Forest
#
# Requirements:
#   pip install scikit-learn pandas numpy joblib
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

DATA_PATH    = "WA_Fn-UseC_-Telco-Customer-Churn.csv"   # Path to dataset CSV
TARGET_COL   = "Churn"                                   # Binary target column
DROP_COLS    = ["customerID"]                            # Columns not useful for modelling
TEST_SIZE    = 0.2                                       # 80/20 train-test split
RANDOM_STATE = 42                                        # Reproducibility seed
CV_FOLDS     = 5                                         # Cross-validation folds for GridSearchCV
OUTPUT_MODEL = "churn_pipeline.joblib"                   # Path to export the final pipeline


# ---------------------------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load the Telco Churn CSV into a pandas DataFrame.

    Args:
        path : File path to the CSV

    Returns:
        df   : Raw DataFrame
    """
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Churn distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


# ---------------------------------------------------------------------------
# 3. Data Cleaning & Feature Engineering
# ---------------------------------------------------------------------------

def preprocess_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform initial cleaning steps before the sklearn Pipeline:
      - Drop unused ID columns
      - Fix known data-type issues (TotalCharges stored as str)
      - Encode the binary target column

    Args:
        df : Raw DataFrame

    Returns:
        X  : Feature matrix (DataFrame)
        y  : Binary target series (0 = No Churn, 1 = Churn)
    """
    df = df.copy()

    # Drop columns that carry no predictive value
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # TotalCharges is loaded as object due to whitespace-only entries → convert
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target: "Yes" → 1, "No" → 0
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y


# ---------------------------------------------------------------------------
# 4. Feature Type Detection
# ---------------------------------------------------------------------------

def detect_feature_types(X: pd.DataFrame) -> tuple[list, list]:
    """
    Automatically detect numeric and categorical feature columns.

    Args:
        X : Feature DataFrame

    Returns:
        num_cols : List of numeric column names
        cat_cols : List of categorical column names
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"  Numeric features    ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features({len(cat_cols)}): {cat_cols}\n")

    return num_cols, cat_cols


# ---------------------------------------------------------------------------
# 5. Preprocessing Sub-Pipeline (ColumnTransformer)
# ---------------------------------------------------------------------------

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies different preprocessing to
    numeric and categorical features using sklearn Pipeline objects.

    Numeric  : Impute (median) → Standard Scale
    Categorical: Impute (most_frequent) → One-Hot Encode

    Args:
        num_cols : List of numeric column names
        cat_cols : List of categorical column names

    Returns:
        preprocessor : ColumnTransformer
    """
    # --- Numeric pipeline ---
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # Fill NaN with median
        ("scaler",  StandardScaler()),                   # Zero-mean, unit-variance scaling
    ])

    # --- Categorical pipeline ---
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),   # Fill NaN with mode
        ("encoder", OneHotEncoder(handle_unknown="ignore",      # Ignore unseen categories
                                  drop="first")),               # Drop first level (avoid multicollinearity)
    ])

    # --- Combine into ColumnTransformer ---
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# 6. Full Model Pipelines
# ---------------------------------------------------------------------------

def build_pipelines(preprocessor: ColumnTransformer) -> dict:
    """
    Wrap the preprocessor with each candidate classifier in a single Pipeline.

    Args:
        preprocessor : ColumnTransformer built by build_preprocessor()

    Returns:
        pipelines : dict mapping model name → Pipeline
    """
    pipelines = {
        "logistic_regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier",   LogisticRegression(
                max_iter=1000,
                class_weight="balanced",   # Handle class imbalance
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier",   RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,   # Use all CPU cores
            )),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# 7. Hyperparameter Grids for GridSearchCV
# ---------------------------------------------------------------------------

# Note: parameter names follow sklearn Pipeline convention:
#   <step_name>__<param_name>

PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C":       [0.01, 0.1, 1, 10],       # Inverse regularisation strength
        "classifier__solver":  ["liblinear", "lbfgs"],    # Optimisation algorithm
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],           # Number of trees
        "classifier__max_depth":    [None, 10, 20],       # Maximum tree depth
        "classifier__min_samples_split": [2, 5],          # Min samples to split a node
    },
}


# ---------------------------------------------------------------------------
# 8. Training with GridSearchCV
# ---------------------------------------------------------------------------

def train_with_grid_search(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Run GridSearchCV for each pipeline to find the best hyperparameters.

    Args:
        pipelines : dict of model name → Pipeline
        X_train   : Training features
        y_train   : Training labels

    Returns:
        best_estimators : dict of model name → best fitted Pipeline
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_estimators = {}

    for name, pipeline in pipelines.items():
        print(f"Running GridSearchCV for: {name}")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRIDS[name],
            cv=cv,
            scoring="f1",        # Optimise for F1 (useful with imbalanced classes)
            n_jobs=-1,            # Parallelise across CPU cores
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        print(f"  Best params : {grid_search.best_params_}")
        print(f"  Best CV F1  : {grid_search.best_score_:.4f}\n")

        # Store the best pipeline (already refitted on full training data)
        best_estimators[name] = grid_search.best_estimator_

    return best_estimators


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    best_estimators: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate all best pipelines on the held-out test set.

    Args:
        best_estimators : dict of model name → fitted Pipeline
        X_test          : Test features
        y_test          : Test labels

    Returns:
        results : dict of model name → metrics dict
    """
    results = {}

    for name, estimator in best_estimators.items():
        preds      = estimator.predict(X_test)
        proba      = estimator.predict_proba(X_test)[:, 1]  # Probability of churn

        acc     = accuracy_score(y_test, preds)
        f1      = f1_score(y_test, preds, average="weighted")
        roc_auc = roc_auc_score(y_test, proba)

        results[name] = {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

        print(f"=== {name.upper()} ===")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    return results


# ---------------------------------------------------------------------------
# 10. Select Best Model & Export
# ---------------------------------------------------------------------------

def export_best_pipeline(
    best_estimators: dict,
    results: dict,
    output_path: str,
) -> None:
    """
    Select the model with the highest test F1-score and export it with joblib.

    Args:
        best_estimators : dict of model name → fitted Pipeline
        results         : dict of model name → metrics dict
        output_path     : File path to save the joblib file
    """
    # Pick winner by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = best_estimators[best_name]

    print(f"Best model: {best_name} (F1={results[best_name]['f1']:.4f})")
    print(f"Exporting pipeline to: {output_path}")

    joblib.dump(best_pipeline, output_path)
    print("Export complete.\n")


# ---------------------------------------------------------------------------
# 11. Inference Demo
# ---------------------------------------------------------------------------

def predict_churn(pipeline, sample: pd.DataFrame) -> None:
    """
    Demonstrate inference on a new customer record.

    Args:
        pipeline : Loaded joblib Pipeline
        sample   : Single-row DataFrame with the same columns as training data
    """
    pred  = pipeline.predict(sample)[0]
    proba = pipeline.predict_proba(sample)[0, 1]
    label = "CHURN" if pred == 1 else "NO CHURN"
    print(f"Prediction: {label} (churn probability: {proba:.2%})")


# ---------------------------------------------------------------------------
# 12. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # A) Load & clean raw data
    # ------------------------------------------------------------------
    df       = load_data(DATA_PATH)
    X, y     = preprocess_raw(df)

    # ------------------------------------------------------------------
    # B) Train / test split (stratified to preserve class balance)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows\n")

    # ------------------------------------------------------------------
    # C) Detect feature types and build preprocessor
    # ------------------------------------------------------------------
    num_cols, cat_cols = detect_feature_types(X_train)
    preprocessor       = build_preprocessor(num_cols, cat_cols)

    # ------------------------------------------------------------------
    # D) Build full pipelines (preprocessor + classifier)
    # ------------------------------------------------------------------
    pipelines = build_pipelines(preprocessor)

    # ------------------------------------------------------------------
    # E) Hyperparameter search with GridSearchCV
    # ------------------------------------------------------------------
    best_estimators = train_with_grid_search(pipelines, X_train, y_train)

    # ------------------------------------------------------------------
    # F) Evaluate on test set
    # ------------------------------------------------------------------
    results = evaluate_models(best_estimators, X_test, y_test)

    # ------------------------------------------------------------------
    # G) Export best pipeline with joblib
    # ------------------------------------------------------------------
    export_best_pipeline(best_estimators, results, OUTPUT_MODEL)

    # ------------------------------------------------------------------
    # H) Demo: reload exported pipeline and run inference
    # ------------------------------------------------------------------
    print("Loading exported pipeline for demo inference...")
    loaded_pipeline = joblib.load(OUTPUT_MODEL)

    # Use the first test row as a dummy new customer
    sample = X_test.iloc[[0]]
    print(f"Sample customer features:\n{sample.to_string()}\n")
    predict_churn(loaded_pipeline, sample)


if __name__ == "__main__":
    main()
