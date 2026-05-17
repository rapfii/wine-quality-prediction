# ============================================================
# src/preprocess.py
# Purpose: Load, clean, engineer features, and scale the
#          wine quality dataset. Returns train/test splits
#          ready for model training.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH_RED   = os.path.join(os.path.dirname(__file__), '..', 'data', 'winequality-red.csv')
DATA_PATH_WHITE = os.path.join(os.path.dirname(__file__), '..', 'data', 'winequality-white.csv')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

QUALITY_THRESHOLD = 7   # quality >= 7 → Good (1), else Bad (0)
TEST_SIZE         = 0.2
RANDOM_STATE      = 42


# ── Step 1: Load Data ────────────────────────────────────────
def load_data(path_red: str = DATA_PATH_RED, path_white: str = DATA_PATH_WHITE) -> pd.DataFrame:
    """
    Load the red and white wine quality CSV files into a single DataFrame.
    Supports semicolon-separated (UCI format) and comma-separated files.
    """
    try:
        def read_file(path, wine_type):
            df = pd.read_csv(path, sep=',')
            if df.shape[1] == 1:
                df = pd.read_csv(path, sep=';')
            df['type'] = wine_type
            return df

        df_red = read_file(path_red, 'red')
        df_white = read_file(path_white, 'white')
        
        df = pd.concat([df_red, df_white], ignore_index=True)

        print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns (Red: {df_red.shape[0]}, White: {df_white.shape[0]})")
        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"\n❌ Dataset not found.\nError: {e}"
            f"\n👉 Download from: https://archive.ics.uci.edu/ml/datasets/wine+quality"
            f"\n   Place the files at: data/winequality-red.csv and data/winequality-white.csv"
        )


# ── Step 2: Inspect Data ─────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a concise overview of the dataset."""
    print("\n── Dataset Info ──────────────────────────────")
    print(f"   Shape     : {df.shape}")
    print(f"   Columns   : {list(df.columns)}")
    print(f"\n── Missing Values ────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print(missing[missing > 0])

    print(f"\n── Quality Score Distribution ────────────────")
    print(df['quality'].value_counts().sort_index().to_string())
    print()


# ── Step 3: Feature Engineering ──────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the multi-class 'quality' score (0–10) into a
    binary label: Good (1) if quality >= 7, Bad (0) otherwise.

    Why binary?
    - Simplifies the problem for a first ML project
    - Reduces class imbalance compared to 10-class prediction
    - Directly answers a practical question: "Is this wine good?"
    """
    df = df.copy()
    df['label'] = (df['quality'] >= QUALITY_THRESHOLD).astype(int)

    good_count = df['label'].sum()
    bad_count  = len(df) - good_count
    print(f"── Label Distribution ────────────────────────")
    print(f"   Good wines (1): {good_count} ({good_count/len(df)*100:.1f}%)")
    print(f"   Bad  wines (0): {bad_count} ({bad_count/len(df)*100:.1f}%)")
    print()

    return df


# ── Step 4: Split Features and Target ────────────────────────
def split_features_target(df: pd.DataFrame):
    """
    Separate the DataFrame into:
    - X : feature matrix (all physicochemical properties)
    - y : target vector (binary label: 0 or 1)
    """
    drop_cols = ['quality', 'label']

    # Drop 'type' column if it exists (wine type: red/white)
    if 'type' in df.columns:
        drop_cols.append('type')

    X = df.drop(columns=drop_cols)
    y = df['label']

    print(f"── Features & Target ─────────────────────────")
    print(f"   Feature columns : {list(X.columns)}")
    print(f"   X shape         : {X.shape}")
    print(f"   y shape         : {y.shape}")
    print()

    return X, y


# ── Step 5: Train/Test Split ──────────────────────────────────
def split_train_test(X, y):
    """
    Split data into 80% training and 20% testing.

    stratify=y ensures that both splits have the same
    proportion of Good/Bad wines — important for imbalanced data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"── Train / Test Split ────────────────────────")
    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing  samples : {len(X_test)}")
    print()

    return X_train, X_test, y_train, y_test


# ── Step 6: Feature Scaling ───────────────────────────────────
def scale_features(X_train, X_test, save_scaler: bool = True):
    """
    Apply StandardScaler: transforms each feature to have
    mean = 0 and standard deviation = 1.

    WHY scale?
    - Logistic Regression is sensitive to feature magnitude
    - Ensures no single feature dominates due to large values
    - Random Forest doesn't strictly need it, but it doesn't hurt

    IMPORTANT: Fit on X_train ONLY, then transform both.
    Fitting on X_test would cause data leakage (cheating!).
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)         # transform only

    # Save the scaler so app.py can use the same scaling logic
    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler saved → {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


# ── Master Pipeline ───────────────────────────────────────────
def run_preprocessing(path_red: str = DATA_PATH_RED, path_white: str = DATA_PATH_WHITE):
    """
    Run the full preprocessing pipeline end-to-end.
    Returns everything needed for model training.
    """
    print("=" * 50)
    print("  🍷 Wine Quality — Preprocessing Pipeline")
    print("=" * 50)

    df              = load_data(path_red, path_white)
    inspect_data(df)
    df              = engineer_features(df)
    X, y            = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    print("✅ Preprocessing complete!\n")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, list(X.columns)


# ── Run standalone ────────────────────────────────────────────
if __name__ == '__main__':
    run_preprocessing()
