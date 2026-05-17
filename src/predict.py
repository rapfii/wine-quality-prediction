# ============================================================
# src/predict.py
# Purpose: Load the saved model and scaler, then predict
#          the quality label for one or more wine samples.
#          Can be used standalone or imported by app.py.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Allow imports from src/ when running standalone
sys.path.insert(0, os.path.dirname(__file__))
from train import load_model, MODEL_PATH

# ── Constants ───────────────────────────────────────────────
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

FEATURE_NAMES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

LABEL_MAP = {0: '❌ Bad Quality', 1: '✅ Good Quality'}


# ── Load Scaler ───────────────────────────────────────────────
def load_scaler(path: str = SCALER_PATH):
    """
    Load the StandardScaler that was fitted during preprocessing.
    MUST use the same scaler used during training — otherwise
    the feature values will be on a completely different scale
    and predictions will be meaningless.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ Scaler not found at: {path}"
            f"\n👉 Run train.py first — it saves the scaler automatically."
        )
    scaler = joblib.load(path)
    print(f"✅ Scaler loaded ← {path}")
    return scaler


# ── Validate Input ────────────────────────────────────────────
def validate_input(sample: dict) -> pd.DataFrame:
    """
    Ensure the input sample contains all required features.
    Converts dict → single-row DataFrame for sklearn compatibility.
    """
    missing = [f for f in FEATURE_NAMES if f not in sample]
    if missing:
        raise ValueError(
            f"\n❌ Missing features in input: {missing}"
            f"\n👉 Required features: {FEATURE_NAMES}"
        )

    df = pd.DataFrame([sample])[FEATURE_NAMES]  # enforce column order
    return df


# ── Core Prediction Function ──────────────────────────────────
def predict(sample: dict, model=None, scaler=None) -> dict:
    """
    Predict wine quality for a single wine sample.

    Parameters:
        sample (dict): Feature values keyed by feature name.
                       Example: {'alcohol': 9.5, 'pH': 3.2, ...}
        model        : Pre-loaded model (loads from disk if None)
        scaler       : Pre-loaded scaler (loads from disk if None)

    Returns:
        dict with keys:
            - 'label'       : 0 (Bad) or 1 (Good)
            - 'label_text'  : Human-readable label
            - 'probability' : Confidence score [0.0 – 1.0]
            - 'input'       : Original input values
    """
    # Lazy-load model and scaler if not passed in
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    # Validate and format input
    input_df = validate_input(sample)

    # Scale using the SAME scaler from training
    input_scaled = scaler.transform(input_df)

    # Predict label
    label = int(model.predict(input_scaled)[0])

    # Predict probability (confidence)
    proba = model.predict_proba(input_scaled)[0]
    confidence = float(proba[label])

    return {
        'label'      : label,
        'label_text' : LABEL_MAP[label],
        'probability': round(confidence, 4),
        'input'      : sample
    }


# ── Batch Prediction ──────────────────────────────────────────
def predict_batch(samples: list, model=None, scaler=None) -> pd.DataFrame:
    """
    Predict quality for multiple wine samples at once.

    Parameters:
        samples (list of dict): Each dict is one wine sample.

    Returns:
        pd.DataFrame with predictions appended to original features.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    df           = pd.DataFrame(samples)[FEATURE_NAMES]
    scaled       = scaler.transform(df)
    labels       = model.predict(scaled)
    probabilities = model.predict_proba(scaled)

    df['predicted_label'] = labels
    df['label_text']      = [LABEL_MAP[l] for l in labels]
    df['confidence']      = [
        round(probabilities[i][labels[i]], 4)
        for i in range(len(labels))
    ]

    return df


# ── Pretty Print Result ───────────────────────────────────────
def print_prediction(result: dict) -> None:
    """Print a formatted prediction result to the terminal."""
    print("\n" + "=" * 50)
    print("  🍷 Prediction Result")
    print("=" * 50)
    print(f"\n  Result     : {result['label_text']}")
    print(f"  Confidence : {result['probability'] * 100:.1f}%")
    print(f"\n── Input Features ────────────────────────────")
    for feature, value in result['input'].items():
        print(f"   {feature:<28}: {value}")
    print()


# ── Example Samples ───────────────────────────────────────────
SAMPLE_GOOD_WINE = {
    'fixed acidity'       : 7.4,
    'volatile acidity'    : 0.35,
    'citric acid'         : 0.40,
    'residual sugar'      : 2.1,
    'chlorides'           : 0.074,
    'free sulfur dioxide' : 28.0,
    'total sulfur dioxide': 92.0,
    'density'             : 0.9940,
    'pH'                  : 3.20,
    'sulphates'           : 0.73,
    'alcohol'             : 11.5
}

SAMPLE_BAD_WINE = {
    'fixed acidity'       : 8.1,
    'volatile acidity'    : 0.66,
    'citric acid'         : 0.10,
    'residual sugar'      : 2.5,
    'chlorides'           : 0.094,
    'free sulfur dioxide' : 18.0,
    'total sulfur dioxide': 65.0,
    'density'             : 0.9980,
    'pH'                  : 3.55,
    'sulphates'           : 0.52,
    'alcohol'             : 9.2
}


# ── Run standalone ────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  🍷 Wine Quality — Prediction Demo")
    print("=" * 50)

    # Load once, reuse for both predictions
    model  = load_model()
    scaler = load_scaler()

    print("\n📌 Sample 1 — Expected: GOOD wine")
    result1 = predict(SAMPLE_GOOD_WINE, model=model, scaler=scaler)
    print_prediction(result1)

    print("📌 Sample 2 — Expected: BAD wine")
    result2 = predict(SAMPLE_BAD_WINE, model=model, scaler=scaler)
    print_prediction(result2)

    print("\n📌 Batch Prediction — Both samples together")
    batch_df = predict_batch(
        [SAMPLE_GOOD_WINE, SAMPLE_BAD_WINE],
        model=model, scaler=scaler
    )
    print(batch_df[['alcohol', 'volatile acidity', 'label_text', 'confidence']])
    print()
