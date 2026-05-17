# ============================================================
# src/train.py
# Purpose: Train Logistic Regression and Random Forest models,
#          compare their performance, and save the best one.
# ============================================================

import os
import joblib
import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, classification_report

from preprocess import run_preprocessing

# ── Constants ───────────────────────────────────────────────
MODEL_DIR       = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH      = os.path.join(MODEL_DIR, 'wine_quality_model.pkl')
RANDOM_STATE    = 42


# ── Model Definitions ────────────────────────────────────────
def get_models() -> dict:
    """
    Returns a dictionary of models to train and compare.

    Logistic Regression:
    - Baseline linear classifier
    - Fast, interpretable, good for linearly separable data
    - max_iter=1000 prevents convergence warnings on this dataset

    Random Forest:
    - Ensemble of 100 decision trees
    - Handles non-linear relationships well
    - Naturally provides feature importance scores
    - Less prone to overfitting than a single decision tree
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,       # number of trees in the forest
            max_depth=None,         # trees grow until leaves are pure
            random_state=RANDOM_STATE,
            n_jobs=-1               # use all CPU cores for speed
        )
    }


# ── Train a Single Model ──────────────────────────────────────
def train_model(model, X_train, y_train):
    """
    Fit a model on the training data.
    Returns the trained model.
    """
    model.fit(X_train, y_train)
    return model


# ── Evaluate a Single Model ───────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> float:
    """
    Print accuracy and classification report for one model.
    Returns accuracy score for comparison.
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n── {model_name} ──────────────────────────────")
    print(f"   Accuracy : {accuracy * 100:.2f}%")
    print(f"\n   Classification Report:")

    report = classification_report(
        y_test, y_pred,
        target_names=['Bad (0)', 'Good (1)']
    )
    # Indent the report for cleaner output
    for line in report.splitlines():
        print(f"   {line}")

    return accuracy


# ── Compare All Models ────────────────────────────────────────
def train_and_compare(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train every model in get_models(), evaluate each on the
    test set, and return the best model with its name.
    """
    models  = get_models()
    results = {}

    print("\n" + "=" * 50)
    print("  📊 Model Training & Comparison")
    print("=" * 50)

    for name, model in models.items():
        print(f"\n🔄 Training: {name}...")
        trained = train_model(model, X_train, y_train)
        acc     = evaluate_model(trained, X_test, y_test, name)
        results[name] = (trained, acc)

    # ── Pick the Winner ──────────────────────────────────────
    best_name, (best_model, best_acc) = max(
        results.items(),
        key=lambda item: item[1][1]   # sort by accuracy
    )

    print("\n" + "=" * 50)
    print(f"  🏆 Best Model : {best_name}")
    print(f"  🎯 Accuracy   : {best_acc * 100:.2f}%")
    print("=" * 50)

    return best_model, best_name, results


# ── Save the Best Model ───────────────────────────────────────
def save_model(model, path: str = MODEL_PATH) -> None:
    """
    Serialize the trained model to disk using joblib.

    WHY joblib instead of pickle?
    - joblib is optimized for large numpy arrays
    - Faster and more memory-efficient for sklearn models
    - Standard practice in the ML community
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n✅ Model saved → {path}")


# ── Load a Saved Model ────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    """
    Load a previously saved model from disk.
    Used by predict.py and app.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ Model not found at: {path}"
            f"\n👉 Run train.py first to generate the model."
        )
    model = joblib.load(path)
    print(f"✅ Model loaded ← {path}")
    return model


# ── Feature Importance (Random Forest only) ───────────────────
def print_feature_importance(model, feature_names: list) -> None:
    """
    Print feature importance scores if the model supports it
    (Random Forest, Gradient Boosting, etc.)
    """
    if not hasattr(model, 'feature_importances_'):
        print("\nℹ️  This model does not support feature importance.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    print("\n── Feature Importance (Top 11) ───────────────")
    for rank, idx in enumerate(indices, start=1):
        bar = "█" * int(importances[idx] * 100)
        print(f"   {rank:2}. {feature_names[idx]:<25} {importances[idx]:.4f}  {bar}")


# ── Master Pipeline ───────────────────────────────────────────
def run_training():
    """
    Full training pipeline:
    1. Preprocess the data
    2. Train and compare models
    3. Save the best model
    4. Print feature importance
    """
    print("=" * 50)
    print("  🍷 Wine Quality — Training Pipeline")
    print("=" * 50)

    # Get preprocessed data from preprocess.py
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing()

    # Train, compare, select best
    best_model, best_name, all_results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    # Save winner
    save_model(best_model)

    # Bonus: show what the model considers important
    print_feature_importance(best_model, feature_names)

    print("\n🎉 Training pipeline complete!")
    print(f"   Best model '{best_name}' saved and ready for predictions.\n")

    return best_model, feature_names


# ── Run standalone ────────────────────────────────────────────
if __name__ == '__main__':
    # When running directly: cd wine-quality-prediction && python src/train.py
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    run_training()
