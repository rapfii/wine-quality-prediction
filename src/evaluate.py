# ============================================================
# src/evaluate.py
# Purpose: Generate all evaluation metrics and save the three
#          key visualizations: heatmap, confusion matrix,
#          and feature importance chart.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Allow imports from src/ when running standalone
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_preprocessing, load_data
from train      import run_training, load_model, MODEL_PATH

# ── Constants ───────────────────────────────────────────────
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Consistent color palette across all charts
PALETTE = {
    'bg'       : '#0f0f0f',
    'panel'    : '#1a1a1a',
    'accent'   : '#c8a96e',   # warm gold
    'good'     : '#4caf82',   # green
    'bad'      : '#e05c5c',   # red
    'text'     : '#e8e0d0',
    'subtext'  : '#888880',
}


# ── Shared style helper ───────────────────────────────────────
def _apply_dark_style(ax, title: str, xlabel: str = '', ylabel: str = '') -> None:
    """Apply consistent dark-theme styling to any Axes object."""
    ax.set_facecolor(PALETTE['panel'])
    ax.set_title(title, color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=PALETTE['subtext'], fontsize=11)
    ax.set_ylabel(ylabel, color=PALETTE['subtext'], fontsize=11)
    ax.tick_params(colors=PALETTE['subtext'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


# ── Plot 1: Correlation Heatmap ───────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Visualize pairwise Pearson correlations between all features.

    HOW TO READ IT:
    - Dark red  (+1.0) → strong positive correlation
    - Dark blue (-1.0) → strong negative correlation
    - Near zero        → weak / no linear relationship
    - Look at the 'quality' row to spot the best predictors
    """
    # Drop non-numeric cols if present
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor='#1a1a1a',
        annot_kws={'size': 8, 'color': PALETTE['text']},
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=PALETTE['subtext'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['subtext'])
    cbar.ax.set_facecolor(PALETTE['panel'])

    ax.set_title(
        '🍷 Feature Correlation Heatmap',
        color=PALETTE['text'], fontsize=15, fontweight='bold', pad=18
    )
    ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save:
        path = os.path.join(IMAGES_DIR, 'heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved → {path}")

    plt.show()
    plt.close()


# ── Plot 2: Confusion Matrix ──────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, save: bool = True) -> None:
    """
    Visualize predicted vs actual labels in a 2×2 grid.

    HOW TO READ IT:
    ┌──────────────┬──────────────┐
    │  True Neg    │  False Pos   │  ← Actual: Bad
    │  (correct)   │  (missed)    │
    ├──────────────┼──────────────┤
    │  False Neg   │  True Pos    │  ← Actual: Good
    │  (missed)    │  (correct)   │
    └──────────────┴──────────────┘
      Predicted: Bad   Predicted: Good

    - High diagonal numbers = good model
    - Off-diagonal = errors the model made
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Bad (0)', 'Good (1)']
    )
    disp.plot(
        ax=ax,
        cmap='YlOrBr',
        colorbar=False,
        values_format='d'
    )

    # Recolor text inside cells
    for text in disp.text_.ravel():
        text.set_color(PALETTE['bg'])
        text.set_fontsize(18)
        text.set_fontweight('bold')

    _apply_dark_style(
        ax,
        title='Confusion Matrix',
        xlabel='Predicted Label',
        ylabel='Actual Label'
    )

    # Axis label colors
    ax.xaxis.label.set_color(PALETTE['subtext'])
    ax.yaxis.label.set_color(PALETTE['subtext'])
    ax.tick_params(colors=PALETTE['text'], labelsize=11)

    # Annotate accuracy
    acc = accuracy_score(y_test, y_pred)
    fig.text(
        0.5, 0.01,
        f'Overall Accuracy: {acc*100:.2f}%',
        ha='center', color=PALETTE['accent'],
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        path = os.path.join(IMAGES_DIR, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved → {path}")

    plt.show()
    plt.close()


# ── Plot 3: Feature Importance ────────────────────────────────
def plot_feature_importance(model, feature_names: list, save: bool = True) -> None:
    """
    Horizontal bar chart showing which features drive predictions.
    Only available for tree-based models (Random Forest).

    HOW TO READ IT:
    - Longer bar = more influential feature in predictions
    - Alcohol consistently ranks #1 for wine quality
    - Low-importance features could be dropped to simplify the model
    """
    if not hasattr(model, 'feature_importances_'):
        print("ℹ️  Feature importance not available for this model type.")
        return

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    sorted_features     = [feature_names[i] for i in sorted_idx]
    sorted_importances  = importances[sorted_idx]

    # Color: top 3 features highlighted in gold
    colors = [
        PALETTE['accent'] if i >= len(sorted_idx) - 3 else '#4a6fa5'
        for i in range(len(sorted_idx))
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(PALETTE['bg'])

    bars = ax.barh(
        sorted_features,
        sorted_importances,
        color=colors,
        edgecolor='none',
        height=0.65
    )

    # Value labels on bars
    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', color=PALETTE['subtext'], fontsize=9
        )

    _apply_dark_style(
        ax,
        title='🌲 Random Forest — Feature Importance',
        xlabel='Importance Score',
        ylabel=''
    )

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.set_xlim(0, sorted_importances.max() * 1.18)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE['accent'], label='Top 3 Features'),
        Patch(facecolor='#4a6fa5',        label='Other Features'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        facecolor=PALETTE['panel'],
        edgecolor='#333333',
        labelcolor=PALETTE['text'],
        fontsize=10
    )

    plt.tight_layout()

    if save:
        path = os.path.join(IMAGES_DIR, 'feature_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved → {path}")

    plt.show()
    plt.close()


# ── Plot 4: Quality Distribution ─────────────────────────────
def plot_quality_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart of raw quality score counts (0–10).
    Shows the class imbalance problem before binarization.

    HOW TO READ IT:
    - Most wines cluster at scores 5 and 6 (mediocre)
    - Very few wines score 3 or 9 (extreme ends)
    - This imbalance is WHY we binarize (Good vs Bad)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE['bg'])

    quality_counts = df['quality'].value_counts().sort_index()

    bar_colors = [
        PALETTE['good'] if q >= 7 else PALETTE['bad']
        for q in quality_counts.index
    ]

    bars = ax.bar(
        quality_counts.index.astype(str),
        quality_counts.values,
        color=bar_colors,
        edgecolor='none',
        width=0.6
    )

    # Count labels on top of each bar
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(int(bar.get_height())),
            ha='center', va='bottom',
            color=PALETTE['subtext'], fontsize=9
        )

    _apply_dark_style(
        ax,
        title='🍷 Wine Quality Score Distribution',
        xlabel='Quality Score',
        ylabel='Count'
    )

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE['good'], label='Good (score ≥ 7)'),
            Patch(facecolor=PALETTE['bad'],  label='Bad  (score < 7)'),
        ],
        facecolor=PALETTE['panel'],
        edgecolor='#333333',
        labelcolor=PALETTE['text'],
        fontsize=10
    )

    plt.tight_layout()

    if save:
        path = os.path.join(IMAGES_DIR, 'quality_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"✅ Saved → {path}")

    plt.show()
    plt.close()


# ── Full Metrics Report ───────────────────────────────────────
def print_full_report(model, X_test, y_test) -> None:
    """
    Print accuracy + detailed classification report to terminal.
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  📋 Full Evaluation Report")
    print("=" * 50)
    print(f"\n  Accuracy : {acc * 100:.2f}%\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Bad (0)', 'Good (1)']
    ))


# ── Master Pipeline ───────────────────────────────────────────
def run_evaluation():
    """
    Full evaluation pipeline:
    1. Load preprocessed data
    2. Load the saved model
    3. Print metrics report
    4. Generate and save all 4 visualizations
    """
    raw_df = load_data()

    # Get test split for evaluation
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing()

    # Load the saved best model
    model = load_model()

    print_full_report(model, X_test, y_test)

    print("\n── Generating Visualizations ──────────────────")
    plot_quality_distribution(raw_df)
    plot_correlation_heatmap(raw_df)
    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, feature_names)

    print("\n🎉 All visualizations saved to images/\n")


# ── Run standalone ────────────────────────────────────────────
if __name__ == '__main__':
    run_evaluation()
