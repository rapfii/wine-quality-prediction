# ============================================================
# app.py
# Purpose: Streamlit web app — interactive wine quality
#          predictor using the saved model and scaler.
# Run:     streamlit run app.py
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow src/ imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict, FEATURE_NAMES, LABEL_MAP

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH  = os.path.join('models', 'wine_quality_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "🍷 Wine Quality Predictor",
    page_icon  = "🍷",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f0f0f; color: #e8e0d0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #c8a96e;
        color: #0f0f0f;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 15px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Sliders */
    .stSlider > div > div > div > div { background: #c8a96e; }

    /* Result box */
    .result-good {
        background: linear-gradient(135deg, #1a2e22, #1e3828);
        border: 1px solid #4caf82;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-bad {
        background: linear-gradient(135deg, #2e1a1a, #381e1e);
        border: 1px solid #e05c5c;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-title { font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
    .result-sub   { font-size: 1rem; color: #888880; }

    /* Section headers */
    h1, h2, h3 { color: #e8e0d0 !important; }
    .gold { color: #c8a96e; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Cache model and scaler so they're loaded only once."""
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, str(e)


# ── Radar / Bar Chart for Input ───────────────────────────────
def plot_input_bars(sample: dict) -> plt.Figure:
    """
    Horizontal bar chart showing the user's input values,
    normalized to 0–1 range for visual comparison.
    """
    # Approximate min/max per feature for normalization
    ranges = {
        'fixed acidity'       : (4.0, 16.0),
        'volatile acidity'    : (0.08, 1.20),
        'citric acid'         : (0.0, 1.0),
        'residual sugar'      : (0.6, 65.0),
        'chlorides'           : (0.01, 0.20),
        'free sulfur dioxide' : (1.0, 72.0),
        'total sulfur dioxide': (6.0, 440.0),
        'density'             : (0.987, 1.004),
        'pH'                  : (2.7, 4.0),
        'sulphates'           : (0.22, 2.0),
        'alcohol'             : (8.0, 15.0),
    }

    labels = list(sample.keys())
    norms  = []
    for feat, val in sample.items():
        lo, hi = ranges.get(feat, (0, 1))
        norms.append((val - lo) / (hi - lo + 1e-9))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    colors = ['#c8a96e' if n > 0.6 else '#4a6fa5' for n in norms]
    ax.barh(labels, norms, color=colors, edgecolor='none', height=0.55)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Normalized Value (0 = min, 1 = max)',
                  color='#888880', fontsize=9)
    ax.set_title('Input Feature Profile', color='#e8e0d0',
                 fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#888880', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a2a')

    plt.tight_layout()
    return fig


# ── Confidence Gauge ──────────────────────────────────────────
def plot_confidence_gauge(confidence: float, label: int) -> plt.Figure:
    """Semi-circle gauge showing prediction confidence."""
    fig, ax = plt.subplots(figsize=(4, 2.2),
                           subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    color  = '#4caf82' if label == 1 else '#e05c5c'
    angle  = np.pi * confidence          # 0 → π
    bg_theta    = np.linspace(0, np.pi, 100)
    fill_theta  = np.linspace(0, angle,  100)

    ax.plot(bg_theta,   [1] * 100, color='#2a2a2a', linewidth=12)
    ax.plot(fill_theta, [1] * 100, color=color,     linewidth=12)

    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, np.pi)
    ax.axis('off')

    ax.text(np.pi / 2, 0.35, f'{confidence*100:.1f}%',
            ha='center', va='center', fontsize=22,
            fontweight='bold', color=color,
            transform=ax.transData)

    ax.text(np.pi / 2, -0.15, 'Confidence',
            ha='center', va='center', fontsize=10,
            color='#888880', transform=ax.transData)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    model, scaler, error = load_artifacts()

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; padding-top:10px;'>
        🍷 Wine Quality Predictor
    </h1>
    <p style='text-align:center; color:#888880; font-size:15px; margin-top:-10px;'>
        Enter physicochemical properties → Get an instant quality prediction
    </p>
    <hr style='border-color:#2a2a2a; margin: 20px 0;'>
    """, unsafe_allow_html=True)

    # ── Error State ───────────────────────────────────────────
    if error:
        st.error(f"⚠️ Model not found. Run `python src/train.py` first.\n\n`{error}`")
        st.code("python src/train.py", language="bash")
        st.stop()

    # ── Sidebar — Input Sliders ───────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Wine Properties")
        st.markdown("<hr style='border-color:#2a2a2a;'>", unsafe_allow_html=True)

        sample = {
            'fixed acidity'       : st.slider('Fixed Acidity',        4.0,  16.0, 7.4,  0.1,
                                               help="Tartaric acid concentration (g/dm³)"),
            'volatile acidity'    : st.slider('Volatile Acidity',      0.08,  1.20, 0.35, 0.01,
                                               help="Acetic acid — too high = vinegar taste"),
            'citric acid'         : st.slider('Citric Acid',           0.0,   1.0,  0.40, 0.01,
                                               help="Adds freshness and flavor"),
            'residual sugar'      : st.slider('Residual Sugar',        0.6,  65.0,  2.1,  0.1,
                                               help="Sugar remaining after fermentation (g/dm³)"),
            'chlorides'           : st.slider('Chlorides',             0.01,  0.20, 0.074, 0.001,
                                               help="Salt content (g/dm³)"),
            'free sulfur dioxide' : st.slider('Free Sulfur Dioxide',   1.0,  72.0, 28.0,  1.0,
                                               help="Protects wine from oxidation (mg/dm³)"),
            'total sulfur dioxide': st.slider('Total Sulfur Dioxide',  6.0, 440.0, 92.0,  1.0,
                                               help="Total SO₂ — bound + free (mg/dm³)"),
            'density'             : st.slider('Density',               0.987, 1.004, 0.994, 0.0001,
                                               help="Depends on alcohol and sugar content"),
            'pH'                  : st.slider('pH',                    2.7,   4.0,  3.20,  0.01,
                                               help="Lower = more acidic"),
            'sulphates'           : st.slider('Sulphates',             0.22,  2.0,  0.73,  0.01,
                                               help="Antimicrobial additive (g/dm³)"),
            'alcohol'             : st.slider('Alcohol (%)',            8.0,  15.0, 11.5,  0.1,
                                               help="Percentage of alcohol by volume"),
        }

        st.markdown("<hr style='border-color:#2a2a2a;'>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Quality", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("💡 Adjust sliders and click Predict")

    # ── Main Content ──────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("#### 📊 Input Feature Profile")
        fig_bars = plot_input_bars(sample)
        st.pyplot(fig_bars, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("#### 🔍 Prediction")

        if predict_btn:
            with st.spinner("Analyzing wine properties..."):
                result = predict(sample, model=model, scaler=scaler)

            label      = result['label']
            label_text = "✅ Good Quality" if label == 1 else "❌ Bad Quality"
            conf       = result['probability']
            css_class  = "result-good" if label == 1 else "result-bad"
            color      = "#4caf82"     if label == 1 else "#e05c5c"

            # Result card
            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-title" style="color:{color};">{label_text}</div>
                <div class="result-sub">
                    Model is <strong style="color:{color};">{conf*100:.1f}%</strong> confident
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = plot_confidence_gauge(conf, label)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

            # Quick stats
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Alcohol",          f"{sample['alcohol']}%")
            m2.metric("Volatile Acidity", f"{sample['volatile acidity']}")
            m3.metric("Sulphates",        f"{sample['sulphates']}")

        else:
            st.info("👈 Adjust sliders on the left, then click **Predict Quality**.")

    # ── Bottom Section: Images ────────────────────────────────
    st.markdown("<hr style='border-color:#2a2a2a; margin:30px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📈 Model Insights")

    img_col1, img_col2, img_col3 = st.columns(3)
    images = {
        'heatmap.png'            : ('🌡️ Correlation Heatmap',    img_col1),
        'feature_importance.png' : ('🌲 Feature Importance',      img_col2),
        'confusion_matrix.png'   : ('📋 Confusion Matrix',        img_col3),
    }

    for fname, (caption, col) in images.items():
        path = os.path.join('images', fname)
        with col:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)
            else:
                st.warning(f"{fname} not found.\nRun `python src/evaluate.py`")

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <hr style='border-color:#2a2a2a; margin:30px 0;'>
    <p style='text-align:center; color:#444; font-size:13px;'>
        Wine Quality Predictor · Built with Scikit-Learn + Streamlit ·
        Data: UCI ML Repository
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
