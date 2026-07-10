"""
Wildfire Ignition Prediction — Interactive Demo

Run with: streamlit run streamlit/app.py   (from the project root)
Requires the artifact files produced by prepare_demo_data.py in the ./streamlit folder,
plus a background image at ./streamlit/bg.jpg.
"""

import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Wildfire Ignition Predictor", layout="wide")

ASSET_DIR = Path("streamlit")
PRODUCTION_THRESHOLD = 0.37

# ---------- Background image (base64-embedded so CSS can use it) ----------
@st.cache_data
def get_bg_base64():
    bg_path = ASSET_DIR / "bg.jpg"
    if not bg_path.exists():
        return None
    return base64.b64encode(bg_path.read_bytes()).decode()

bg_b64 = get_bg_base64()

if bg_b64:
    bg_layer = f'url("data:image/jpeg;base64,{bg_b64}")'
else:
    bg_layer = "linear-gradient(160deg, #2b1005 0%, #7a2f0d 45%, #c25a1c 75%, #e8892e 100%)"

# ---------- Global CSS: fixed dimmed/blurred backdrop + glass content cards ----------
st.markdown(f"""
<style>
.stApp {{ background: transparent; }}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -2;
    background-image: {bg_layer};
    background-size: cover;
    background-position: center;
}}
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background: rgba(10, 6, 4, var(--dim, 0.4));
    backdrop-filter: blur(var(--blur, 4px));
    -webkit-backdrop-filter: blur(var(--blur, 4px));
    transition: background 0.25s ease, backdrop-filter 0.25s ease;
}}
.hero {{
    position: relative;
    height: 78vh;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    text-align: center; color: #fff;
    text-shadow: 0 2px 20px rgba(0,0,0,0.8);
}}
.hero h1, .hero p, .hero .metrics, .hero .scroll-hint {{
    position: relative;
    z-index: 1;
}}
.hero h1 {{ font-size: 4.2rem; font-weight: 800; margin-bottom: 0.4rem; letter-spacing: -1px; }}
.hero p {{ font-size: 1.35rem; max-width: 720px; opacity: 0.95; line-height: 1.5; }}
.hero .metrics {{ display: flex; gap: 2.5rem; margin-top: 2.2rem; flex-wrap: wrap; justify-content: center; }}
.hero .metric-box {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px; padding: 1rem 1.6rem;
    backdrop-filter: blur(6px);
}}
.hero .metric-box .num {{ font-size: 2.1rem; font-weight: 700; }}
.hero .metric-box .lbl {{ font-size: 0.8rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }}
.scroll-hint {{ margin-top: 2.5rem; font-size: 0.9rem; opacity: 0.75; animation: bob 1.8s ease-in-out infinite; }}
@keyframes bob {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(8px); }} }}
.block-container {{ padding-top: 1rem; max-width: 1200px; }}
/* Dark grey cards. Each st.container carries a unique key (card1, card2, ...),
   which Streamlit renders as class `st-key-cardN` — reliably targetable via a
   prefix match, unlike the data-testid guessing that failed on this build. */
[class*="st-key-card"] {{
    background: rgba(38, 40, 44, 0.9) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 28px !important;
    padding: 1.2rem 1.6rem !important;
    margin-bottom: 1rem !important;
}}
/* Data sources card gets extra breathing room around its grid */
.st-key-card3 {{
    padding: 1.4rem 2.2rem 1.8rem !important;
}}
h2, h3 {{ color: #ffffff !important; }}
.stApp, .stMarkdown, p, label {{ color: #eceff2; }}

/* Chart images get a light panel so their native black text/labels stay readable
   against the dark grey cards, with matching soft-squircle corners. */
[data-testid="stImage"] img,
.stpyplot img,
[data-testid="stImage"] {{
    background: #ffffff;
    border-radius: 20px;
    padding: 10px;
}}

/* Data source cards */
.source-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0.6rem 1.2rem;
}}
.source-card {{
    display: block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    text-decoration: none !important;
    transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
}}
.source-card:hover {{
    transform: translateY(-3px);
    background: rgba(255,255,255,0.09);
    border-color: rgba(255,180,120,0.5);
}}
.source-name {{
    font-size: 1.02rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 0.25rem;
}}
.source-desc {{
    font-size: 0.82rem;
    color: #b9c0c7 !important;
    line-height: 1.35;
}}

/* "Why" motivation callout */
.why-box {{
    display: flex;
    align-items: center;
    gap: 1.6rem;
    background: linear-gradient(135deg, rgba(226,90,28,0.16), rgba(120,47,13,0.10));
    border: 1px solid rgba(255,150,80,0.28);
    border-left: 4px solid #e2851c;
    border-radius: 18px;
    padding: 1.4rem 1.7rem;
    margin: 0.3rem 0 0.5rem;
}}
.why-stat {{
    flex: 0 0 auto;
    text-align: center;
    line-height: 1;
}}
.why-stat .big {{
    font-size: 3.2rem;
    font-weight: 800;
    color: #ff9d4d;
    display: block;
}}
.why-stat .cap {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #d8c3b2;
    margin-top: 0.3rem;
    display: block;
}}
.why-text {{
    font-size: 1.02rem;
    line-height: 1.5;
    color: #f2ede9;
}}
.why-text b {{ color: #ffffff; }}

/* "Key findings" — header + card grid */
.findings {{ margin: 0.5rem 0 0.3rem; }}
.findings-title {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.3rem;
    font-weight: 800;
    color: #7ee2a0 !important;
    margin: 0 0 0.25rem;
}}
.findings-sub {{
    font-size: 0.9rem;
    color: #a9b4ab !important;
    margin: 0 0 1.15rem;
}}
.findings-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
    gap: 1rem;
}}
.finding-card {{
    position: relative;
    background: linear-gradient(160deg, rgba(46,160,86,0.11), rgba(20,83,45,0.04));
    border: 1px solid rgba(120,220,150,0.18);
    border-radius: 20px;
    padding: 1.3rem 1.35rem 1.4rem;
    overflow: hidden;
    transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
}}
.finding-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(120,220,150,0.45);
    background: linear-gradient(160deg, rgba(46,160,86,0.17), rgba(20,83,45,0.07));
}}
.finding-card .fc-num {{
    position: absolute;
    top: 0.6rem; right: 1.05rem;
    font-size: 2.7rem;
    font-weight: 800;
    color: rgba(126,226,160,0.13);
    line-height: 1;
    pointer-events: none;
}}
.finding-card .fc-title {{
    font-size: 1.02rem;
    font-weight: 700;
    color: #ffffff !important;
    line-height: 1.35;
    margin-bottom: 0.5rem;
}}
.finding-card .fc-body {{
    font-size: 0.9rem;
    line-height: 1.55;
    color: #cdd6ce !important;
}}
.finding-card .fc-body b {{ color: #eaf5ec !important; }}
.finding-card .fc-tag {{
    display: inline-block;
    margin-top: 0.9rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: #8fe6ac !important;
    background: rgba(46,160,86,0.14);
    border: 1px solid rgba(120,220,150,0.22);
    border-radius: 999px;
    padding: 0.3rem 0.72rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------- Load artifacts ----------
import urllib.request

# Large model files aren't stored in the repo (GitHub's file-size limits).
# They're hosted as GitHub Release assets and downloaded on first run.
RELEASE_BASE = "https://github.com/eimirjihad/wildfire-ignition-prediction/releases/download/v1"
REMOTE_ARTIFACTS = ["rf_model.joblib", "selected_features.joblib", "shap_explainer.joblib"]

def _ensure_artifact(name):
    """Fetch an artifact from the GitHub Release if it isn't already on disk."""
    path = ASSET_DIR / name
    if path.exists():
        return path
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading {name} (first run only)…"):
        urllib.request.urlretrieve(f"{RELEASE_BASE}/{name}", path)
    return path

@st.cache_resource
def load_artifacts():
    for name in REMOTE_ARTIFACTS:
        _ensure_artifact(name)
    rf = joblib.load(ASSET_DIR / 'rf_model.joblib')
    selected_features = joblib.load(ASSET_DIR / 'selected_features.joblib')
    explainer = joblib.load(ASSET_DIR / 'shap_explainer.joblib')
    return rf, selected_features, explainer

@st.cache_data
def load_data():
    feature_df = pd.read_csv(ASSET_DIR / 'demo_feature_df.csv')
    daily_sequences = pd.read_csv(ASSET_DIR / 'demo_daily_sequences.csv')
    return feature_df, daily_sequences

@st.cache_data
def load_full_predictions():
    p = ASSET_DIR / 'full_predictions.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

@st.cache_data
def load_test_predictions():
    p = ASSET_DIR / 'test_predictions.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

rf, selected_features, explainer = load_artifacts()
feature_df, daily_sequences = load_data()
full_preds = load_full_predictions()
test_preds = load_test_predictions()

TREND_FEATURES = ["precip_mm", "temp_max_c", "fuel_moist_100hr_pct", "burning_index", "vpd_kpa"]
TREND_LABELS = {
    "precip_mm": "Precipitation (mm/day)",
    "temp_max_c": "Max Temperature (°C)",
    "fuel_moist_100hr_pct": "Fuel Moisture, 100hr (%)",
    "burning_index": "Burning Index",
    "vpd_kpa": "Vapor Pressure Deficit (kPa)",
}

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>Wildfire Ignition Predictor</h1>
    <p>Predicting where a wildfire is about to start using weather, population, and land cover data
    across the US (2014–2025).</p>
    <div class="metrics">
        <div class="metric-box"><div class="num">0.61</div><div class="lbl">F2 (5-fold CV)</div></div>
        <div class="metric-box"><div class="num">70%</div><div class="lbl">Recall</div></div>
        <div class="metric-box"><div class="num">125,731</div><div class="lbl">Fire Windows</div></div>
    </div>
    <div class="scroll-hint">↓ scroll to explore predictions</div>
</div>
""", unsafe_allow_html=True)

# ---------- INTRO ----------
with st.container(border=False, key="card1"):
    st.markdown("""
<div class="why-box">
  <div class="why-stat">
    <span class="big">84%</span>
    <span class="cap">of US wildfires<br>are human-caused</span>
  </div>
  <div class="why-text">
    <b>Why this project exists.</b> The original <a href="https://arxiv.org/abs/2601.14238" target="_blank">FireCastRL</a>
    model predicted wildfire ignition from weather alone, but a large majority of wildfires are started due to humans.
     This project adds the missing human signal (population density, land cover,
    holidays) to test whether it makes the model better.
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="findings">
  <div class="findings-title">Key findings</div>
  <div class="findings-sub">Three takeaways from the importance and ablation analysis.</div>
  <div class="findings-grid">
    <div class="finding-card">
      <span class="fc-num">01</span>
      <div class="fc-title">Weather alone predicts fires well</div>
      <div class="fc-body">Most ignitions are human-caused, yet the model stays accurate — a fire only spreads when the weather is dry, hot, and windy. Those conditions gate whether any spark becomes a wildfire.</div>
      <span class="fc-tag">Weather gates ignition</span>
    </div>
    <div class="finding-card">
      <span class="fc-num">02</span>
      <div class="fc-title">Population density matters — but doesn't add accuracy</div>
      <div class="fc-body">It ranks <b>#2</b> in permutation importance and SHAP leans on it, yet a controlled ablation shows removing it doesn't hurt performance. The signal is already recovered from correlated weather features.</div>
      <span class="fc-tag">Feature redundancy</span>
    </div>
    <div class="finding-card">
      <span class="fc-num">03</span>
      <div class="fc-title">Population density is invisible to linear models</div>
      <div class="fc-body">In logistic regression it ranks near the bottom; only tree-based models surface its importance. Its effect is interaction-driven, not additive — so a linear model structurally can't see it.</div>
      <span class="fc-tag">Interaction-driven</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- WHAT THIS IS ----------
with st.container(border=False, key="card_what"):
    st.markdown("""
### What this is

This is a **Random Forest classifier** that estimates the probability a given location will see a wildfire
ignition from the weather leading up to it, plus human-activity signals.
                
It extends the [FireCastRL](https://arxiv.org/abs/2601.14238) dataset and is tuned for recall over precision.
Missing a real fire is far costlier than a false alarm, so the decision threshold is set to 0.37 to catch 
roughly 70% of real ignitions at the cost of more false positives.
                
Every window below is real held-out data the model never trained on. Explore individual predictions, 
see the weather that drove them, and read the model's own per-prediction reasoning via SHAP.
Then scroll to the confusion matrix and a sanity check against famous named fires.
""")

# ---------- DATA SOURCES ----------
with st.container(border=False, key="card3"):
    st.markdown("""
### Data sources

<div class="source-grid">
  <a class="source-card" href="https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset" target="_blank">
    <div class="source-name">GRIDMET + IRWIN</div>
    <div class="source-desc">Daily weather &amp; ignition records, 2014–2025 · via FireCastRL</div>
  </a>
  <a class="source-card" href="https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11" target="_blank">
    <div class="source-name">NASA SEDAC GPWv4</div>
    <div class="source-desc">Gridded population density</div>
  </a>
  <a class="source-card" href="https://www.mrlc.gov/data" target="_blank">
    <div class="source-name">USGS NLCD</div>
    <div class="source-desc">Land cover classification</div>
  </a>
  <a class="source-card" href="https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/" target="_blank">
    <div class="source-name">HIFLD</div>
    <div class="source-desc">Electric transmission lines</div>
  </a>
</div>
""", unsafe_allow_html=True)
    

# ---------- WINDOW EXPLORER (single card) ----------
with st.container(border=False, key="card2"):
    st.markdown("## Explore Predictions")

    # Compact filter — segmented control sits right above what it affects
    FILTERS = ["Any", "Fires", "Non-fires", "Mistakes"]
    FILTER_MAP = {
        "Any": "Any window", "Fires": "Real fires only",
        "Non-fires": "Non-fire windows only", "Mistakes": "Model's biggest mistakes",
    }
    picked = st.segmented_control("Filter", FILTERS, default="Any", label_visibility="collapsed")
    filter_choice = FILTER_MAP.get(picked, "Any window")

    if filter_choice == "Real fires only":
        options_df = feature_df[feature_df['label'] == 1]
    elif filter_choice == "Non-fire windows only":
        options_df = feature_df[feature_df['label'] == 0]
    elif filter_choice == "Model's biggest mistakes":
        feature_df['error'] = (feature_df['label'] - feature_df['predicted_proba']).abs()
        options_df = feature_df.sort_values('error', ascending=False).head(20)
    else:
        options_df = feature_df

    options_df = options_df.reset_index(drop=True)
    option_labels = [
        f"{row.latitude:.2f}, {row.longitude:.2f} · {pd.Timestamp(row.event_date).date()} "
        f"({'fire' if row.label == 1 else 'no fire'})"
        for row in options_df.itertuples()
    ]

    if len(options_df) == 0:
        st.warning("No windows match this filter.")
        st.stop()

    selected_label = st.selectbox("Window", option_labels, label_visibility="collapsed")
    selected_idx = option_labels.index(selected_label)
    selected_row = options_df.iloc[selected_idx]

    # Indicators
    lc_active = [c.replace('lc_', '').title() for c in selected_row.index
                 if c.startswith('lc_') and selected_row.get(c)]
    proba = selected_row['predicted_proba']
    predicted_fire = proba >= PRODUCTION_THRESHOLD

    ind1, ind2, ind3, ind4, ind5 = st.columns(5)
    ind1.metric("Pop. Density", f"{selected_row['pop_density']:.0f} /km²")
    ind2.metric("Land Cover", lc_active[0] if lc_active else "—")
    ind3.metric("Fire Probability", f"{proba:.0%}")
    ind4.metric("Prediction", "Fire" if predicted_fire else "No Fire")
    ind5.metric("Actual", "Fire" if selected_row['label'] == 1 else "No Fire")

    st.divider()

    # Weather trend + SHAP side by side
    wcol, scol = st.columns(2)

    with wcol:
        st.markdown("**Weather, 60 days before**")
        window_daily = daily_sequences[
            (daily_sequences['latitude'] == selected_row['latitude']) &
            (daily_sequences['longitude'] == selected_row['longitude']) &
            (daily_sequences['event_date'] == selected_row['event_date'])
        ].sort_values('day_offset')

        if len(window_daily) == 0:
            st.info("Daily weather not saved for this window.")
        else:
            chosen_trend = st.multiselect(
                "Variables", TREND_FEATURES,
                default=["precip_mm", "temp_max_c", "fuel_moist_100hr_pct"],
                format_func=lambda x: TREND_LABELS[x], label_visibility="collapsed"
            )
            if chosen_trend:
                colors = plt.cm.tab10.colors
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                lines, labels = [], []
                for i, feat in enumerate(chosen_trend):
                    plot_ax = ax if i == 0 else ax.twinx()
                    if i > 1:
                        plot_ax.spines['right'].set_position(('outward', 55 * (i - 1)))
                    line, = plot_ax.plot(
                        window_daily['day_offset'], window_daily[feat],
                        color=colors[i], label=TREND_LABELS[feat], linewidth=2
                    )
                    plot_ax.set_ylabel(TREND_LABELS[feat], color=colors[i], fontsize=8)
                    plot_ax.tick_params(axis='y', labelcolor=colors[i], labelsize=7)
                    lines.append(line)
                    labels.append(TREND_LABELS[feat])
                ax.set_xlabel("Days before event", fontsize=8)
                ax.legend(lines, labels, loc='upper left', fontsize=7)
                st.pyplot(fig)

    with scol:
        st.markdown("**Why the model predicted this**")
        st.caption("Red pushes toward fire, blue toward no-fire.")
        X_row = selected_row[selected_features].to_frame().T.astype(float)
        shap_values_row = explainer.shap_values(X_row)
        if isinstance(shap_values_row, list):
            sv = shap_values_row[1][0]
            base_value = explainer.expected_value[1]
        else:
            sv = shap_values_row[0, :, 1]
            base_value = (explainer.expected_value[1]
                          if hasattr(explainer.expected_value, '__len__')
                          else explainer.expected_value)
        fig2, ax = plt.subplots(figsize=(6, 5))
        fig2.patch.set_alpha(0)
        explanation = shap.Explanation(
            values=sv, base_values=base_value, data=X_row.iloc[0].values, feature_names=selected_features
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig2)


# ---------- CONFUSION MATRIX ----------
with st.container(border=False, key="card6"):
    st.markdown("## Overall Performance")
    if test_preds is None:
        st.info("`test_predictions.csv` not found. Run the updated `prepare_demo_data.py` to generate it.")
    else:
        y_true = test_preds['label'].values
        y_pred = (test_preds['predicted_proba'].values >= PRODUCTION_THRESHOLD).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        cmcol1, cmcol2 = st.columns([2, 3])
        with cmcol1:
            st.markdown("### Confusion Matrix")
            st.caption(f"Held-out test set ({len(test_preds):,} unseen windows).")
            fig3, ax = plt.subplots(figsize=(5, 4.2))
            fig3.patch.set_alpha(0)   # transparent so the light CSS panel shows through
            ax.imshow(cm, cmap='Oranges')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['No Fire', 'Fire'])
            ax.set_yticklabels(['No Fire', 'Fire'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            grid = [[f"{tn:,}\n(True Neg)", f"{fp:,}\n(False Alarm)"],
                    [f"{fn:,}\n(Missed Fire)", f"{tp:,}\n(Caught Fire)"]]
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    color = 'white' if val > cm.max() * 0.55 else '#3a1c0a'
                    ax.text(j, i, grid[i][j], ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')
            st.pyplot(fig3)

        with cmcol2:
            st.markdown("### What it means")
            m1, m2 = st.columns(2)
            m1.metric("Recall (fires caught)", f"{recall:.1%}")
            m2.metric("Precision (alerts that were real)", f"{precision:.1%}")
            st.markdown(f"""
On unseen data, the model catches **{recall:.0%}** of real fires (**{tp:,}** caught, **{fn:,}**
missed). High recall is the goal.

The cost: **{precision:.0%}** of alerts are real fires (**{fp:,}** false alarms), set by the 0.37
threshold.

> Honest held-out numbers, matching the headline **F2 = 0.608 ± 0.004** (5-fold CV).
""")

# ---------- FAMOUS FIRES ----------
with st.container(border=False, key="card7"):
    st.markdown("## Famous Fires")
    st.caption(
        "Ten well-known US wildfires, matched to the nearest dataset window. A sanity check, not a "
        "rigorous metric. Matching adds noise."
    )

    FAMOUS_FIRES = [
        {"name": "Camp Fire",       "date": "2018-11-08", "lat": 39.81, "lon": -121.44, "state": "CA"},
        {"name": "Dixie Fire",      "date": "2021-07-13", "lat": 39.88, "lon": -121.39, "state": "CA"},
        {"name": "Thomas Fire",     "date": "2017-12-04", "lat": 34.40, "lon": -119.14, "state": "CA"},
        {"name": "Tubbs Fire",      "date": "2017-10-08", "lat": 38.61, "lon": -122.62, "state": "CA"},
        {"name": "Carr Fire",       "date": "2018-07-23", "lat": 40.65, "lon": -122.63, "state": "CA"},
        {"name": "Woolsey Fire",    "date": "2018-11-08", "lat": 34.23, "lon": -118.71, "state": "CA"},
        {"name": "Marshall Fire",   "date": "2021-12-30", "lat": 39.96, "lon": -105.24, "state": "CO"},
        {"name": "Bootleg Fire",    "date": "2021-07-06", "lat": 42.60, "lon": -121.38, "state": "OR"},
        {"name": "August Complex",  "date": "2020-08-16", "lat": 39.78, "lon": -122.67, "state": "CA"},
        {"name": "Glass Fire",      "date": "2020-09-27", "lat": 38.56, "lon": -122.49, "state": "CA"},
    ]

    if full_preds is None:
        st.info("`full_predictions.csv` not found. Run the updated `prepare_demo_data.py` to generate it.")
    else:
        MAX_DIST_DEG, MAX_DAYS = 0.5, 30
        results = []
        for fire in FAMOUS_FIRES:
            fire_date = pd.Timestamp(fire['date'])
            dist = np.sqrt((full_preds['latitude'] - fire['lat'])**2 +
                           (full_preds['longitude'] - fire['lon'])**2)
            day_diff = (full_preds['event_date'] - fire_date).abs().dt.days
            candidates = full_preds[(dist <= MAX_DIST_DEG) & (day_diff <= MAX_DAYS)].copy()
            if len(candidates) == 0:
                results.append({"Fire": fire['name'], "State": fire['state'], "Date": fire['date'],
                                "Model probability": "—", "Prediction": "Not in dataset"})
                continue

            candidates['score'] = np.sqrt((candidates['latitude'] - fire['lat'])**2 +
                                           (candidates['longitude'] - fire['lon'])**2)
            # Prefer an actual fire-positive window if one exists within tolerance —
            # otherwise the nearest-by-coordinate match may land on a negative or
            # pre-ignition window.
            fire_candidates = candidates[candidates['label'] == 1]
            pool = fire_candidates if len(fire_candidates) > 0 else candidates
            best = pool.loc[pool['score'].idxmin()]

            predicted_fire = best['predicted_proba'] >= PRODUCTION_THRESHOLD
            results.append({
                "Fire": fire['name'], "State": fire['state'], "Date": fire['date'],
                "Model probability": f"{best['predicted_proba']:.1%}",
                "Prediction": "Predicted Fire" if predicted_fire else "Predicted No Fire"
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch', hide_index=True)
        found = results_df[results_df["Prediction"] != "Not in dataset"]
        caught = found[found["Prediction"] == "Predicted Fire"]
        if len(found) > 0:
            st.metric("Flagged by the model", f"{len(caught)} / {len(found)}")
        st.caption(
            "Each fire is matched to its nearest ignition window in the dataset, then run through the model."
        )

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "Background: “Wildfire” — the 2013 Alder Fire, Yellowstone National Park. "
    "Photo by Mike Lewelling, National Park Service (public domain), via "
    "[NPS Climate Change Response on Flickr](https://www.flickr.com/photos/npsclimatechange/14503287131/)."
)
