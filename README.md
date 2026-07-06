# Wildfire Ignition Prediction

Extends [FireCastRL](https://arxiv.org/abs/2601.14238) (GRIDMET weather + IRWIN fire records, 2013–2025) with human-activity features — population density, land cover, holidays, and utility-corridor proximity — to test the paper's own stated limitation: ~84% of US wildfires are human-caused, and a weather-only model can't see that.

**Full writeup:** [`WRITEUP.md`](./WRITEUP.md)

## Headline Finding

**Adding population density does NOT improve test F2, despite ranking #2 of 46 features in permutation importance and showing a clean directional effect in SHAP.**

A controlled ablation on the same 46 selected features:

| Feature set | Test F2 |
|---|---|
| Full (46 features, incl. human-activity) | 0.602 |
| Same set, human-activity features removed | **0.610** |

Yet SHAP shows population density with the widest, cleanest spread of any feature — high values consistently push predictions toward "fire," low values push away:
<p align="center">
  <img width="390" height="639" alt="image" src="https://github.com/user-attachments/assets/da24ec4d-af3a-4cad-9dcd-d5d256e30360" />
</p>

**The reconciliation:** permutation importance and SHAP measure how much a *specific trained model* relies on a feature. The ablation asks whether that feature is *necessary* to reach the model's performance ceiling. Random Forest has enough capacity to recover equivalent signal from weather-variable interactions (precipitation and dryness patterns likely covary with WUI proximity and seasonal human activity), so the model reroutes around population density at no net cost. Individual importance does not guarantee irreplaceability — a concrete illustration of feature redundancy in tree ensembles. See [`WRITEUP.md`](./WRITEUP.md) for the full argument.

## Results

Performance from 5-fold CV with per-fold threshold tuning (precision ≥ 0.4 constraint):

| Model | Precision | Recall | F2 |
|---|---|---|---|
| Logistic Regression | 0.405 ± 0.004 | 0.315 ± 0.008 | 0.330 ± 0.007 |
| **Random Forest** | 0.404 ± 0.004 | 0.696 ± 0.005 | **0.608 ± 0.004** |
| XGBoost (tuned) | 0.403 ± 0.003 | 0.664 ± 0.006 | 0.588 ± 0.005 |

Random Forest wins across all 5 folds against XGBoost (paired comparison, 0.017–0.022 F2 margin per fold — not noise).

## Structure

```
01_preprocessing_and_feature_engineering.ipynb   # Raw CSV -> clean, window-level feature table
02_modelling_and_evaluation.ipynb                # Feature selection, 3-model comparison, SHAP, ablation
wildfire_EDA.ipynb                               # Exploratory analysis, incl. population density deep-dive
WRITEUP.md                                       # Full methodology, findings, limitations
figures/                                         # SHAP summary, ablation chart, EDA plots
```

## Data Sources

- Weather/fire: [Kaggle — firecastrl/us-wildfire-dataset](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)
- Population density: [NASA SEDAC GPWv4](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11)
- Land cover: [USGS NLCD](https://www.mrlc.gov/data)
- Transmission lines: [HIFLD](https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/)

## Setup

```bash
conda create -n wildfire python=3.11
conda activate wildfire
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn xgboost rasterio geopandas pyproj shap
pip install holidays
```

Run notebooks in order: `01_preprocessing...` produces `wildfire_features_full.csv`, which `02_modelling...` consumes.

## Status

Modeling, explainability, and the population-density ablation are complete. Streamlit demo app scoped, not yet built.
