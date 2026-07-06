# Wildfire Ignition Prediction: Does Population Density Improve Forecasts?

**Extending FireCastRL with human-activity features — and finding, via a controlled ablation, that adding population density does not improve test F2 despite it ranking #2 in permutation importance.**

## Motivation

[FireCastRL](https://arxiv.org/abs/2601.14238) (Mathur et al.) built a wildfire ignition classifier from GRIDMET weather data and IRWIN fire records, reporting strong results with a CNN-BiLSTM (73.1% accuracy) but flagged an explicit limitation: **~84% of US wildfire ignitions are human-caused**, and their weather-only model has no way to see that. The paper's own stated future work was to add population density, land use, utility corridors, and holiday/temporal patterns.

This project does that — but the more interesting story turned out to be *why* it works, not just *whether* it does.

## Dataset

Same source as FireCastRL: GRIDMET weather (2013–2025) joined to IRWIN fire incident records, via the [Kaggle `firecastrl/us-wildfire-dataset`](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset) release. 9.5M rows, which turned out to *not* be independent daily observations.

**Discovery:** the dataset is built from 126,799 **75-day windows** — 60 days before + the event day + 15 days after — each anchored to either a real IRWIN ignition (50,720 events) or a synthetic negative sample (76,080, in three tiers: "far," "near," and "yearly" relative to real fires). A window's label is `No` until (if ever) ignition occurs, then `Yes` for the remainder of the window. This structure isn't documented on the Kaggle page itself; it was reverse-engineered from the paper and the author's own example notebook.

This mattered enormously for methodology: naive row-level daily classification would train on **in-fire weather** (post-ignition days are trivially different from pre-ignition days) and would risk **leaking across train/test splits** if the fixed 75-row block structure wasn't respected.

## Data Quality Issues Found (and Why They Mattered)

Real, verified bugs in the source data, all confirmed present in the *raw, unmodified* CSV (not artifacts of this project's cleaning):

1. **Contradictory labels.** ~500–700 windows had the exact same (lat, lon, date) appear twice with opposite `Wildfire` labels — two independently-sampled windows colliding on identical coordinates and overlapping dates. Whole windows dropped (not individual rows, to preserve the 75-row block structure).
2. **Duplicate rows within windows.** A smaller number of windows had one day double-counted and a different day silently dropped, desyncing the 75-row structure. Also dropped whole.
3. **Sentinel value re-corruption.** GRIDMET encodes missing data as `32767`, not `NaN`. This was masked correctly at the start of the project, but was twice silently reintroduced mid-pipeline — once via a stale dataframe reload, once via a `dropna()` call that (correctly) does nothing to a sentinel value that isn't actually `NaN`. Both were caught via correlation-matrix red flags (physically nonsensical variables correlating at r=0.999) before any results were trusted.

After cleaning: **125,731 valid windows**, class balance **26.0% fire / 74.0% no-fire** — notably different from the paper's implied ~40/60 split (50,720 / 126,800). This gap was checked against the *raw* data too (25.25% raw positive rate) and confirmed not to be an artifact of this project's cleaning — the paper's headline ratio appears to describe sampling *design intent*, not the realized contents of the released CSV.

## Feature Engineering

Each window's **pre-event 60 days only** (excluding the ignition day and 15-day tail, to avoid leaking in-fire weather) is collapsed into `mean / max / min / std / linear-trend-slope` per weather variable — turning a sequence-modeling problem into a standard tabular ML problem, with no deep learning required.

**Weather variables** (15): precipitation, humidity (max/min), specific humidity, solar radiation, temperature (min/max), wind speed, burning index, fuel moisture (100hr/1000hr), energy release component, evapotranspiration (actual/potential), vapor pressure deficit.

**Human-activity extensions:**
- **Population density** — NASA SEDAC GPWv4, sampled at three snapshot years (2010/2015/2020) and linearly interpolated/extrapolated per-point to each event's actual year, rather than using a crude nearest-year step function.
- **Land cover** — USGS NLCD, grouped into 8 categories (developed, forest, shrubland, grassland, agriculture, wetland, water, barren).
- **Holiday/temporal flags** — weekend, US federal holiday, July 4th week (fireworks-related human-ignition risk).
- **Distance to nearest electric transmission line** (stretch goal) — HIFLD infrastructure data, testing the utility-caused-ignition pathway.

## Feature Selection

87 engineered features → **permutation importance**, scored directly on F2 (the project's actual metric of interest), chosen over RFE/SFS (too expensive at this scale, unstable under multicollinearity) and Random Forest's built-in MDI importance (known to bias toward high-cardinality/correlated features). Reduced to **46 features**, validated via 5-fold cross-validation to cost no meaningful performance versus the full set (F2 0.529 ± 0.006 vs. 0.524 ± 0.006).

## Modeling

Train/validation/test split: 60/20/20, stratified. Thresholds tuned on validation only, constrained to **precision ≥ 0.4** (an unconstrained F2 sweep found a threshold with 95% recall but 29% precision — mechanically optimal but an alert-fatigue failure mode, not a usable operating point). Test set touched exactly once per model.

Final performance (5-fold cross-validation, mean ± std, evaluated with per-fold threshold tuning on an internal validation split to avoid leakage into the fold's held-out set):

| Model | Precision | Recall | F2 |
|---|---|---|---|
| Logistic Regression | 0.405 ± 0.004 | 0.315 ± 0.008 | 0.330 ± 0.007 |
| **Random Forest** | 0.404 ± 0.004 | 0.696 ± 0.005 | **0.608 ± 0.004** |
| XGBoost (tuned) | 0.403 ± 0.003 | 0.664 ± 0.006 | 0.588 ± 0.005 |

Random Forest is the clear winner. A paired comparison across folds — valid because all three models saw identical fold splits — shows Random Forest beats XGBoost in **all 5 of 5 folds** by a consistent 0.017–0.022 F2 margin, so the ~0.02 gap is not run-to-run noise. Recall nearly doubled moving from logistic regression to Random Forest (32% → 70%), confirming non-linear interaction-capturing capacity, not simply "trying more models," is what mattered for this problem.

## Core Finding: Population Density Doesn't Improve the Model — And Yet It Matters

**The headline result, in one sentence:** adding population density does **not** improve test F2, despite ranking #2 of 46 features in permutation importance and showing a clean, monotonic effect in SHAP.

That sentence looks like a contradiction until you notice it's answering two different questions with two different tools. Here's the full picture:

### The counterintuitive part: a controlled ablation shows no net accuracy gain

Training Random Forest on the identical 46 selected features, once with and once without the human-activity subset (`pop_density`, `lc_agriculture`) removed — same weather composition both times, no confound from independently-run feature selection — gives:

| Feature set | Test F2 |
|---|---|
| Full (46 features, incl. human-activity) | 0.602 |
| Same set, human-activity features removed | **0.610** |

Removing population density and land cover did not hurt performance — if anything, it's marginally higher without them.

### But three independent methods show population density *is* meaningful

1. **Permutation importance on Random Forest** ranks `pop_density` **#2 of 46 features** (≈0.064), essentially tied with the top weather variable. Yet in **logistic regression**, it ranks near the bottom (≈0.008), effectively invisible. This gap is itself informative: the feature's effect appears interaction-driven (e.g., a dry-weather or holiday effect that matters more when people are actually present nearby), which only a model capable of learning interactions can pick up.
2. **SHAP** (TreeExplainer, on a depth-capped Random Forest variant for computational tractability) confirms a clean, monotonic pattern: high population density consistently pushes predicted risk up, low density consistently pushes it down — the widest, cleanest spread of any feature in the summary plot. A drill-down found that values driving strong positive contributions cluster at a median of ~688/km² — suburban/exurban, wildland-urban-interface range, not dense urban-core density (typically 5,000–15,000+/km²) — supporting a genuine human-ignition-proximity mechanism over a pure reporting-density artifact, though reporting bias in the underlying IRWIN data can't be fully ruled out.
3. **Raw EDA** shows the same pattern directly in the data, no model involved: median population density in fire-outcome windows (~10/km²) is roughly 3× higher than in no-fire windows (~3.2/km²).

### Reconciling the two: individual importance ≠ irreplaceability

Permutation importance and SHAP describe how much a *specific trained model* relies on a feature. The ablation asks whether that feature is *necessary* for the model to reach its performance ceiling. Random Forest evidently has enough capacity to recover equivalent signal through weather-variable interactions and correlations (precipitation and dryness patterns plausibly covary with WUI proximity and seasonal human activity), so removing population density lets the model reroute around it at no net cost.

**Honest final conclusion:** population density is a real, individually meaningful signal that the model demonstrably uses, with a sensible and well-evidenced directional effect — but it does **not** provide a demonstrated net accuracy improvement over what tree-based models already infer from weather alone in this pipeline. This is a concrete illustration of feature redundancy in tree ensembles, and a caution against equating high permutation importance with irreplaceability.

## Honest Limitations

- **Land cover carried little independent signal.** Only one category (`lc_agriculture`) survived feature selection, and it ranked near-zero in Random Forest's permutation importance. Likely explanation: a single 2021 snapshot applied across a 2013–2025 date range is too coarse, or the categories are redundant with fuel-moisture/vegetation-linked weather variables already in the model.
- **Utility corridors showed real but marginal, redundant signal.** Distance to nearest transmission line ranked 43rd of 47 features (clearing the noise threshold, so not pure noise) but did not improve overall model F2 (0.606 vs. 0.610 without it) — likely redundant with population density rather than adding independent information. Not included in the final production feature set for this reason.
- **Class balance diverges from the source paper** (26/74 measured vs. ~40/60 implied) — investigated and confirmed not an artifact of this project's cleaning; likely reflects a gap between the paper's sampling design and the realized contents of the released CSV.
- **Reporting bias caveat** on the population density finding, as above — can't be fully separated from a true ignition-risk mechanism with this dataset alone.

## Repository Structure

```
01_preprocessing_and_feature_engineering.ipynb   # Raw CSV -> clean, window-level feature table
02_modelling_and_evaluation.ipynb                # Feature selection, 3-model comparison, SHAP
wildfire_EDA.ipynb                               # Exploratory analysis, including population density deep-dive
wildfire_features_full.csv                       # Output of notebook 1, input to notebook 2
```

## Future Work

- A proper spatiotemporal reworking of utility-corridor and land-cover features (finer-grained/multi-year land cover snapshots) before concluding they don't matter.
- An interactive demo (in progress) allowing exploration of individual predictions with live SHAP explanations.
- Extending Stage 2/3 of the original project plan: severity/size regression, and a continuously-updating operational risk pipeline.
