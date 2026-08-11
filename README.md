# Kenya Electrification Gap Analysis & Microgrid Site Ranking

Predicting where in Kenya people lack electricity access, and ranking the best
locations for solar minigrid deployment — trained and validated on real
household survey data, applied at 500m resolution across the entire country.

## The problem

Kenya's national electrification rate looks decent on paper (~79% in 2023),
but that hides deep unevenness: Nairobi sits at 96.7% access while Turkana is
at 8.8%. A landmark study (Lee, Miguel & Wolfram, 2016) found that in rural
Western Kenya, only 5% of households were connected to electricity even
though half of the unconnected ones were within 200 meters of a live power
line. Grid infrastructure existing isn't the same as people being connected
to it.

Grid extension is expensive — tens of thousands of dollars per km — and
doesn't pay back for small, scattered, remote populations. Solar microgrids
are the economically rational alternative beyond a certain distance from the
grid. This project builds a data-driven tool to find exactly where those
areas are.

## Approach

1. **Ground truth**: Kenya DHS 2022 Household Recode survey — 37,911 real
   households across 1,691 GPS-located clusters, each with a `hv206`
   ("has electricity") response. This is genuine field survey data, not a
   proxy.
2. **Features**: OpenStreetMap-derived building presence, distance to nearest
   building, power pole presence, and distance to nearest power pole — for
   every 500m cell across Kenya (2.91M cells, a perfect 1500×1940 grid).
3. **Model**: DHS clusters are spatially matched to their nearest 500m grid
   cell, producing 1,683 labeled training examples. A Random Forest
   classifier trained on the OSM features predicts electrification
   probability, achieving **AUC 0.72** (5-fold cross-validated) against real
   survey outcomes — beating logistic regression and XGBoost baselines.
4. **National prediction**: The trained model is applied to all 2.91M grid
   cells to produce a nationwide electrification probability map.
5. **Microgrid site ranking**: Cells are filtered to those that are likely
   underserved (probability ≤ 0.35), populated, and far enough from existing
   grid infrastructure (≥500m from the nearest known power pole) that grid
   extension is uneconomical. Remaining candidates are scored by a weighted
   suitability index (40% settlement density, 30% building density, 30%
   distance from grid) and spatially deduplicated so adjacent grid cells in
   the same settlement don't all rank as separate "sites."
6. **Engineering specs**: For the top 100 distinct sites, the pipeline
   estimates household count (via nearby building density within a 1km
   service radius), daily energy demand, solar PV array size, battery
   capacity, recommended topology (AC three-phase vs. DC cluster), and
   capital cost — using published Kenya minigrid benchmarks (GIZ/AfDB).
7. **Interactive map**: A 500m-resolution raster of predicted electrification
   probability across Kenya, with the top-ranked microgrid sites overlaid and
   clickable for their full spec sheet.

## Repo layout

```
src/
  build_training_set.py   # spatial-join DHS clusters to grid cells -> labels
  train_model.py           # trains + compares LR / RF / XGBoost, saves best
  predict_and_rank.py      # predicts nationwide, filters + ranks + dedupes sites
  engineering_specs.py     # sizes solar/battery/cost for top sites
  build_map.py              # renders the interactive HTML map
data/
  raw/                      # DHS + OSM source data (gitignored, from Google Drive)
  processed/                # training_set.csv, DHS cluster labels
output/
  electrification_model.joblib
  model_metrics.json
  ranked_microgrid_sites_deduped.csv
  microgrid_engineering_specs.csv
  kenya_electrification_map.html
```

Run in order: `build_training_set.py` → `train_model.py` →
`predict_and_rank.py` → `engineering_specs.py` → `build_map.py`.

## Known limitations (honest accounting)

- **`building_count` / `pole_count` in this OSM extract are binary presence
  flags per cell, not true counts.** Household estimates and density scoring
  work around this using neighborhood search radii rather than per-cell
  counts — a real building-footprint count would sharpen this considerably.
- **No population raster.** Household counts are inferred from building
  density, not census data.
- **AUC 0.72 is decent but modest.** The single biggest likely improvement is
  adding VIIRS nighttime lights as a feature (planned, blocked on Earth
  Engine API access).
- **County-level CRA electrification rates aren't currently joined in** — no
  county boundary shapefile was available in the downloaded dataset.
- DHS cluster GPS coordinates are randomly displaced by DHS for privacy
  (up to 2km urban, 5-10km rural), which caps the precision of the
  cell-level labels used for training.

## Data sources

- Kenya DHS 2022 Household Recode + GPS datasets (DHS Program, ICF)
- OpenStreetMap Kenya extract (via Geofabrik) — building footprints, power
  infrastructure
- Kenya Commission on Revenue Allocation — county electrification rates
  (not yet integrated as a model feature, held for future work)
