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
2. **Features**: OpenStreetMap-derived building/pole presence and distance,
   real WorldPop 2020 population (sampled per cell, apportioned so per-cell
   values sum back to Kenya's true ~54.8M population instead of quadruple-
   counting), and continuous neighborhood density (buildings within 1km/3km,
   poles within 5km/10km via KDTree) — for every 500m cell across Kenya
   (2.91M cells, a perfect 1500×1940 grid).
3. **Model**: DHS clusters are spatially matched to their nearest 500m grid
   cell, producing 1,683 labeled training examples. A Random Forest
   classifier trained on the enriched features predicts electrification
   probability, achieving **AUC 0.84** (5-fold cross-validated) against real
   survey outcomes — up from an initial AUC 0.72 baseline using only raw OSM
   binary flags, beating logistic regression and XGBoost.
4. **National prediction**: The trained model is applied to all 2.91M grid
   cells to produce a nationwide electrification probability map.
5. **Microgrid site ranking**: Cells are filtered to those that are likely
   underserved (probability ≤ 0.35), genuinely populated (real WorldPop count
   ≥ 5, not just "has an OSM-mapped building" — OSM mapping completeness is
   very uneven across Kenya and would otherwise systematically exclude
   OSM-sparse but inhabited arid counties), and far enough from existing grid
   infrastructure (≥500m from the nearest known power pole) that grid
   extension is uneconomical.

   Sites are then selected **one best candidate per ~55km region**, with
   suitability (40% population, 30% building density, 30% distance from grid)
   normalized *within* each region rather than nationally. A purely national
   ranking is dominated by population and always crowns the same
   high-density western settlements, completely starving the actual
   lowest-electrification counties (Turkana 2.4%, West Pokot 2.0%, Mandera
   2.5% per CRA data) of any representation just because they're less densely
   populated. Region-relative scoring instead asks "what's the best
   opportunity within this region," so every underserved region of the
   country gets a shot — and when there are more qualifying regions than
   site slots, the neediest regions (lowest mean electrification probability)
   are prioritized first.
6. **Engineering specs**: For the top 150 sites, the pipeline sums real
   population within a 1.5km service radius (not an OSM-building-count
   proxy) to estimate households served, then computes daily energy demand,
   solar PV array size, battery capacity, recommended topology (AC
   three-phase vs. DC cluster), and capital cost — using published Kenya
   minigrid benchmarks (GIZ/AfDB).
7. **Interactive map**: A rasterized 500m-resolution choropleth of predicted
   electrification probability across Kenya (red→green), with ranked
   microgrid sites overlaid and color-coded by how urgent their surrounding
   region's need is, each clickable for its full engineering spec sheet.

## Repo layout

```
src/
  enrich_features.py       # samples WorldPop population + KDTree neighborhood density
  build_training_set.py    # spatial-join DHS clusters to grid cells -> labels
  train_model.py            # trains + compares LR / RF / XGBoost, saves best
  predict_and_rank.py       # predicts nationwide, filters + region-normalized ranking
  engineering_specs.py      # sizes solar/battery/cost for top sites
  build_map.py               # renders the interactive HTML map (rasterized choropleth)
data/
  raw/                       # DHS + OSM + WorldPop source data (gitignored)
  processed/                 # enriched_features.csv, training_set.csv (gitignored)
output/
  electrification_model.joblib
  model_metrics.json
  ranked_microgrid_sites_deduped.csv
  microgrid_engineering_specs.csv
  electrification_raster.png
  kenya_electrification_map.html
```

Run in order: `enrich_features.py` → `build_training_set.py` →
`train_model.py` → `predict_and_rank.py` → `engineering_specs.py` →
`build_map.py`.

## Known limitations (honest accounting)

- **`building_count` / `pole_count` in this OSM extract are binary presence
  flags per cell, not true counts.** We work around this with KDTree
  neighborhood density counts, but a real building-footprint count would
  sharpen this further.
- **AUC 0.84 is good but not great.** The single biggest likely further
  improvement is VIIRS nighttime lights — we attempted to pull these
  directly (bypassing Google Earth Engine) but NOAA/EOG's hosting requires
  an authenticated account login, so this remains blocked pending
  registration.
- **County-level CRA electrification rates aren't currently joined in as a
  model feature** — no county boundary shapefile was available locally, and
  public sources we tried (geoBoundaries API, GitHub mirrors) were either
  unreachable or Git-LFS-gated. The region-normalized ranking approach
  (55km grid bins) sidesteps this without needing county polygons, but a
  real county join would let the model use the CRA ground-truth rates
  directly as a feature too.
- **Region bin size (0.5°, ~55km) is a simplification of real administrative
  or economic catchment boundaries** — a production version should bin by
  actual county/sub-county boundaries or grid-extension cost surfaces
  instead of a fixed-size lat/lon grid.
- DHS cluster GPS coordinates are randomly displaced by DHS for privacy
  (up to 2km urban, 5-10km rural), which caps the precision of the
  cell-level labels used for training.
- WorldPop population is a 2020 modeled estimate at 1km resolution,
  resampled to our 500m grid — it's real census-calibrated data, but not a
  literal household count.

## Data sources

- Kenya DHS 2022 Household Recode + GPS datasets (DHS Program, ICF)
- OpenStreetMap Kenya extract (via Geofabrik) — building footprints, power
  infrastructure
- WorldPop Kenya 2020 population count, 1km resolution (data.worldpop.org)
- Kenya Commission on Revenue Allocation — county electrification rates
  (not yet integrated as a model feature, held for future work)
