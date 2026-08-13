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
   cell, producing 1,683 labeled training examples across 495 spatial blocks
   (~20km each). An XGBoost classifier trained on the enriched features
   (population, building/pole/road neighborhood density, VIIRS nighttime
   lights) achieves **spatial test AUC 0.857, accuracy 77%** — evaluated
   with spatial-block holdout (entire ~20km blocks held out of training),
   not random K-fold: 62% of DHS clusters sit within 5km of another cluster,
   so random splitting would let near-duplicate points leak across
   train/test and overstate accuracy. This is up from an initial AUC 0.72
   baseline using only raw binary OSM flags.

   `roads_within_2000m` and `dist_to_nearest_road_m` (pulled from OSM via
   Overpass API) rank as the 2nd and 5th most important features — roads are
   one of the strongest electrification predictors in the literature.
   `ntl_radiance` (VIIRS annual nighttime-lights composite, see below) adds a
   further, more moderate gain — 7th of 12 features by importance, likely
   capped by the fact that 95.5% of cells (including most rural DHS training
   points) have literally zero recorded light, limiting its power to
   discriminate within that dominant zero-band even though it clearly
   separates known-urban from known-rural ground truth. Net effect across
   all enrichment: Nairobi/Kiambu/Mombasa mean predicted probability rose
   from ~0.68 to ~0.73-0.74; Turkana/Wajir fell to ~0.14-0.16, sharpening the
   gap in the correct direction against real CRA county rates.

   **VIIRS nighttime lights**: the original project plan flagged this as
   blocked pending Earth Engine or NOAA/EOG account access. We registered a
   free EOG account and pulled the 2025 annual VNL v2.2 composite
   (`average_masked`, background already zeroed) directly. The global file
   is a ~11.6GB gzip-compressed BigTIFF with its IFD (metadata) stored at
   the very end -- a "data first, metadata last" streaming layout -- so
   rather than decompressing the whole thing (which exceeded available
   local disk space), `src/crop_viirs.py` inspects the raw header, confirms
   the pixel layout is fully predictable from geometry alone (86400x33600,
   row-major, uncompressed float32, 180W-180E/75N-65S), and stream-decompresses
   sequentially through only Kenya's row range (~5.8GB of skip + ~21.5MB
   kept), discarding everything else without ever writing the full raster
   to disk.
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
  add_road_features.py     # pulls OSM roads via Overpass, adds road-distance/density
  crop_viirs.py             # streams Kenya's slice out of the global VIIRS composite
  add_viirs_features.py     # samples nighttime radiance onto the grid
  build_training_set.py     # spatial-join DHS clusters to grid cells -> labels
  train_model.py             # trains + compares LR / RF / XGBoost with spatial-block CV
  predict_and_rank.py        # predicts nationwide, filters + region-normalized ranking
  engineering_specs.py       # sizes solar/battery/cost for top sites
  build_map.py                # renders the interactive HTML map (rasterized choropleth)
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

Run in order: `enrich_features.py` → `add_road_features.py` →
`crop_viirs.py` → `add_viirs_features.py` → `build_training_set.py` →
`train_model.py` → `predict_and_rank.py` → `engineering_specs.py` →
`build_map.py`. (`crop_viirs.py` needs the global VIIRS composite downloaded
to `data/raw/viirs/` first -- requires a free EOG account, see below.)

## Known limitations (honest accounting)

- **`building_count` / `pole_count` in this OSM extract are binary presence
  flags per cell, not true counts.** We work around this with KDTree
  neighborhood density counts, but a real building-footprint count would
  sharpen this further.
- **AUC 0.857 / 77% accuracy is good but not great** — roughly 1 in 4-5
  cells could be misclassified. With nighttime lights now integrated, the
  next likely lever is more/denser ground-truth labels (currently only
  1,683 DHS-labeled cells nationwide) or finer building-footprint data than
  this OSM extract's binary presence flags.
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
- OpenStreetMap major road network (trunk/primary/secondary/tertiary), pulled
  live via Overpass API
- WorldPop Kenya 2020 population count, 1km resolution (data.worldpop.org)
- VIIRS annual nighttime lights composite, 2025 v2.2 (Earth Observation
  Group, Payne Institute, eogdata.mines.edu) — free account required
- Kenya Commission on Revenue Allocation — county electrification rates
  (not yet integrated as a model feature, held for future work)
