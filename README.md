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
   counting), continuous neighborhood density (buildings within 1km/3km,
   poles within 5km/10km via KDTree), OSM road distance/density, VIIRS
   nighttime lights, and **Google Open Buildings** real building counts
   (satellite ML detection, see below) — for every 500m cell across Kenya
   (2.91M cells, a perfect 1500×1940 grid).
3. **Model**: DHS clusters are spatially matched to their nearest 500m grid
   cell, producing 1,683 labeled training examples across 495 spatial blocks
   (~20km each). An XGBoost classifier trained on the enriched features
   achieves **spatial test AUC 0.886, accuracy 80.6%** — evaluated with
   spatial-block holdout (entire ~20km blocks held out of training), not
   random K-fold: 62% of DHS clusters sit within 5km of another cluster, so
   random splitting would let near-duplicate points leak across train/test
   and overstate accuracy. This is up from an initial AUC 0.72 baseline
   using only raw binary OSM flags — see the progression below.

   | Stage | Spatial test AUC | Accuracy |
   |---|---|---|
   | Binary OSM building/pole flags only | 0.72 | ~66% |
   | + real WorldPop population, neighborhood density | 0.84 | ~77% |
   | + OSM road distance/density | 0.855 | ~78% |
   | + VIIRS nighttime lights | 0.857 | ~77% |
   | + **Google Open Buildings** (real building counts) | **0.886** | **80.6%** |

   `google_buildings_within_1000m` is the single most important feature by a
   wide margin (0.270, more than double the next), confirming the hypothesis
   that OSM's building data was the weakest link: OSM only flags 6.7% of
   Kenya's cells as having a building (community mapping is patchy,
   especially outside urban areas), while Google Open Buildings — satellite
   imagery run through an ML detector, no community mapping required — flags
   21.6% of cells with actual counts, 31.5M buildings total nationwide. Real,
   complete building data mattered more than any other single feature added.

   Nairobi/Kiambu/Mombasa mean predicted probability is now ~0.68-0.74;
   Turkana/Wajir ~0.11-0.17 — correctly and increasingly sharply separated in
   the direction real CRA county rates say they should be.

   **90%+ accuracy is very unlikely achievable with this approach and we
   don't claim it's a near-term target.** DHS survey coordinates are randomly
   displaced up to 2-10km for privacy, so even a perfect model is learning
   from labels that don't precisely match the 500m cell they're attached to.
   Published academic work in this space typically tops out around AUC
   0.85-0.90 / accuracy in the low-to-mid 80s; higher claims on this kind of
   task are usually a red flag for leakage, not genuine skill. 0.886 sits
   near the top of that realistic range.

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

   **Google Open Buildings**: fully public GCS bucket
   (`storage.googleapis.com/open-buildings-data`), no login required.
   `notebooks/kenya_open_buildings.ipynb` (run in Colab, since the ~30M-point
   download needs disk headroom this machine didn't have) finds the 11 S2
   level-4 tiles covering Kenya, downloads each tile's building point data,
   and bins ~31.5M building locations onto our exact 500m grid. The result
   (`kenya_google_building_density.csv`) is merged in via **nearest-neighbor
   spatial join, not an exact lon/lat merge** — both grids use the same
   definition but were computed independently (locally vs. in Colab) via
   `np.arange`/rounding, and tiny floating-point representation differences
   meant an exact-equality merge matched only 21 of 2.91M rows on one attempt
   (a real bug caught by checking the merged output's nonzero fraction
   against the source file's, not just checking the columns existed).
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
notebooks/
  kenya_open_buildings.ipynb          # run in Colab: fetches Google Open Buildings for Kenya
  colab_open_buildings_snippet.py     # same code, plain .py reference
src/
  enrich_features.py        # samples WorldPop population + KDTree neighborhood density
  add_road_features.py      # pulls OSM roads via Overpass, adds road-distance/density
  crop_viirs.py              # streams Kenya's slice out of the global VIIRS composite
  add_viirs_features.py      # samples nighttime radiance onto the grid
  add_google_buildings.py    # merges Google Open Buildings density (nearest-neighbor join)
  build_training_set.py      # spatial-join DHS clusters to grid cells -> labels
  train_model.py              # trains + compares LR / RF / XGBoost with spatial-block CV
  predict_and_rank.py         # predicts nationwide, filters + region-normalized ranking
  engineering_specs.py        # sizes solar/battery/cost for top sites
  build_map.py                 # renders the interactive HTML map (rasterized choropleth)
data/
  raw/                        # DHS + OSM + WorldPop + VIIRS + Google Buildings (gitignored)
  processed/                  # enriched_features.csv, training_set.csv (gitignored)
output/
  electrification_model.joblib
  model_metrics.json
  ranked_microgrid_sites_deduped.csv
  microgrid_engineering_specs.csv
  electrification_raster.png
  kenya_electrification_map.html
```

Run in order: `enrich_features.py` → `add_road_features.py` →
`crop_viirs.py` → `add_viirs_features.py` → run
`notebooks/kenya_open_buildings.ipynb` in Colab and place its output CSV in
`data/raw/google_buildings/` → `add_google_buildings.py` →
`build_training_set.py` → `train_model.py` → `predict_and_rank.py` →
`engineering_specs.py` → `build_map.py`. (`crop_viirs.py` needs the global
VIIRS composite downloaded to `data/raw/viirs/` first — requires a free EOG
account, see below.)

## Known limitations (honest accounting)

- **`building_count` / `pole_count` in the OSM extract are binary presence
  flags per cell, not true counts** — largely superseded now by real Google
  Open Buildings counts, but `pole_count` (grid infrastructure) has no
  equivalent alternative source and remains sparse (see below).
- **AUC 0.886 / 80.6% accuracy is good, near the realistic ceiling for this
  kind of DHS-validated approach, but not perfect** — roughly 1 in 5 cells
  could still be misclassified. The next likely lever is more/denser
  ground-truth labels (currently only 1,683 DHS-labeled cells nationwide) —
  diminishing returns from here likely require genuinely new label sources
  (a newer/larger household survey), not more satellite features.
- **OSM power-pole data is extremely sparse: only 0.06% of Kenya's 2.91M
  cells (1,677) have a mapped pole.** `dist_to_nearest_pole_m` is therefore
  really "distance to nearest of 1,677 scattered points," not distance to
  the real grid -- and it initially misclassified 1,559 candidate cells as
  underserved-and-far-from-grid despite those cells showing detectable VIIRS
  night light (direct evidence they're already electrified). Fixed with a
  hard override: any candidate with `ntl_radiance > 0` is excluded regardless
  of what the model or the pole-distance proxy says, since observed light is
  stronger ground truth than either. None of the final 40 ranked sites were
  affected by this specific bug, but it's a real latent risk in the
  underlying candidate pool worth flagging prominently, and a caution against
  trusting `dist_to_nearest_pole_m` (or the visible line/banding patterns it
  creates in the raw prediction grid) as precise "distance to the real grid."
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
- Google Open Buildings v3 (Google Research, storage.googleapis.com/open-buildings-data)
  — public, no account required
- Kenya Commission on Revenue Allocation — county electrification rates
  (not yet integrated as a model feature, held for future work)
