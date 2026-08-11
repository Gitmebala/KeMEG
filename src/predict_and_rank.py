"""
Apply the trained model to all ~200k grid cells nationwide, then filter and
rank underserved cells as microgrid candidate sites.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURES_PATH = "data/raw/full_kenya_500m_features.csv"
MODEL_PATH = "output/electrification_model.joblib"
PREDICTIONS_OUT = "output/national_predictions.csv"
RANKED_SITES_OUT = "output/ranked_microgrid_sites.csv"
DEDUPED_SITES_OUT = "output/ranked_microgrid_sites_deduped.csv"

# Microgrid candidate filters
# NOTE: building_count in this dataset is a binary presence flag (0/1) per 500m cell,
# not a true structure count -- so "populated" means building_count >= 1.
MAX_ELECTRIFICATION_PROB = 0.35   # likely underserved
MIN_BUILDING_COUNT = 1            # populated, not empty wilderness
MIN_DIST_TO_GRID_M = 500          # grid extension not economical beyond this

# Deduplication: adjacent 500m cells in the same settlement would otherwise all
# rank as top candidates. We greedily pick the best-scoring cell in each
# neighborhood and suppress other candidates within MIN_SITE_SPACING_M of it,
# so ranked sites represent distinct settlements, not repeated grid cells.
TOP_CANDIDATES_FOR_DEDUP = 5000
MIN_SITE_SPACING_M = 1500
N_DISTINCT_SITES = 150


def to_xy(lon, lat):
    R = 6371000.0
    x = R * np.radians(lon)
    y = R * np.radians(lat)
    return x, y


def dedupe_sites(ranked):
    top = ranked.head(TOP_CANDIDATES_FOR_DEDUP).reset_index(drop=True)
    x, y = to_xy(top["lon"].values, top["lat"].values)
    coords = np.column_stack([x, y])

    selected_idx, selected_coords = [], []
    for i in range(len(top)):
        pt = coords[i]
        if selected_coords:
            d = np.min(np.linalg.norm(np.array(selected_coords) - pt, axis=1))
            if d < MIN_SITE_SPACING_M:
                continue
        selected_idx.append(i)
        selected_coords.append(pt)
        if len(selected_idx) >= N_DISTINCT_SITES:
            break

    deduped = top.iloc[selected_idx].reset_index(drop=True)
    deduped["rank"] = deduped.index + 1
    return deduped


def main():
    print("Loading model...")
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    print(f"Model: {bundle['name']}, features: {features}")

    print("Loading national feature grid...")
    usecols = ["lon", "lat", "building_count", "dist_to_nearest_building_m",
               "pole_count", "dist_to_nearest_pole_m"]
    df = pd.read_csv(FEATURES_PATH, usecols=usecols)
    X = df[features].fillna(0)

    print("Predicting electrification probability for all cells...")
    df["electrification_prob"] = model.predict_proba(X)[:, 1]

    df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"Saved full national predictions to {PREDICTIONS_OUT}")

    # --- Microgrid candidate filtering ---
    # dist_to_nearest_pole_m is our proxy for distance to grid infrastructure
    candidates = df[df["electrification_prob"] <= MAX_ELECTRIFICATION_PROB].copy()
    candidates = candidates[candidates["building_count"] >= MIN_BUILDING_COUNT]
    candidates = candidates[candidates["dist_to_nearest_pole_m"] >= MIN_DIST_TO_GRID_M]

    print(f"{len(candidates)} candidate cells after filtering (underserved + populated + far from grid)")

    # --- Suitability score: 40% population proxy, 30% building density, 30% distance from grid ---
    def norm(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    # No population raster available. building_count is binary (presence/absence) in this
    # dataset, so it can't rank density -- dist_to_nearest_building_m is our continuous
    # settlement-density proxy instead (closer neighboring building = denser cluster).
    candidates["score_pop"] = 1 - norm(candidates["dist_to_nearest_building_m"])
    candidates["score_building"] = 1 - norm(candidates["dist_to_nearest_building_m"])
    candidates["score_grid_dist"] = norm(candidates["dist_to_nearest_pole_m"])

    candidates["suitability_score"] = (
        0.40 * candidates["score_pop"]
        + 0.30 * candidates["score_building"]
        + 0.30 * candidates["score_grid_dist"]
    )

    ranked = candidates.sort_values("suitability_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    ranked.to_csv(RANKED_SITES_OUT, index=False)
    print(f"Saved {len(ranked)} ranked microgrid candidate sites to {RANKED_SITES_OUT}")

    print("Deduplicating into spatially distinct settlement sites...")
    deduped = dedupe_sites(ranked)
    deduped.to_csv(DEDUPED_SITES_OUT, index=False)
    print(f"Saved {len(deduped)} spatially distinct top sites to {DEDUPED_SITES_OUT}")
    print(deduped[["lon", "lat", "electrification_prob", "suitability_score", "rank"]].head(20))


if __name__ == "__main__":
    main()
