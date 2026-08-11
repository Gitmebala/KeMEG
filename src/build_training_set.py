"""
Join DHS cluster-level electrification data (ground truth) to the 500m
Kenya feature grid, producing a labeled training set for the ML model.

DHS clusters are GPS points (jittered up to 2km urban / 5-10km rural per
DHS privacy protocol), so we snap each cluster to its nearest grid cell
using a KDTree nearest-neighbor search in projected (metric) space.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURES_PATH = "data/processed/enriched_features.csv"
DHS_PATH = "data/processed/dhs_cluster_electrification.csv"
OUT_PATH = "data/processed/training_set.csv"

# Rough equirectangular projection centered on Kenya is fine for nearest-neighbor
# matching at this scale (we only need relative distances, not survey-grade accuracy).
KENYA_LAT0 = 0.0  # equator-ish center


def to_xy(lon, lat):
    R = 6371000.0
    lat0 = np.radians(KENYA_LAT0)
    x = R * np.radians(lon) * np.cos(lat0)
    y = R * np.radians(lat)
    return x, y


def main():
    print("Loading DHS cluster electrification data...")
    dhs = pd.read_csv(DHS_PATH)
    print(f"  {len(dhs)} DHS clusters loaded")

    print("Loading enriched 500m feature grid...")
    feat = pd.read_csv(FEATURES_PATH)
    print(f"  {len(feat)} grid cells loaded, columns: {list(feat.columns)}")

    if "lon" not in feat.columns or "lat" not in feat.columns:
        raise ValueError(f"Expected lon/lat columns in features file, got: {feat.columns.tolist()}")

    print("Building KDTree over grid cells...")
    gx, gy = to_xy(feat["lon"].values, feat["lat"].values)
    tree = cKDTree(np.column_stack([gx, gy]))

    print("Matching DHS clusters to nearest grid cells...")
    dx, dy = to_xy(dhs["lon"].values, dhs["lat"].values)
    dist, idx = tree.query(np.column_stack([dx, dy]), k=1)

    matched = feat.iloc[idx].reset_index(drop=True)
    dhs_reset = dhs.reset_index(drop=True)

    training = pd.concat([matched, dhs_reset[["hv001", "pct_electrified", "n_households", "URBAN_RURA"]]], axis=1)
    training["match_dist_m"] = dist
    training["electrified_label"] = (training["pct_electrified"] >= 0.5).astype(int)

    # Drop matches that snapped absurdly far away (bad data / edge of grid)
    before = len(training)
    training = training[training["match_dist_m"] <= 2000]
    print(f"  Dropped {before - len(training)} clusters with match distance > 2000m")

    training.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(training)} labeled training rows to {OUT_PATH}")
    print(training["electrified_label"].value_counts())


if __name__ == "__main__":
    main()
