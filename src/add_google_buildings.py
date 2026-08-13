"""
Merge Google Open Buildings density (real building counts per 500m cell,
derived from satellite ML detection) into the enriched feature grid.

This replaces reliance on OSM's binary building presence flag, which only
covers 6.7% of Kenya's cells due to incomplete community mapping. Google
Open Buildings covers 21.6% of cells with actual counts (not just presence),
detected via satellite imagery -- far more complete, especially in remote
areas OSM contributors haven't mapped.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GOOGLE_BUILDINGS_PATH = "data/raw/google_buildings/kenya_google_building_density.csv"
FEATURES_PATH = "data/processed/enriched_features.csv"
OUT_PATH = "data/processed/enriched_features.csv"

RADIUS_M = 1000


def to_xy(lon, lat):
    R = 6371000.0
    x = R * np.radians(lon)
    y = R * np.radians(lat)
    return x, y


def main():
    print("Loading Google Open Buildings density...")
    gb = pd.read_csv(GOOGLE_BUILDINGS_PATH)
    print(f"  {len(gb)} cells, {gb['google_building_count'].sum():,.0f} total buildings")

    print("Loading feature grid...")
    df = pd.read_csv(FEATURES_PATH)

    # Both grids use the same 500m definition but were computed independently (one
    # locally, one in Colab) via np.arange/rounding -- tiny floating-point
    # representation differences mean an exact-equality merge on lon/lat only
    # catches coincidental bit-for-bit matches (verified: 21 of 2.91M rows).
    # Nearest-neighbor spatial join is robust to this.
    print("Matching via nearest-neighbor (not exact float merge)...")
    gx, gy = to_xy(gb["lon"].values, gb["lat"].values)
    tree = cKDTree(np.column_stack([gx, gy]))
    dx, dy = to_xy(df["lon"].values, df["lat"].values)
    dist, idx = tree.query(np.column_stack([dx, dy]), k=1)
    within_tolerance = dist <= 50  # cells should coincide almost exactly; 50m is generous

    merged = df.copy()
    merged["google_building_count"] = 0.0
    merged.loc[within_tolerance, "google_building_count"] = gb["google_building_count"].values[idx[within_tolerance]]
    print(f"  Matched {within_tolerance.sum()} of {len(df)} cells within 50m "
          f"({within_tolerance.mean():.4f}), max match distance considered: 50m")
    print(f"  Merged: {(merged['google_building_count']>0).mean():.4f} nonzero fraction")

    # Neighborhood density (mirrors buildings_within_1000m/3000m, but on the richer source)
    print("Computing neighborhood building density from Google data...")
    x, y = to_xy(merged["lon"].values, merged["lat"].values)
    coords = np.column_stack([x, y])
    weights = merged["google_building_count"].values

    # Weighted local sum via KDTree: for each cell, sum google_building_count of all
    # cells within RADIUS_M (including itself)
    tree = cKDTree(coords)
    pairs = tree.query_ball_point(coords, r=RADIUS_M)
    google_buildings_within_1000m = np.array([weights[idxs].sum() for idxs in pairs])
    merged["google_buildings_within_1000m"] = google_buildings_within_1000m

    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    print(merged[["google_building_count", "google_buildings_within_1000m"]].describe())


if __name__ == "__main__":
    main()
