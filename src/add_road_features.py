"""
Add road-network features to the enriched grid: distance to nearest major
road and road density within a service radius, for every 500m cell.

Roads are one of the strongest electrification predictors in the remote-
sensing literature (grid lines are built alongside road corridors far more
often than across open terrain) and weren't used at all before this. Data
pulled from OSM via Overpass API (trunk/primary/secondary/tertiary ways for
all of Kenya) -- see data/raw/osm_roads/kenya_major_roads.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROADS_JSON_PATH = "data/raw/osm_roads/kenya_major_roads.json"
FEATURES_IN_PATH = "data/processed/enriched_features.csv"
FEATURES_OUT_PATH = "data/processed/enriched_features.csv"

ROAD_DENSITY_RADIUS_M = 2000


def to_xy(lon, lat):
    R = 6371000.0
    x = R * np.radians(lon)
    y = R * np.radians(lat)
    return x, y


def load_road_vertices():
    print("Loading road network from Overpass extract...")
    with open(ROADS_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    lons, lats = [], []
    for way in data["elements"]:
        for pt in way.get("geometry", []):
            lons.append(pt["lon"])
            lats.append(pt["lat"])

    print(f"  {len(data['elements'])} road ways, {len(lons)} vertices")
    return np.array(lons), np.array(lats)


def main():
    road_lons, road_lats = load_road_vertices()
    rx, ry = to_xy(road_lons, road_lats)
    road_tree = cKDTree(np.column_stack([rx, ry]))

    print("Loading grid...")
    df = pd.read_csv(FEATURES_IN_PATH)
    gx, gy = to_xy(df["lon"].values, df["lat"].values)
    grid_coords = np.column_stack([gx, gy])

    print("Computing distance to nearest road for all cells...")
    dist, _ = road_tree.query(grid_coords, k=1)
    df["dist_to_nearest_road_m"] = dist

    print(f"Computing road density within {ROAD_DENSITY_RADIUS_M}m...")
    df["roads_within_2000m"] = road_tree.query_ball_point(grid_coords, r=ROAD_DENSITY_RADIUS_M, return_length=True)

    df.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"Saved with road features to {FEATURES_OUT_PATH}")
    print(df[["dist_to_nearest_road_m", "roads_within_2000m"]].describe())


if __name__ == "__main__":
    main()
