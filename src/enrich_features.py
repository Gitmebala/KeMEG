"""
Enrich the raw 500m grid with:
1. Real population count, sampled from the WorldPop Kenya raster (replaces the
   binary OSM building_count as the "populated" signal -- OSM building mapping
   completeness varies wildly across Kenya, badly under-representing remote
   pastoralist counties like Turkana/Marsabit/Wajir that are OSM-sparse but
   very much inhabited and underserved).
2. Continuous neighborhood-density features from KDTree radius queries:
   building presence and pole presence are binary per-cell flags in this OSM
   extract, so on their own they carry almost no gradation. Counting how many
   building/pole cells fall within a real service radius turns them into
   genuinely informative continuous features -- and, as a side effect, kills
   the massive score plateaus that were causing ranked sites to clump
   wherever cells happened to tie and appear first in file order.
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None

FEATURES_PATH = "data/raw/full_kenya_500m_features.csv"
POP_RASTER_PATH = "data/raw/population/ken_ppp_2020_1km.tif"
OUT_PATH = "data/processed/enriched_features.csv"

BUILDING_RADII_M = [1000, 3000]
POLE_RADII_M = [5000, 10000]


def to_xy(lon, lat):
    R = 6371000.0
    x = R * np.radians(lon)
    y = R * np.radians(lat)
    return x, y


def sample_population(lons, lats):
    # rasterio's compiled GDAL binaries are blocked by this machine's Windows
    # Application Control policy, so we read the GeoTIFF's raw pixel array and
    # georeferencing tags directly via Pillow instead (WorldPop is plain
    # EPSG:4326 lat/lon, so no reprojection math is needed).
    print("Sampling WorldPop population raster at each grid cell (via Pillow)...")
    img = Image.open(POP_RASTER_PATH)
    tags = img.tag_v2
    scale_x, scale_y, _ = tags[33550]
    origin_x, origin_y = tags[33922][3], tags[33922][4]
    nodata = float(tags.get(42113, -99999))

    arr = np.array(img, dtype=np.float32)  # (rows, cols)
    n_rows, n_cols = arr.shape

    col = np.round((lons - origin_x) / scale_x).astype(int)
    row = np.round((origin_y - lats) / scale_y).astype(int)
    col = np.clip(col, 0, n_cols - 1)
    row = np.clip(row, 0, n_rows - 1)

    vals = arr[row, col]
    vals[(vals == nodata) | (vals < 0) | np.isnan(vals)] = 0.0

    # Our 500m grid is ~4x finer than WorldPop's 1km pixels, so multiple grid
    # cells share the same underlying pixel and would each report its full
    # population -- inflating any later sum by ~4x. Apportion each pixel's
    # population evenly across however many grid cells actually fall inside it,
    # so per-cell values sum back to the real total population.
    pixel_id = row.astype(np.int64) * n_cols + col.astype(np.int64)
    _, inverse, counts = np.unique(pixel_id, return_inverse=True, return_counts=True)
    shares = counts[inverse]
    vals = vals / shares

    return vals


def add_neighborhood_density(df):
    print("Building KDTrees for neighborhood density features...")
    x, y = to_xy(df["lon"].values, df["lat"].values)
    all_coords = np.column_stack([x, y])

    building_mask = df["building_count"].values >= 1
    pole_mask = df["pole_count"].values >= 1
    print(f"  {building_mask.sum()} building-present cells, {pole_mask.sum()} pole-present cells")

    building_tree = cKDTree(all_coords[building_mask])
    pole_tree = cKDTree(all_coords[pole_mask])

    for r in BUILDING_RADII_M:
        print(f"  Counting buildings within {r}m...")
        df[f"buildings_within_{r}m"] = building_tree.query_ball_point(all_coords, r=r, return_length=True)

    for r in POLE_RADII_M:
        print(f"  Counting poles within {r}m...")
        df[f"poles_within_{r}m"] = pole_tree.query_ball_point(all_coords, r=r, return_length=True)

    return df


def main():
    print("Loading feature grid...")
    usecols = ["lon", "lat", "building_count", "dist_to_nearest_building_m",
               "pole_count", "dist_to_nearest_pole_m"]
    df = pd.read_csv(FEATURES_PATH, usecols=usecols)
    print(f"  {len(df)} cells loaded")

    df["population"] = sample_population(df["lon"].values, df["lat"].values)
    print(f"  Population sampled: total={df['population'].sum():,.0f}, "
          f"nonzero cells={int((df['population']>0).sum()):,}")

    df = add_neighborhood_density(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved enriched features to {OUT_PATH}")
    print(df.describe())


if __name__ == "__main__":
    main()
