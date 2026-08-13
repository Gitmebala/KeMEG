# Run this in Google Colab (colab.research.google.com -> New notebook)
# Pulls Google Open Buildings data for Kenya's S2 tiles, aggregates building
# count onto our exact 500m grid, saves result to Drive for download.

import csv, re, gzip, io
import numpy as np
import pandas as pd
import urllib.request

# 1. Find which S2 level-4 tiles cover Kenya
KENYA_BBOX = (33.5, -5.5, 42.0, 5.5)  # lon_min, lat_min, lon_max, lat_max

def bbox_overlaps(poly_wkt, bbox):
    coords = re.findall(r'(-?\d+\.?\d*) (-?\d+\.?\d*)', poly_wkt)
    lons = [float(c[0]) for c in coords]
    lats = [float(c[1]) for c in coords]
    return not (max(lons) < bbox[0] or min(lons) > bbox[2] or max(lats) < bbox[1] or min(lats) > bbox[3])

print("Fetching S2 tile index...")
url = "https://storage.googleapis.com/open-buildings-data/v3/score_thresholds_s2_level_4.csv"
with urllib.request.urlopen(url) as resp:
    text = resp.read().decode("utf-8")

reader = csv.DictReader(io.StringIO(text))
tiles = [row["s2_token"] for row in reader if bbox_overlaps(row["geometry"], KENYA_BBOX)]
print(f"Found {len(tiles)} S2 tiles covering Kenya: {tiles}")

# 2. Our exact 500m grid definition (matches the project's feature grid)
LON_MIN, LON_MAX = 34.0025, 41.4975
LAT_MIN, LAT_MAX = -4.6975, 4.9975
STEP = 0.005

grid_lons = np.round(np.arange(LON_MIN, LON_MAX + STEP/2, STEP), 4)
grid_lats = np.round(np.arange(LAT_MIN, LAT_MAX + STEP/2, STEP), 4)
n_cols, n_rows = len(grid_lons), len(grid_lats)
print(f"Grid: {n_rows} x {n_cols} = {n_rows*n_cols} cells")

building_count = np.zeros((n_rows, n_cols), dtype=np.int32)

# 3. Download each tile's building points and bin into our grid
for token in tiles:
    print(f"Downloading tile {token}...")
    tile_url = f"https://storage.googleapis.com/open-buildings-data/v3/points_s2_level_4_gzip/{token}_buildings.csv.gz"
    try:
        with urllib.request.urlopen(tile_url) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"  skip {token}: {e}")
        continue

    with gzip.open(io.BytesIO(raw), "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (KENYA_BBOX[0] <= lon <= KENYA_BBOX[2] and KENYA_BBOX[1] <= lat <= KENYA_BBOX[3]):
                continue
            col = int(round((lon - LON_MIN) / STEP))
            row_i = int(round((lat - LAT_MIN) / STEP))
            if 0 <= col < n_cols and 0 <= row_i < n_rows:
                building_count[row_i, col] += 1
    print(f"  done, running total buildings binned: {building_count.sum()}")

# 4. Save result
lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
out = pd.DataFrame({
    "lon": lon_grid.ravel(),
    "lat": lat_grid.ravel(),
    "google_building_count": building_count.ravel(),
})
out.to_csv("kenya_google_building_density.csv", index=False)
print("Saved kenya_google_building_density.csv")
print(out.describe())

# 5. Save to Drive so it can be pulled down
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy("kenya_google_building_density.csv", "/content/drive/MyDrive/kenya_google_building_density.csv")
print("Copied to Google Drive root.")
