"""
Add VIIRS nighttime-lights radiance as a feature to the enriched grid.

Nighttime radiance is one of the strongest available proxies for
electrification -- literally observing artificial light from space -- and is
the feature the original project plan flagged as blocked pending API/account
access. Source: cropped Kenya slice of the VIIRS annual composite (2025,
average_masked -- background already zeroed), see crop_viirs.py.
"""
import numpy as np
import pandas as pd

VIIRS_PATH = "data/raw/viirs/kenya_viirs_ntl.npz"
FEATURES_PATH = "data/processed/enriched_features.csv"
OUT_PATH = "data/processed/enriched_features.csv"

VIIRS_PIXELS_PER_DEG = 240  # 15 arc-second


def main():
    print("Loading cropped VIIRS raster...")
    d = np.load(VIIRS_PATH)
    radiance, lons, lats = d["radiance"], d["lons"], d["lats"]
    lon_min, lat_max = lons[0], lats[0]
    n_rows, n_cols = radiance.shape
    print(f"  VIIRS grid: {radiance.shape}, lon [{lons[0]:.2f},{lons[-1]:.2f}] lat [{lats[-1]:.2f},{lats[0]:.2f}]")

    print("Loading feature grid...")
    df = pd.read_csv(FEATURES_PATH)

    col = np.round((df["lon"].values - lon_min) * VIIRS_PIXELS_PER_DEG).astype(int)
    row = np.round((lat_max - df["lat"].values) * VIIRS_PIXELS_PER_DEG).astype(int)
    col = np.clip(col, 0, n_cols - 1)
    row = np.clip(row, 0, n_rows - 1)

    df["ntl_radiance"] = radiance[row, col]

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved with ntl_radiance feature to {OUT_PATH}")
    print(df["ntl_radiance"].describe())
    print("nonzero fraction:", (df["ntl_radiance"] > 0).mean())


if __name__ == "__main__":
    main()
