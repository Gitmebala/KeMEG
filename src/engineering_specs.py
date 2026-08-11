"""
Generate preliminary minigrid engineering specifications for the top-ranked
candidate sites: household count, daily demand, solar PV size, battery
capacity, topology recommendation, and capital cost estimate.

Benchmarks are drawn from published GIZ / AfDB Kenya mini-grid figures:
- Avg household load: 0.15 kWh/day (basic rural tier, lighting + phone + small appliances)
- Solar irradiance (Kenya average): 5.5 peak sun hours/day
- System losses (inverter, wiring, battery round-trip): 25%
- Battery autonomy: 1.5 days, 80% usable depth of discharge
- Capital cost benchmark: $4,000-6,000 per connection (GIZ/AfDB mini-grid survey range)
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RANKED_SITES_PATH = "output/ranked_microgrid_sites_deduped.csv"
FEATURES_PATH = "data/raw/full_kenya_500m_features.csv"
SPECS_OUT = "output/microgrid_engineering_specs.csv"

TOP_N = 100
SERVICE_RADIUS_M = 1000  # buildings within this radius of a site are assumed served by it

AVG_HOUSEHOLD_LOAD_KWH_DAY = 0.15
PEOPLE_PER_HOUSEHOLD = 4.5
SOLAR_PEAK_SUN_HOURS = 5.5
SYSTEM_LOSS_FACTOR = 0.75      # i.e. 25% losses
BATTERY_AUTONOMY_DAYS = 1.5
BATTERY_DOD = 0.8
COST_PER_CONNECTION_USD = 5000  # midpoint of GIZ/AfDB range
AC_MINIGRID_HOUSEHOLD_THRESHOLD = 50  # above this, recommend 3-phase AC


def to_xy(lon, lat):
    R = 6371000.0
    x = R * np.radians(lon)  # Kenya spans the equator, so cos(lat0)~1 is fine here
    y = R * np.radians(lat)
    return x, y


def main():
    df = pd.read_csv(RANKED_SITES_PATH).head(TOP_N).copy()

    # building_count in the source data is a binary presence flag per 500m cell, not a
    # count -- so we estimate settlement size by counting how many building-present cells
    # fall within a realistic microgrid service radius of each candidate site.
    print("Loading building-present cells for neighborhood density lookup...")
    feat = pd.read_csv(FEATURES_PATH, usecols=["lon", "lat", "building_count"])
    buildings = feat[feat["building_count"] >= 1]
    bx, by = to_xy(buildings["lon"].values, buildings["lat"].values)
    tree = cKDTree(np.column_stack([bx, by]))

    sx, sy = to_xy(df["lon"].values, df["lat"].values)
    neighbor_counts = tree.query_ball_point(np.column_stack([sx, sy]), r=SERVICE_RADIUS_M, return_length=True)
    df["nearby_building_cells"] = neighbor_counts

    # ~80% of OSM-mapped structures are assumed residential (documented assumption --
    # no ground-truth building-use tags available in this OSM extract).
    df["est_households"] = (df["nearby_building_cells"] * 0.8).round().clip(lower=1).astype(int)
    df["daily_demand_kwh"] = df["est_households"] * AVG_HOUSEHOLD_LOAD_KWH_DAY
    df["solar_pv_kw"] = df["daily_demand_kwh"] / (SOLAR_PEAK_SUN_HOURS * SYSTEM_LOSS_FACTOR)
    df["battery_kwh"] = (df["daily_demand_kwh"] * BATTERY_AUTONOMY_DAYS) / BATTERY_DOD
    df["topology"] = df["est_households"].apply(
        lambda h: "AC three-phase mini-grid" if h >= AC_MINIGRID_HOUSEHOLD_THRESHOLD else "DC cluster"
    )
    df["capex_usd"] = df["est_households"] * COST_PER_CONNECTION_USD

    cols = [c for c in [
        "rank", "lon", "lat", "electrification_prob", "suitability_score",
        "est_households", "daily_demand_kwh", "solar_pv_kw", "battery_kwh",
        "topology", "capex_usd"
    ] if c in df.columns]

    df[cols].to_csv(SPECS_OUT, index=False)
    print(f"Saved engineering specs for top {len(df)} sites to {SPECS_OUT}")
    print(df[cols].head(10))


if __name__ == "__main__":
    main()
