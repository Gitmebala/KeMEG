"""
Build a standalone interactive HTML map of Kenya:
- A true 500m-resolution raster of electrification probability (red = underserved,
  green = electrified), rendered as an image overlay -- not a blurred point heatmap.
- Distinct markers for the top ranked, spatially-deduplicated microgrid candidate
  sites, each poppuped with its full engineering spec.
"""
import numpy as np
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay
from matplotlib import cm
from PIL import Image

PREDICTIONS_PATH = "output/national_predictions.csv"
SPECS_PATH = "output/microgrid_engineering_specs.csv"
RASTER_PNG_OUT = "output/electrification_raster.png"
MAP_OUT = "output/kenya_electrification_map.html"

KENYA_CENTER = [0.15, 37.9]


def build_raster():
    print("Loading national predictions...")
    df = pd.read_csv(PREDICTIONS_PATH, usecols=["lon", "lat", "electrification_prob"])

    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    step = 0.005  # 500m grid spacing, confirmed regular lattice

    n_cols = int(round((lon_max - lon_min) / step)) + 1
    n_rows = int(round((lat_max - lat_min) / step)) + 1
    print(f"Rasterizing to {n_rows} x {n_cols} grid...")

    col = np.round((df["lon"].values - lon_min) / step).astype(int)
    row = np.round((lat_max - df["lat"].values) / step).astype(int)  # row 0 = top = max lat

    # Most of Kenya's land area is arid/sparse and legitimately scores low
    # (median predicted probability ~0.27), so a raw linear 0-1 red-yellow-green
    # scale makes "yellow" (0.5) almost unreachable -- nearly the whole country
    # renders red/orange and only the extreme top percentile (Nairobi, western
    # highlands) shows green. We instead color by percentile rank within
    # Kenya's own prediction distribution, so the map shows relative standing
    # (top half greener, bottom half redder) rather than being crushed by the
    # low absolute baseline everywhere outside the highest-density areas.
    order = df["electrification_prob"].values.argsort()
    percentile = np.empty(len(df), dtype=np.float32)
    percentile[order] = np.linspace(0, 1, len(df))

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[row, col] = percentile

    cmap = cm.get_cmap("RdYlGn")
    normed = np.clip(grid, 0, 1)
    rgba = cmap(normed, bytes=True)  # (n_rows, n_cols, 4) uint8

    # Transparent where there's no data (ocean, outside Kenya boundary)
    nodata_mask = np.isnan(grid)
    rgba[nodata_mask, 3] = 0
    rgba[~nodata_mask, 3] = 200  # slightly transparent so basemap labels stay visible

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(RASTER_PNG_OUT)
    print(f"Saved raster to {RASTER_PNG_OUT}")

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    return bounds


def need_color(region_mean_prob):
    # Colored by how underserved the surrounding region is (lower = more urgent),
    # not by suitability score -- suitability is normalized *within* each region
    # so it isn't comparable across regions, but electrification need is.
    if region_mean_prob < 0.15:
        return "red"
    elif region_mean_prob < 0.25:
        return "orange"
    else:
        return "beige"


def main():
    bounds = build_raster()

    m = folium.Map(location=KENYA_CENTER, zoom_start=6.3, tiles="cartodbpositron", control_scale=True)

    ImageOverlay(
        image=RASTER_PNG_OUT,
        bounds=bounds,
        opacity=0.75,
        name="Electrification probability (500m resolution)",
    ).add_to(m)

    print("Adding ranked microgrid site markers...")
    specs = pd.read_csv(SPECS_PATH)
    site_layer = folium.FeatureGroup(name="Top ranked microgrid candidate sites").add_to(m)

    for _, row_ in specs.iterrows():
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px;">
        <b>Microgrid Site — Rank #{int(row_['rank'])}</b><br>
        Cell electrification probability: {row_['electrification_prob']:.2f}<br>
        Surrounding region avg. electrification probability: {row_['region_mean_electrification_prob']:.2f}<br>
        Best-in-region suitability score: {row_['local_suitability_score']:.2f}<br>
        <hr style="margin:4px 0;">
        Est. population served: {int(row_['served_population'])}<br>
        Est. households served: {int(row_['est_households'])}<br>
        Daily demand: {row_['daily_demand_kwh']:.1f} kWh<br>
        Solar PV size: {row_['solar_pv_kw']:.1f} kW<br>
        Battery storage: {row_['battery_kwh']:.1f} kWh<br>
        Topology: {row_['topology']}<br>
        Est. capital cost: ${row_['capex_usd']:,.0f}
        </div>
        """
        folium.CircleMarker(
            location=[row_["lat"], row_["lon"]],
            radius=6,
            color="black",
            weight=1,
            fill=True,
            fill_color=need_color(row_["region_mean_electrification_prob"]),
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(site_layer)

    # --- Legend ---
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 12px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-family: sans-serif; font-size: 13px;">
      <b>Kenya Electrification Map</b><br>
      <span style="color:#666;">Color = relative standing vs. rest of Kenya (percentile), not raw probability</span><br><br>
      <div><span style="display:inline-block;width:14px;height:14px;background:#1a9850;margin-right:6px;"></span>Top-half electrification (above national median)</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#fee08b;margin-right:6px;"></span>Around the median</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#d73027;margin-right:6px;"></span>Bottom-half electrification (underserved)</div>
      <br>
      <b>Ranked microgrid sites</b> (one best per ~55km region)<br>
      <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:red;border:1px solid black;margin-right:6px;"></span>Most urgent region (avg. electrification &lt; 15%)</div>
      <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:orange;border:1px solid black;margin-right:6px;"></span>High need (15–25%)</div>
      <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:beige;border:1px solid black;margin-right:6px;"></span>Underserved (&gt; 25%)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(MAP_OUT)
    print(f"Saved interactive map to {MAP_OUT}")


if __name__ == "__main__":
    main()
