"""
Apply the trained model to all ~2.9M grid cells nationwide, then filter and
rank underserved cells as microgrid candidate sites.

Site selection uses real WorldPop population (not the binary OSM building
flag) so remote, OSM-sparse but genuinely inhabited regions -- exactly the
least-electrified counties per CRA data (Turkana, Marsabit, Wajir, Mandera)
-- aren't filtered out just because nobody has mapped their buildings on OSM.

Ranked sites are picked one-per-macro-cell (10km bins) rather than by a
greedy score-then-suppress pass, so the result reflects genuine nationwide
geographic spread instead of clustering wherever candidates happen to tie or
appear first in the source file's row order.
"""
import joblib
import numpy as np
import pandas as pd

FEATURES_PATH = "data/processed/enriched_features.csv"
MODEL_PATH = "output/electrification_model.joblib"
PREDICTIONS_OUT = "output/national_predictions.csv"
RANKED_SITES_OUT = "output/ranked_microgrid_sites.csv"
DEDUPED_SITES_OUT = "output/ranked_microgrid_sites_deduped.csv"

# Microgrid candidate filters. Narrowed from the earlier pass (0.35 prob / pop
# >= 5 / 150 sites) to a smaller, higher-confidence shortlist: national median
# predicted probability is ~0.27, so 0.35 was barely below-average, not
# genuinely underserved -- 0.20 means the model is confidently calling the
# cell underserved. MIN_POPULATION raised from 5 to 20 so sites represent a
# real hamlet, not a handful of scattered structures.
MAX_ELECTRIFICATION_PROB = 0.20   # confidently underserved, not just below-average
MIN_POPULATION = 20               # a genuine small settlement (real WorldPop count)
MIN_DIST_TO_GRID_M = 500          # grid extension not economical beyond this

# Geographic diversification: a purely national suitability ranking, dominated
# by population, will always crown the same high-density western settlements
# and completely starve low-density-but-highest-need arid counties (Turkana
# 2.4% electrified, West Pokot 2.0%, Mandera 2.5% per CRA data) of any
# representation. Instead we bin candidates into REGION_CELL_DEG regions and
# normalize suitability *within* each region -- so a region's own best
# opportunity is judged against its own distribution, not against Nairobi's or
# Kiambu's population scale. We then take the single best site per qualifying
# region, prioritizing the neediest (lowest mean electrification probability)
# regions first when capping the total. Region size doubled from ~55km to
# ~110km and the site cap cut from 150 to 40 -- each surviving site now
# represents a meaningfully larger, more distinct catchment area instead of
# a dense scatter of markers every 55km.
REGION_CELL_DEG = 1.0   # ~110km at the equator
MIN_CANDIDATES_PER_REGION = 25  # ignore tiny/noisy regions with too few candidate cells
N_DISTINCT_SITES = 40


def pick_best_per_region(candidates):
    c = candidates.copy()
    c["region_lon"] = (c["lon"] // REGION_CELL_DEG).astype(int)
    c["region_lat"] = (c["lat"] // REGION_CELL_DEG).astype(int)

    def norm(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0 + 0.5

    winners = []
    region_priority = []
    for (rlon, rlat), grp in c.groupby(["region_lon", "region_lat"]):
        if len(grp) < MIN_CANDIDATES_PER_REGION:
            continue
        local_score = (
            0.40 * norm(grp["population"])
            + 0.30 * norm(grp["buildings_within_1000m"])
            + 0.30 * norm(grp["dist_to_nearest_pole_m"])
        )
        best = grp.loc[local_score.idxmax()].copy()
        best["local_suitability_score"] = local_score.max()
        best["region_n_candidates"] = len(grp)
        best["region_mean_electrification_prob"] = grp["electrification_prob"].mean()
        winners.append(best)
        region_priority.append(best["region_mean_electrification_prob"])

    winners_df = pd.DataFrame(winners).drop(columns=["region_lon", "region_lat"])
    # Neediest regions (lowest mean electrification probability) get priority
    # when we have more qualifying regions than slots to fill.
    winners_df = winners_df.sort_values("region_mean_electrification_prob", ascending=True).reset_index(drop=True)
    winners_df = winners_df.head(N_DISTINCT_SITES)
    winners_df["rank"] = winners_df.index + 1
    return winners_df


def main():
    print("Loading model...")
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    print(f"Model: {bundle['name']}, features: {features}")

    print("Loading enriched national feature grid...")
    df = pd.read_csv(FEATURES_PATH)
    X = df[features].fillna(0)

    print("Predicting electrification probability for all cells...")
    df["electrification_prob"] = model.predict_proba(X)[:, 1]

    df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"Saved full national predictions to {PREDICTIONS_OUT}")

    # --- Microgrid candidate filtering ---
    candidates = df[df["electrification_prob"] <= MAX_ELECTRIFICATION_PROB].copy()
    candidates = candidates[candidates["population"] >= MIN_POPULATION]
    candidates = candidates[candidates["dist_to_nearest_pole_m"] >= MIN_DIST_TO_GRID_M]

    print(f"{len(candidates)} candidate cells after filtering (underserved + populated + far from grid)")
    print("Candidate distribution by region (rough quadrants):")
    print(f"  lon<36 (west): {(candidates['lon'] < 36).sum()}   lon>=36 (east): {(candidates['lon'] >= 36).sum()}")
    print(f"  lat<0 (south): {(candidates['lat'] < 0).sum()}   lat>=0 (north): {(candidates['lat'] >= 0).sum()}")

    # --- Suitability score: 40% population, 30% building density, 30% distance from grid ---
    def norm(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    candidates["score_pop"] = norm(candidates["population"])
    candidates["score_building"] = norm(candidates["buildings_within_1000m"])
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

    print("Selecting the best site per ~55km region (locally normalized, neediest regions first)...")
    deduped = pick_best_per_region(candidates)
    deduped.to_csv(DEDUPED_SITES_OUT, index=False)
    print(f"Saved {len(deduped)} spatially distinct top sites to {DEDUPED_SITES_OUT}")
    print(f"  lon range: {deduped['lon'].min():.2f} to {deduped['lon'].max():.2f}")
    print(f"  lat range: {deduped['lat'].min():.2f} to {deduped['lat'].max():.2f}")
    print(deduped[["lon", "lat", "population", "electrification_prob", "suitability_score", "rank"]].head(20))


if __name__ == "__main__":
    main()
