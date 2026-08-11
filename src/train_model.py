"""
Train a model predicting electrification status from satellite/OSM-derived
cell features, validated against real DHS household survey ground truth.

Uses spatial-block cross-validation, not random K-fold. 62% of DHS clusters
are within 5km of another labeled cluster (DHS surveys multiple clusters per
urban area), so random folds let near-duplicate, spatially-correlated points
leak across train/test -- inflating AUC with a score the model doesn't
actually earn on genuinely unseen locations. Grouping by ~20km spatial block
before splitting gives an honest estimate of how well this generalizes to
places the model hasn't seen anything nearby.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

TRAINING_PATH = "data/processed/training_set.csv"
MODEL_OUT = "output/electrification_model.joblib"
METRICS_OUT = "output/model_metrics.json"

SPATIAL_BLOCK_DEG = 0.18  # ~20km at the equator -- clusters in the same block never split across train/test

# Candidate features -- filtered down to whatever actually exists in the data
CANDIDATE_FEATURES = [
    "building_count",
    "dist_to_nearest_building_m",
    "pole_count",
    "dist_to_nearest_pole_m",
    "population",
    "buildings_within_1000m",
    "buildings_within_3000m",
    "poles_within_5000m",
    "poles_within_10000m",
    "dist_to_nearest_road_m",
    "roads_within_2000m",
]


def spatial_blocks(df):
    return (
        (df["lon"] // SPATIAL_BLOCK_DEG).astype(int).astype(str)
        + "_"
        + (df["lat"] // SPATIAL_BLOCK_DEG).astype(int).astype(str)
    )


def main():
    df = pd.read_csv(TRAINING_PATH)
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing:
        print(f"NOTE: skipping unavailable features (run enrich_features/roads first): {missing}")
    print(f"Using features: {features}")

    X = df[features].fillna(0)
    y = df["electrified_label"]
    groups = spatial_blocks(df)
    n_blocks = groups.nunique()
    print(f"{len(df)} labeled cells across {n_blocks} spatial blocks (~20km each)")

    # Spatial-block holdout test set: hold out ~20% of BLOCKS entirely (not just rows),
    # so every test-set cell is genuinely far from every training-set cell.
    rng = np.random.RandomState(42)
    unique_blocks = groups.unique()
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(0.2 * len(unique_blocks)))
    test_blocks = set(unique_blocks[:n_test_blocks])
    test_mask = groups.isin(test_blocks)

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    groups_train = groups[~test_mask]
    print(f"Spatial holdout: {len(X_train)} train cells, {len(X_test)} test cells "
          f"({len(test_blocks)} held-out blocks)")

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42
        ),
    }

    results = {}
    n_splits = min(5, groups_train.nunique())
    cv = GroupKFold(n_splits=n_splits)
    best_name, best_model, best_auc = None, None, -1

    for name, model in models.items():
        cv_auc = cross_val_score(
            model, X_train, y_train, cv=cv, groups=groups_train, scoring="roc_auc"
        ).mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        results[name] = {"spatial_cv_auc": cv_auc, "spatial_test_auc": test_auc, "spatial_test_accuracy": acc}
        print(f"\n{name}: Spatial CV AUC={cv_auc:.3f}  Spatial Test AUC={test_auc:.3f}  Test Acc={acc:.3f}")
        print(classification_report(y_test, pred))

        if test_auc > best_auc:
            best_auc, best_name, best_model = test_auc, name, model

    print(f"\nBest model: {best_name} (Spatial Test AUC={best_auc:.3f})")

    # Refit best model on ALL labeled data before deploying to full grid
    best_model.fit(X, y)
    joblib.dump({"model": best_model, "features": features, "name": best_name}, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump({
            "results": results,
            "best_model": best_name,
            "n_train": len(df),
            "n_spatial_blocks": int(n_blocks),
            "note": "AUC/accuracy computed with spatial-block holdout (entire ~20km blocks held out), "
                    "not random K-fold, to avoid leakage from spatially-clustered DHS survey design.",
        }, f, indent=2)

    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
