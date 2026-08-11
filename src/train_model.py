"""
Train a model predicting electrification status from satellite/OSM-derived
cell features, validated against real DHS household survey ground truth.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

TRAINING_PATH = "data/processed/training_set.csv"
MODEL_OUT = "output/electrification_model.joblib"
METRICS_OUT = "output/model_metrics.json"

# Candidate features -- filtered down to whatever actually exists in the data
CANDIDATE_FEATURES = [
    "building_count",
    "dist_to_nearest_building_m",
    "pole_count",
    "dist_to_nearest_pole_m",
]


def main():
    df = pd.read_csv(TRAINING_PATH)
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    print(f"Using features: {features}")

    X = df[features].fillna(0)
    y = df["electrified_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_model, best_auc = None, None, -1

    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        results[name] = {"cv_auc": cv_auc, "test_auc": test_auc, "test_accuracy": acc}
        print(f"\n{name}: CV AUC={cv_auc:.3f}  Test AUC={test_auc:.3f}  Test Acc={acc:.3f}")
        print(classification_report(y_test, pred))

        if test_auc > best_auc:
            best_auc, best_name, best_model = test_auc, name, model

    print(f"\nBest model: {best_name} (Test AUC={best_auc:.3f})")

    # Refit best model on ALL labeled data before deploying to full grid
    best_model.fit(X, y)
    joblib.dump({"model": best_model, "features": features, "name": best_name}, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump({"results": results, "best_model": best_name, "n_train": len(df)}, f, indent=2)

    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
