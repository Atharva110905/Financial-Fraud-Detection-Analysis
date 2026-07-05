"""
Fraud Detection Model Training & Comparison
----------------------------------------------
Trains and evaluates:
  Supervised:   Logistic Regression, Random Forest, XGBoost, LightGBM
  Unsupervised: Isolation Forest, One-Class SVM
  Deep Learning: Autoencoder (reconstruction-error based anomaly detection)

Saves trained models + evaluation metrics + ROC/PR curve data for the
Streamlit dashboard to load without retraining on every page refresh.
"""

import sqlite3
import json
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    average_precision_score
)

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DB_PATH = "/home/claude/fraud_project/fraud_detection.db"
ARTIFACTS_DIR = "/home/claude/fraud_project/artifacts"

import os
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Loading data from SQLite...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()
print(f"  Loaded {len(df):,} rows")

# ---------------------------------------------------------------
# Feature engineering for modeling
# ---------------------------------------------------------------
print("\nPreparing features...")

feature_cols_numeric = [
    "amount", "distance_from_home_km", "account_age_days", "hour_of_day",
    "is_weekend", "minutes_since_last_txn", "avg_monthly_spend",
    "amount_to_avg_spend_ratio", "is_new_device"
]
feature_cols_categorical = ["merchant_category", "channel", "device_type", "card_type", "customer_risk_segment"]

model_df = df[feature_cols_numeric + feature_cols_categorical + ["is_fraud", "transaction_id"]].copy()

# Encode categoricals
encoders = {}
for col in feature_cols_categorical:
    le = LabelEncoder()
    model_df[col + "_enc"] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

feature_cols_final = feature_cols_numeric + [c + "_enc" for c in feature_cols_categorical]

X = model_df[feature_cols_final].to_numpy()
y = model_df["is_fraud"].to_numpy()
txn_ids = model_df["transaction_id"].astype(str).to_numpy(dtype=object)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, txn_ids, test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Fraud rate train: {y_train.mean()*100:.2f}%, test: {y_test.mean()*100:.2f}%")

results = {}
roc_data = {}
pr_data = {}
trained_models = {}

def evaluate(name, y_true, y_pred, y_score, train_time):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4) if len(np.unique(y_true)) > 1 else None,
        "avg_precision": round(average_precision_score(y_true, y_score), 4),
        "train_time_sec": round(train_time, 2),
    }
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_data[name] = {"fpr": fpr[::max(1, len(fpr)//200)].tolist(),
                       "tpr": tpr[::max(1, len(tpr)//200)].tolist()}

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_data[name] = {"precision": prec[::max(1, len(prec)//200)].tolist(),
                      "recall": rec[::max(1, len(rec)//200)].tolist()}

    results[name] = metrics
    print(f"  [{name}] Acc={metrics['accuracy']} Prec={metrics['precision']} "
          f"Rec={metrics['recall']} F1={metrics['f1_score']} AUC={metrics['roc_auc']} "
          f"({train_time:.1f}s)")

# =================================================================
# 1. Logistic Regression
# =================================================================
print("\n[1/7] Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr.fit(X_train_scaled, y_train)
train_time = time.time() - t0
y_score = lr.predict_proba(X_test_scaled)[:, 1]
y_pred = lr.predict(X_test_scaled)
evaluate("Logistic Regression", y_test, y_pred, y_score, train_time)
trained_models["Logistic Regression"] = lr

# =================================================================
# 2. Random Forest
# =================================================================
print("\n[2/7] Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
train_time = time.time() - t0
y_score = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
evaluate("Random Forest", y_test, y_pred, y_score, train_time)
trained_models["Random Forest"] = rf
rf_feature_importance = dict(zip(feature_cols_final, rf.feature_importances_.round(4).tolist()))

# =================================================================
# 3. XGBoost
# =================================================================
print("\n[3/7] XGBoost...")
t0 = time.time()
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=SEED, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
train_time = time.time() - t0
y_score = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)
evaluate("XGBoost", y_test, y_pred, y_score, train_time)
trained_models["XGBoost"] = xgb_model

# =================================================================
# 4. LightGBM
# =================================================================
print("\n[4/7] LightGBM...")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=SEED,
    n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)
train_time = time.time() - t0
y_score = lgb_model.predict_proba(X_test)[:, 1]
y_pred = lgb_model.predict(X_test)
evaluate("LightGBM", y_test, y_pred, y_score, train_time)
trained_models["LightGBM"] = lgb_model

# =================================================================
# 5. Isolation Forest (unsupervised)
# =================================================================
print("\n[5/7] Isolation Forest...")
t0 = time.time()
contamination = max(y_train.mean(), 0.01)
iso = IsolationForest(
    n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1
)
iso.fit(X_train_scaled)
train_time = time.time() - t0
# IsolationForest: -1 = anomaly, 1 = normal. Convert to fraud=1 convention.
raw_pred = iso.predict(X_test_scaled)
y_pred = (raw_pred == -1).astype(int)
# score_samples: lower = more abnormal. Flip sign so higher = more fraud-like.
y_score = -iso.score_samples(X_test_scaled)
evaluate("Isolation Forest", y_test, y_pred, y_score, train_time)
trained_models["Isolation Forest"] = iso

# =================================================================
# 6. One-Class SVM (unsupervised) - trained on normal txns only, subsampled for speed
# =================================================================
print("\n[6/7] One-Class SVM...")
t0 = time.time()
normal_idx = np.where(y_train == 0)[0]
sample_size = min(8000, len(normal_idx))
sample_idx = np.random.choice(normal_idx, sample_size, replace=False)
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=max(y_train.mean(), 0.02))
ocsvm.fit(X_train_scaled[sample_idx])
train_time = time.time() - t0
raw_pred = ocsvm.predict(X_test_scaled)
y_pred = (raw_pred == -1).astype(int)
y_score = -ocsvm.decision_function(X_test_scaled)
evaluate("One-Class SVM", y_test, y_pred, y_score, train_time)
trained_models["One-Class SVM"] = ocsvm

# =================================================================
# 7. Autoencoder (deep learning anomaly detection)
# =================================================================
print("\n[7/7] Autoencoder...")
t0 = time.time()
input_dim = X_train_scaled.shape[1]
normal_train = X_train_scaled[y_train == 0]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(12, activation="relu"),
    layers.Dense(6, activation="relu"),
    layers.Dense(3, activation="relu"),
    layers.Dense(6, activation="relu"),
    layers.Dense(12, activation="relu"),
    layers.Dense(input_dim, activation="linear"),
])
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    normal_train, normal_train,
    epochs=25, batch_size=256, shuffle=True, verbose=0,
    validation_split=0.1
)
train_time = time.time() - t0

reconstructions = autoencoder.predict(X_test_scaled, verbose=0)
recon_error = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
threshold = np.percentile(recon_error[y_test == 0], 95)
y_pred = (recon_error > threshold).astype(int)
y_score = recon_error
evaluate("Autoencoder", y_test, y_pred, y_score, train_time)
trained_models["Autoencoder"] = autoencoder

# ---------------------------------------------------------------
# Save everything for the dashboard
# ---------------------------------------------------------------
print("\nSaving artifacts...")

with open(f"{ARTIFACTS_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{ARTIFACTS_DIR}/roc_data.json", "w") as f:
    json.dump(roc_data, f)

with open(f"{ARTIFACTS_DIR}/pr_data.json", "w") as f:
    json.dump(pr_data, f)

with open(f"{ARTIFACTS_DIR}/feature_importance.json", "w") as f:
    json.dump(rf_feature_importance, f, indent=2)

with open(f"{ARTIFACTS_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{ARTIFACTS_DIR}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open(f"{ARTIFACTS_DIR}/feature_cols.json", "w") as f:
    json.dump({"numeric": feature_cols_numeric, "categorical": feature_cols_categorical,
               "final": feature_cols_final}, f)

# Save sklearn-compatible models with pickle, autoencoder via keras format
for name, model in trained_models.items():
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    if name == "Autoencoder":
        model.save(f"{ARTIFACTS_DIR}/{safe_name}.keras")
    else:
        with open(f"{ARTIFACTS_DIR}/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

# Save test set predictions for a "model comparison on real transactions" view
test_predictions_df = pd.DataFrame({"transaction_id": id_test, "actual_fraud": y_test})
for name, model in trained_models.items():
    if name == "Autoencoder":
        recon = model.predict(X_test_scaled, verbose=0)
        score = np.mean(np.square(X_test_scaled - recon), axis=1)
    elif name in ["Isolation Forest"]:
        score = -model.score_samples(X_test_scaled)
    elif name == "One-Class SVM":
        score = -model.decision_function(X_test_scaled)
    elif name == "Logistic Regression":
        score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        score = model.predict_proba(X_test)[:, 1]
    col = name.lower().replace(" ", "_").replace("-", "_") + "_score"
    test_predictions_df[col] = score

test_predictions_df.to_csv(f"{ARTIFACTS_DIR}/test_predictions.csv", index=False)

print("\nAll models trained and artifacts saved.")
print(f"\nFinal comparison:")
results_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision"]]
print(results_df.to_string())
