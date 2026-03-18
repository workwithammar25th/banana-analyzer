"""
=============================================================
Model Building based on Banana Freshness Detection and
Shelf Life Prediction using ML
=============================================================
Key fix: Features now correctly use brightness (V channel)
as the primary differentiator between stages, since ALL
banana stages share similar hue (18-45 in HSV).

Stage breakdown by HSV brightness (V):
  Fresh    : V > 180, moderate saturation, some green (H 30-50)
  Ripe     : V > 200, high saturation, pure yellow (H 18-30)
  Overripe : V 80-180, high saturation, dark yellow-brown
  Rotten   : V < 80, any saturation (very dark)
=============================================================
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Model Building based on Banana Freshness Detection")
print("  and Shelf Life Prediction using ML")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Problem Definition
# ─────────────────────────────────────────────
print("\n[STEP 1] Problem Definition")
print("  Classify banana freshness stage from image features")
print("  Predict shelf life (days remaining)")
print("  Detect if image contains a banana at all")
print("  Handle multiple bananas in one image")

# ─────────────────────────────────────────────
# STEP 2: Data Collection
# ─────────────────────────────────────────────
print("\n[STEP 2] Data Collection")
print("  Generating realistic feature dataset...")
print("  Based on real HSV analysis of banana color stages")

np.random.seed(42)
N = 2000

def generate_data(n):
    """
    5 features derived from banana image HSV analysis:
      1. yellow_ratio   : % pixels with H=18-32, S>150, V>150  (ripe yellow)
      2. green_ratio    : % pixels with H=33-55, S>100, V>120  (fresh green)
      3. brown_ratio    : % pixels with H=10-22, S>100, V=40-130 (overripe)
      4. dark_ratio     : % pixels with V < 60                  (rotten/very dark)
      5. brightness_mean: mean V channel value across image
    """
    X, y_labels, y_days = [], [], []

    per_class = n // 4

    # FRESH — mostly green-yellow, high brightness, low brown/dark
    for _ in range(per_class):
        yellow = np.random.uniform(0.15, 0.40)
        green  = np.random.uniform(0.25, 0.55)
        brown  = np.random.uniform(0.00, 0.06)
        dark   = np.random.uniform(0.00, 0.04)
        bright = np.random.uniform(170, 220)
        X.append([yellow, green, brown, dark, bright])
        y_labels.append("Fresh")
        y_days.append(np.random.randint(5, 9))

    # RIPE — mostly bright yellow, minimal green, very low brown/dark
    for _ in range(per_class):
        yellow = np.random.uniform(0.55, 0.85)
        green  = np.random.uniform(0.00, 0.10)
        brown  = np.random.uniform(0.01, 0.08)
        dark   = np.random.uniform(0.00, 0.04)
        bright = np.random.uniform(200, 240)
        X.append([yellow, green, brown, dark, bright])
        y_labels.append("Ripe")
        y_days.append(np.random.randint(2, 5))

    # OVERRIPE — yellow fading, significant brown patches, lower brightness
    for _ in range(per_class):
        yellow = np.random.uniform(0.20, 0.50)
        green  = np.random.uniform(0.00, 0.05)
        brown  = np.random.uniform(0.25, 0.55)
        dark   = np.random.uniform(0.05, 0.18)
        bright = np.random.uniform(100, 175)
        X.append([yellow, green, brown, dark, bright])
        y_labels.append("Overripe")
        y_days.append(np.random.randint(0, 2))

    # ROTTEN — dominated by dark pixels, very low brightness
    for _ in range(per_class):
        yellow = np.random.uniform(0.02, 0.15)
        green  = np.random.uniform(0.00, 0.03)
        brown  = np.random.uniform(0.15, 0.40)
        dark   = np.random.uniform(0.40, 0.85)
        bright = np.random.uniform(25, 95)
        X.append([yellow, green, brown, dark, bright])
        y_labels.append("Rotten")
        y_days.append(0)

    return np.array(X), np.array(y_labels), np.array(y_days)

X, y_labels, y_days = generate_data(N)
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Classes: {np.unique(y_labels)}")

# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Data Preprocessing")

le = LabelEncoder()
y_enc = le.fit_transform(y_labels)
print(f"  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("  Features normalized with MinMaxScaler")

X_tr, X_te, y_tr_c, y_te_c, y_tr_r, y_te_r = train_test_split(
    X_scaled, y_enc, y_days, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

# ─────────────────────────────────────────────
# STEP 4: EDA
# ─────────────────────────────────────────────
print("\n[STEP 4] Exploratory Data Analysis")
feat_names = ["Yellow Ratio", "Green Ratio", "Brown Ratio", "Dark Ratio", "Brightness"]
for i, f in enumerate(feat_names):
    print(f"  {f:15s}: mean={X[:,i].mean():.3f}  std={X[:,i].std():.3f}")

print("\n  Class Distribution:")
for cls, cnt in zip(*np.unique(y_labels, return_counts=True)):
    bar = "█" * (cnt // 25)
    print(f"  {cls:10s}: {cnt}  {bar}")

# ─────────────────────────────────────────────
# STEP 5: Feature Engineering
# ─────────────────────────────────────────────
print("\n[STEP 5] Feature Engineering")
print("  1. yellow_ratio   — H:18-32, S>150, V>150 pixel fraction")
print("  2. green_ratio    — H:33-55, S>100, V>120 pixel fraction")
print("  3. brown_ratio    — H:10-22, S>100, V:40-130 pixel fraction")
print("  4. dark_ratio     — V<60 pixel fraction (decay indicator)")
print("  5. brightness_mean— mean HSV V channel (overall luminance)")

# ─────────────────────────────────────────────
# STEP 6: Model Training
# ─────────────────────────────────────────────
print("\n[STEP 6] Model Training")

clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
clf.fit(X_tr, y_tr_c)
print("  Classifier (Random Forest, 200 trees) trained")

reg = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
reg.fit(X_tr, y_tr_r)
print("  Regressor (Random Forest, 200 trees) trained")

# ─────────────────────────────────────────────
# STEP 7: Evaluation
# ─────────────────────────────────────────────
print("\n[STEP 7] Model Evaluation")

y_pred = clf.predict(X_te)
acc = accuracy_score(y_te_c, y_pred)
print(f"\n  Classifier Accuracy : {acc*100:.2f}%")
print(f"\n{classification_report(y_te_c, y_pred, target_names=le.classes_)}")

from sklearn.metrics import mean_absolute_error, r2_score
y_pred_r = np.clip(reg.predict(X_te).round().astype(int), 0, 10)
print(f"  Regressor MAE : {mean_absolute_error(y_te_r, y_pred_r):.2f} days")
print(f"  Regressor R²  : {r2_score(y_te_r, y_pred_r):.4f}")

print("\n  Feature Importances:")
for name, imp in sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "▓" * int(imp * 50)
    print(f"  {name:16s}: {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# STEP 8: Save Models
# ─────────────────────────────────────────────
print("\n[STEP 8] Saving Models")
os.makedirs("model", exist_ok=True)
for name, obj in [("classifier", clf), ("regressor", reg), ("scaler", scaler), ("label_encoder", le)]:
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    print(f"  model/{name}.pkl saved")

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETE. Run: python app.py")
print("=" * 60)
