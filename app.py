"""
Model Building based on Banana Freshness Detection and Shelf Life Prediction using ML
Flask Web Application
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import cv2
import os
import random

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def _load(name):
    for d in [BASE, os.path.join(BASE, "model"), "model", "."]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"{name} not found. DIR={BASE} FILES={os.listdir(BASE)}")

clf    = _load("model/classifier.pkl")
reg    = _load("model/regressor.pkl")
scaler = _load("model/scaler.pkl")
le     = _load("model/label_encoder.pkl")

# ─────────────────────────────────────────
# Dark humour comments
# ─────────────────────────────────────────
COMMENTS = {
    "Fresh": [
        "Congratulations, it's alive. Enjoy it before it starts making plans to die.",
        "Peak youth. It will never be this good again. Neither will you, probably.",
        "Still has all its dreams intact. Tragic, given what's coming.",
        "Young, firm, full of potential. Give it a week.",
        "Practically a newborn. Try not to outlive it.",
    ],
    "Ripe": [
        "This is it. The moment. Stop reading and eat the banana.",
        "Nature spent days perfecting this. You have 48 hours. No pressure.",
        "Maximum sweetness achieved. Everything from here is just entropy.",
        "The banana equivalent of living your best life. It ends tomorrow.",
        "Perfect. Which means it is already starting to not be perfect.",
    ],
    "Overripe": [
        "It is not dead, it is just given up on being a snack. Make banana bread.",
        "One foot in the bin. The other in a recipe from 2009 you will never make.",
        "Still technically edible, which is the lowest bar we set for food.",
        "Past its prime. Relatable, honestly.",
        "This banana has accepted its fate. The question is whether you have.",
    ],
    "Rotten": [
        "This banana is legally deceased. Please respect its memory.",
        "It has returned to the earth. Spiritually at least. Physically it is in your kitchen.",
        "The decomposition is complete. Time of death: approximately 4 days ago.",
        "Not a banana anymore. Just a vibe, and a bad one at that.",
        "Even the fruit flies are looking elsewhere. That says something.",
    ],
}

SAFETY = {
    "Fresh":    {"safe": True,  "pill": "Safe to Eat",    "action": "Store at room temperature. Eat within 8 days."},
    "Ripe":     {"safe": True,  "pill": "Eat Now",        "action": "Eat today or tomorrow. Peak sweetness, do not wait."},
    "Overripe": {"safe": True,  "pill": "Barely Safe",    "action": "Too soft to eat directly. Use in baking only."},
    "Rotten":   {"safe": False, "pill": "Discard",        "action": "Do not consume. Dispose of immediately."},
}


# ─────────────────────────────────────────
# SHAPE DETECTION
# Bananas: low circularity (curved/elongated) + low-medium solidity
# Round fruits (orange, apple): high circularity + high solidity
# ─────────────────────────────────────────
def is_banana_shape(contour):
    """
    Validate contour shape is banana-like using two complementary metrics:

    1. Circularity = 4π·area / perimeter²
       - Perfect circle = 1.0 (orange, apple)
       - Curved/elongated = 0.2–0.55 (real banana photos)
       - Rectangles = 0.5–0.8

    2. Min-area rectangle aspect ratio = long_side / short_side
       - Circle/square ≈ 1.0 (orange, apple)
       - Elongated banana ≥ 1.5 (often 2.0–4.0 in real photos)

    Rule: banana if (circularity < 0.65) OR (rect_aspect > 1.5)
    This handles both real curved-banana photos AND synthetic elongated shapes.
    Reliably rejects round fruits (orange, apple, grape) which have
    circularity ~0.75–0.95 AND rect_aspect ~1.0.
    """
    area = cv2.contourArea(contour)
    if area < 200:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rect_aspect = max(rw, rh) / (min(rw, rh) + 0.001)

    return (circularity < 0.65) or (rect_aspect > 1.5)


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# All ratios computed over banana pixels ONLY — ignores background
# ─────────────────────────────────────────
def extract_features(img_bgr, mask=None):
    """
    Extract colour features relative to banana pixels only.
    This handles any background colour (black, white, busy).

    If a contour mask is provided, features are computed within that region.
    Otherwise, banana pixels are auto-detected from colour.
    """
    img = cv2.resize(img_bgr, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    if mask is not None:
        mask_r = cv2.resize(mask.astype(np.uint8), (300, 300))
        region = mask_r > 0
    else:
        region = (H >= 10) & (H <= 55) & (S > 80) & (V > 50)

    banana_count = float(np.sum(region))
    if banana_count < 100:
        # Fallback: use all pixels
        region = np.ones_like(H, dtype=bool)
        banana_count = float(H.size)

    # Colour ratios — denominator is banana pixels ONLY
    yellow = (H >= 18) & (H <= 32) & (S > 150) & (V > 150) & region
    green  = (H >= 33) & (H <= 55) & (S > 100) & (V > 120) & region
    brown  = (H >= 10) & (H <= 22) & (S > 100) & (V >= 40) & (V <= 130) & region
    dark   = (V < 60) & region

    yellow_ratio = float(np.sum(yellow)) / banana_count
    green_ratio  = float(np.sum(green))  / banana_count
    brown_ratio  = float(np.sum(brown))  / banana_count
    dark_ratio   = float(np.sum(dark))   / banana_count
    brightness   = float(np.mean(V[region]))

    return [yellow_ratio, green_ratio, brown_ratio, dark_ratio, brightness]


def predict_crop(crop_bgr, mask=None):
    feats  = extract_features(crop_bgr, mask)
    scaled = scaler.transform([feats])
    stage  = le.inverse_transform([clf.predict(scaled)[0]])[0]
    days   = max(0, int(round(reg.predict(scaled)[0])))
    return stage, days, feats


# ─────────────────────────────────────────
# MULTI-BANANA DETECTION
# Finds individual banana regions using contour + shape validation
# ─────────────────────────────────────────
def detect_bananas(img_bgr):
    """
    Detect individual banana regions in the image.

    Returns list of dicts: { 'crop': img, 'mask': binary_mask }
    Each dict represents one detected banana.

    Strategy:
    1. Build colour mask for banana-range pixels
    2. Find contours with small morphological kernel (avoids merging separate bananas)
    3. Validate each contour with shape analysis (rejects round fruits)
    4. Return cropped regions with their masks
    """
    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Colour mask: banana hue range, non-background saturation and brightness
    colour_mask = ((H >= 10) & (H <= 55) & (S > 70) & (V > 40)).astype(np.uint8) * 255

    # Small kernel: fills holes within a banana but doesn't bridge gaps between bananas
    kernel = np.ones((12, 12), np.uint8)
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN,  np.ones((6, 6), np.uint8))

    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.01   # at least 1% of image
    max_area = h * w * 0.98   # not the entire image

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Shape check — reject round fruits
        if not is_banana_shape(cnt):
            continue

        # Crop with padding
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        crop = img_bgr[y1:y2, x1:x2]

        # Binary mask for this contour (for background-independent features)
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        crop_mask = region_mask[y1:y2, x1:x2]

        results.append({'crop': crop, 'mask': crop_mask})

    return results


def whole_image_is_banana(img_bgr):
    """
    Fallback check: if no individual regions are found,
    check if the whole image is one big banana (fills most of frame).
    Also validates shape of the whole banana blob.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    colour_mask = ((H >= 10) & (H <= 55) & (S > 70) & (V > 40)).astype(np.uint8) * 255
    kernel = np.ones((25, 25), np.uint8)
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)

    banana_ratio = np.sum(colour_mask > 0) / (h * w)

    if banana_ratio < 0.05:
        return False

    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    # Always validate shape — never skip for oranges/apples even if they fill the frame
    return is_banana_shape(largest)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Detect banana regions
    regions = detect_bananas(img)

    # If no regions found, try whole-image fallback
    if not regions:
        if whole_image_is_banana(img):
            regions = [{'crop': img, 'mask': None}]
        else:
            return jsonify({
                "is_banana": False,
                "message": "No banana detected. The shape and colour do not match a banana. This is a very focused operation."
            })

    # Predict each region
    bananas = []
    for r in regions:
        stage, days, feats = predict_crop(r['crop'], r['mask'])
        info = SAFETY[stage]
        bananas.append({
            "freshness":    stage,
            "days_left":    days,
            "safe":         info["safe"],
            "pill":         info["pill"],
            "action":       info["action"],
            "comment":      random.choice(COMMENTS[stage]),
            "features": {
                "yellow_pct": round(feats[0] * 100, 1),
                "green_pct":  round(feats[1] * 100, 1),
                "brown_pct":  round(feats[2] * 100, 1),
                "dark_pct":   round(feats[3] * 100, 1),
                "brightness": round(feats[4], 0),
            }
        })

    return jsonify({
        "is_banana": True,
        "bananas":   bananas,
        "count":     len(bananas)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
