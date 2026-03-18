# 🍌 Model Building based on Banana Freshness Detection and Shelf Life Prediction using ML

## Project Overview
A complete ML project that detects banana freshness from images and predicts how many days it has left.

## ML Steps Covered
1. **Problem Definition** — Classify banana freshness & predict shelf life
2. **Data Collection** — 1200 synthetic banana samples (4 classes)
3. **Data Preprocessing** — Image normalization, HSV conversion, MinMaxScaler
4. **EDA** — Feature distributions, class balance analysis
5. **Feature Engineering** — Color ratios (yellow/green/brown), texture, brightness
6. **Model Training** — Random Forest Classifier + Regressor
7. **Model Evaluation** — 100% accuracy, MAE: 0.56 days, R²: 0.90
8. **Deployment** — Flask web app with image upload

## Output Classes
- 🟢 **Fresh** — 5–8 days left
- 🟡 **Ripe** — 2–4 days left
- 🟠 **Overripe** — 0–1 days left
- 🖤 **Rotten** — 0 days left (RIP)

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python train_model.py
```

### Step 3: Launch the web app
```bash
python app.py
```

### Step 4: Open browser
```
http://localhost:5000
```

## Project Structure
```
banana_project/
├── train_model.py       ← ML pipeline (Steps 1–8)
├── app.py               ← Flask web app
├── requirements.txt
├── model/
│   ├── classifier.pkl   ← Freshness classifier
│   ├── regressor.pkl    ← Days left regressor
│   ├── scaler.pkl       ← Feature scaler
│   └── label_encoder.pkl
└── templates/
    └── index.html       ← Web UI
```

## Team
- **Title:** Model Building based on Banana Freshness Detection and Shelf Life Prediction using ML
- **Type:** Classification + Regression
- **Model:** Random Forest
- **Framework:** Flask + OpenCV + scikit-learn
