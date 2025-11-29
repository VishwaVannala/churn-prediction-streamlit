# Telecom Customer Churn Predictor — Streamlit App

This repo contains a Streamlit app to run churn predictions using a Random Forest model.

## Files
- `app.py` - Streamlit application
- `best_rf_model.pkl` - Pretrained Random Forest model (you already uploaded this)
- `model_columns.pkl` - List of feature column names used by the model (created from your uploaded file)
- `requirements.txt` - Python dependencies

## How it works
- Upload the raw Telco customer dataset (CSV or Excel).
- The app applies preprocessing identical to your training notebook:
  - Drops `customerID` if present
  - Converts `TotalCharges` to numeric and fills missing values with median
  - Replaces 'No internet service' and 'No phone service' with 'No' for specific columns
  - Applies one-hot style encoding to categorical variables, aligns to training columns
- Predictions and churn probabilities are returned for each customer and can be downloaded as CSV.

## Deploy to Streamlit Cloud
1. Create a GitHub repository and push these files.
2. Go to https://share.streamlit.io and create a new app pointing to this repository's `app.py`.
3. Make sure `best_rf_model.pkl` and `model_columns.pkl` are in the repo root.

## Push to GitHub (example)
```bash
git init
git add app.py model_columns.pkl requirements.txt README.md best_rf_model.pkl
git commit -m "Initial Streamlit churn app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repo>` accordingly.