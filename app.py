import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="wide")

# -------------------------------
# Load Trained Model and Columns
# -------------------------------
@st.cache_resource
def load_resources():
    model = pickle.load(open("best_rf_model.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, model_columns

model, MODEL_COLUMNS = load_resources()

st.title("📞 Telecom Customer Churn Predictor (Random Forest)")
st.write("Upload the **raw Telco customer dataset (CSV or Excel)** to get churn predictions. The app will apply the same preprocessing used during model training.")

# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload the Telco Customer dataset to begin. Example: WA_Fn-UseC_-Telco-Customer-Churn.csv")
    st.stop()

# Read file
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("🔍 Preview of Uploaded Data")
st.dataframe(df.head())

# ------------------------------------
# Preprocessing (exactly like notebook)
# ------------------------------------
working = df.copy()

# Remove customerID if present
if "customerID" in working.columns:
    working = working.drop("customerID", axis=1)

# TotalCharges -> numeric
if "TotalCharges" in working.columns:
    working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
    working["TotalCharges"].fillna(working["TotalCharges"].median(), inplace=True)

# Replace service values
replace_cols = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]
for col in replace_cols:
    if col in working.columns:
        working[col] = working[col].replace({'No internet service': 'No'})

if "MultipleLines" in working.columns:
    working["MultipleLines"] = working["MultipleLines"].replace({'No phone service': 'No'})

# Remove churn column if exists
for cand in ["Churn", "churn", "CHURN"]:
    if cand in working.columns:
        working = working.drop(columns=[cand])

# Identify numerical + categorical
numeric_cols = []
for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    if c in working.columns:
        numeric_cols.append(c)

categorical_cols = [c for c in working.columns if working[c].dtype == "object"]

# One-hot encode
df_cat = pd.get_dummies(working[categorical_cols].astype(str), drop_first=True)
df_num = working[numeric_cols] if len(numeric_cols) else pd.DataFrame()

df_preprocessed = pd.concat([df_num, df_cat], axis=1)

# Align with training columns
missing_cols = [c for c in MODEL_COLUMNS if c not in df_preprocessed.columns]
for c in missing_cols:
    df_preprocessed[c] = 0

df_preprocessed = df_preprocessed[MODEL_COLUMNS]

st.subheader("🛠️ Transformed Feature Data (First 5 Rows)")
st.dataframe(df_preprocessed.head())

# ------------------------------------
# Predictions
# ------------------------------------
try:
    preds = model.predict(df_preprocessed)
    probs = model.predict_proba(df_preprocessed)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

result = df.copy()
result["Churn Prediction"] = preds
result["Churn Probability"] = probs

st.subheader("🔮 Prediction Results (sample first rows)")
st.dataframe(result.head())

churn_rate = round(result["Churn Prediction"].mean() * 100, 2)
st.metric("📌 Overall Predicted Churn Rate", f"{churn_rate}%")

# ------------------------------------
# Download Button
# ------------------------------------
csv_data = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download Predictions as CSV",
    data=csv_data,
    file_name="churn_predictions.csv",
    mime="text/csv"
)

# ------------------------------------
# FEATURE IMPORTANCE (if available)
# ------------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("🌟 Top 20 Feature Importances")
    importances = model.feature_importances_
    fi = pd.Series(importances, index=MODEL_COLUMNS).sort_values(ascending=False).head(20)
    st.bar_chart(fi)

# ============================================================================================
# 📘 INSIGHT PANEL (DETAILED OPTION-B)
# ============================================================================================

st.subheader("📘 Detailed Churn Insight Report")

high_risk = result[result["Churn Prediction"] == 1]
insight_text = ""

# Contract Type
if "Contract" in df.columns:
    month_to_month_pct = round((high_risk["Contract"] == "Month-to-month").mean() * 100, 2)
    if month_to_month_pct > 50:
        insight_text += f"### 🔥 {month_to_month_pct}% of churners are on Month-to-Month contracts.\n"
        insight_text += "**Solution:** Offer discounts, loyalty credits, or upgrades to annual plans.\n\n"

# Fiber Optic Risk
if "InternetService" in df.columns:
    fiber_pct = round((high_risk["InternetService"] == "Fiber optic").mean() * 100, 2)
    if fiber_pct > 50:
        insight_text += f"### ⚡ {fiber_pct}% of churners use Fiber Optic internet.\n"
        insight_text += "Fiber customers churn due to high bills + expectations.\n"
        insight_text += "**Solution:** Bundle Fiber + Streaming, reduce pricing, improve speed stability.\n\n"

# Monthly Charges
if "MonthlyCharges" in df.columns:
    avg_charge = round(high_risk["MonthlyCharges"].mean(), 2)
    insight_text += f"### 💵 Avg monthly bill of churn-risk customers: **${avg_charge}**\n"
    if avg_charge > 80:
        insight_text += "**Solution:** Offer bill reviews, discounts, loyalty points.\n\n"

# Missing Support Services
support_features = ["OnlineSecurity", "TechSupport", "DeviceProtection", "OnlineBackup"]
for feat in support_features:
    if feat in df.columns:
        pct_no = round((high_risk[feat] == "No").mean() * 100, 2)
        if pct_no > 50:
            insight_text += f"### 🛡️ {pct_no}% of churners have NO {feat}.\n"
            insight_text += f"**Solution:** Provide free trials, low-cost upgrades, or bundled offers for {feat}.\n\n"

# Early Tenure Churn
if "tenure" in df.columns:
    early_pct = round((high_risk["tenure"] < 6).mean() * 100, 2)
    insight_text += f"### 🧭 {early_pct}% of churners leave within their first 6 months.\n"
    insight_text += "**Solution:** Improve onboarding, customer engagement in first 30–60 days.\n\n"

# Show Insights
if insight_text.strip() == "":
    st.info("No major churn patterns detected. Upload a larger dataset for deeper insights.")
else:
    st.markdown(insight_text)

# ============================================================================================
# 💡 RECOMMENDED BUSINESS SOLUTIONS
# ============================================================================================

st.subheader("💡 Recommended Solutions for the Company")

solutions = """
### 🎯 1. Convert Month-to-Month Customers
- Provide special 1-year or 2-year plan discounts  
- Introduce “1 Month Free Upgrade”  
- Push loyalty reward points  

---

### ⚡ 2. Reduce Fiber Optic Churn
- Offer Fiber + Streaming bundles  
- Provide premium support at discounted rates  
- Give speed-boost weekends  

---

### 🛡️ 3. Improve Value-Added Services
Customers lacking security/support churn more.
- Free TechSupport month  
- Free DeviceProtection trial  
- Value packs with OnlineSecurity  

---

### 💰 4. Reduce High Monthly Charges
- Personalized billing plans  
- Loyalty points on high bills  
- Bill optimization reviews  

---

### 🚀 5. Strengthen Early Customer Onboarding
- Welcome call + email sequence  
- First-month discount  
- 24/7 new-user support  

---

### 🔍 6. Predictive Retention Strategy
- Target customers with high churn probability  
- Tailored SMS/email retention campaigns  
- Prioritize high-ARPU customers showing churn signals
"""

st.markdown(solutions)
