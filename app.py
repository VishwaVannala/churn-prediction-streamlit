import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_resources():
    model = pickle.load(open("best_rf_model.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, model_columns

model, MODEL_COLUMNS = load_resources()

st.title("📞 Telecom Customer Churn Predictor (Random Forest)")
st.write("Upload the raw Telco customer dataset (CSV or Excel) to get churn predictions. The app will apply the same preprocessing used during model training.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("Upload your dataset to begin. Example file: WA_Fn-UseC_-Telco-Customer-Churn.csv")
else:
    # Read file
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # ------------------- preprocessing (match training notebook) -------------------
    working = df.copy()

    # drop customerID if present
    if "customerID" in working.columns:
        working = working.drop("customerID", axis=1)

    # TotalCharges -> numeric, fillna with median
    if "TotalCharges" in working.columns:
        working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
        working["TotalCharges"].fillna(working["TotalCharges"].median(), inplace=True)

    # Replace 'No internet service' and 'No phone service' with 'No' for specific cols
    replace_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    for col in replace_cols:
        if col in working.columns:
            working[col] = working[col].replace({'No internet service': 'No'})

    if "MultipleLines" in working.columns:
        working["MultipleLines"] = working["MultipleLines"].replace({'No phone service': 'No'})

    # Remove target column if present
    target_col = None
    for cand in ["Churn", "churn", "CHURN"]:
        if cand in working.columns:
            target_col = cand
            working = working.drop(columns=[cand])
            break

    # Identify numeric and categorical columns
    numeric_cols = [c for c in working.columns if working[c].dtype in [np.float64, np.int64] and c in ["tenure","MonthlyCharges","TotalCharges"]]
    # Fallback: ensure these numeric columns exist
    for n in ["tenure","MonthlyCharges","TotalCharges"]:
        if n in working.columns and n not in numeric_cols:
            numeric_cols.append(n)

    categorical_cols = [c for c in working.columns if working[c].dtype == "object"]

    # Apply get_dummies to categorical columns (drop_first to mimic OneHotEncoder(drop='first'))
    df_cat = pd.get_dummies(working[categorical_cols].astype(str), drop_first=True) if len(categorical_cols)>0 else pd.DataFrame(index=working.index)
    df_num = working[numeric_cols] if len(numeric_cols)>0 else pd.DataFrame(index=working.index)

    df_preprocessed = pd.concat([df_num, df_cat], axis=1)

    # Align with training columns
    # MODEL_COLUMNS is expected to be a list of the final training feature names (one-hot expanded)
    missing_cols = [c for c in MODEL_COLUMNS if c not in df_preprocessed.columns]
    extra_cols = [c for c in df_preprocessed.columns if c not in MODEL_COLUMNS]

    # Add missing columns with zeros
    for c in missing_cols:
        df_preprocessed[c] = 0

    # Finally, reorder to MODEL_COLUMNS exactly
    df_preprocessed = df_preprocessed[MODEL_COLUMNS]

    st.subheader("Transformed feature preview (first 5 rows)")
    st.dataframe(df_preprocessed.head())

    # ------------------- predictions -------------------
    try:
        preds = model.predict(df_preprocessed)
        probs = model.predict_proba(df_preprocessed)[:,1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    result = df.copy()
    result["Churn Prediction"] = preds
    result["Churn Probability"] = probs

    st.subheader("Prediction sample")
    st.dataframe(result.head())

    churn_rate = round(result["Churn Prediction"].mean()*100,2)
    st.metric("Predicted churn rate", f"{churn_rate}%")

    # Download button
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

    # Optionally show feature importance if available
    try:
        if hasattr(model, "feature_importances_"):
            st.subheader("Model Feature Importances (top 20)")
            importances = model.feature_importances_
            fi = pd.Series(importances, index=MODEL_COLUMNS).sort_values(ascending=False).head(20)
            st.bar_chart(fi)
    except Exception:
        pass