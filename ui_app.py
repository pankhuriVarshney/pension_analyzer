import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from transformers import AutoModelForCausalLM, AutoTokenizer


st.set_page_config(page_title="Pension AI Insights", layout="wide")

# === Load models, feature list, and scaler ===
@st.cache_resource
def load_assets():
    reg_model = joblib.load("outputs/tabnet_reg.pkl")
    xgb_model = joblib.load("outputs/xgb_clf.pkl")
    y_scaler = joblib.load("outputs/y_scaler.pkl")  # NEW — Load target scaler
    with open("outputs/feature_columns.txt", "r") as f:
        features = f.read().splitlines()
    return reg_model, xgb_model, y_scaler, features

reg_model, xgb_model, y_scaler, feature_columns = load_assets()

# === Sidebar settings ===
st.sidebar.header("Settings")
user_type = st.sidebar.selectbox("Select User Type", ["Member", "Advisor", "Regulator"])
uploaded_file = st.sidebar.file_uploader("Upload Pension Data (CSV/XLSX)", type=["csv", "xlsx"])

# === Load data ===
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.warning("No file uploaded — showing insights from training dataset.")
    if os.path.exists("pension_data_processed.csv"):
        df = pd.read_csv("pension_data_processed.csv")
    else:
        st.error("Training dataset not found.")
        st.stop()

# === Targets ===
REG_TARGET = "Projected_Pension_Amount"
CLASS_TARGET = "Suspicious_Flag"

# Ensure User_ID exists
if 'User_ID' not in df.columns:
    df['User_ID'] = df.index.astype(str)

# Encode Suspicious_Flag if exists
has_class_target = CLASS_TARGET in df.columns
if has_class_target and df[CLASS_TARGET].dtype == object:
    le_sf = LabelEncoder()
    df[CLASS_TARGET] = le_sf.fit_transform(df[CLASS_TARGET].astype(str))

# Drop identifier columns
drop_candidate_cols = ['Transaction_ID', 'IP_Address', 'Device_ID', 'Geo_Location', 'Time_of_Transaction']
for c in drop_candidate_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# === Prepare feature matrix ===
features = [c for c in df.columns if c not in [REG_TARGET, CLASS_TARGET, 'User_ID']]
X = df[features].astype(np.float32).values

# === Run predictions and inverse-transform ===
st.subheader(f"Insights for: **{user_type}**")
pension_preds_scaled = reg_model.predict(X).flatten()
pension_preds = y_scaler.inverse_transform(pension_preds_scaled.reshape(-1, 1)).flatten()  # Unscaled
pension_preds = np.clip(pension_preds, 0, None)  # Avoid negative pensions

fraud_preds = xgb_model.predict_proba(X)[:, 1] if has_class_target else None

# === Model Accuracy Section ===
if REG_TARGET in df.columns:
    y_true_scaled = df[REG_TARGET].values
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()  # Unscaled true values
    r2 = r2_score(y_true, pension_preds)
    mae = mean_absolute_error(y_true, pension_preds)
    rmse = np.sqrt(mean_squared_error(y_true, pension_preds))
    st.write(f"**Regression Model (Pension Prediction)** — R²: {r2:.3f}, MAE: {mae:,.2f}, RMSE: {rmse:,.2f}")

if has_class_target:
    y_class = df[CLASS_TARGET].values
    threshold = 0.3
    class_preds = (fraud_preds >= threshold).astype(int)
    acc = accuracy_score(y_class, class_preds)
    prec = precision_score(y_class, class_preds)
    rec = recall_score(y_class, class_preds)
    f1 = f1_score(y_class, class_preds)
    auc = roc_auc_score(y_class, fraud_preds)
    cm = confusion_matrix(y_class, class_preds)
    # st.write("Confusion Matrix:")
    # st.write(cm)
    # st.write(f"**Classification Model (Fraud Detection)** — Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

# === Summary metrics ===
col1, col2, col3 = st.columns(3)
col1.metric("Avg Projected Pension", f"${np.mean(pension_preds):,.2f}")
col2.metric("Fraud Risk", f"{np.mean(fraud_preds)*100:.1f}%" if fraud_preds is not None else "N/A")
col3.metric("Sample Size", len(df))

# === Insights in plain language ===
st.write("### Interpretation")
avg_pension = np.mean(pension_preds)
avg_fraud = np.mean(fraud_preds) * 100 if fraud_preds is not None else 0
if user_type == "Member":
    st.info(f"As a member, your predicted pension average is **${avg_pension:,.2f}**. This is your estimated retirement payout based on current data.")
elif user_type == "Advisor":
    st.info(f"As an advisor, the model predicts an average client pension of **${avg_pension:,.2f}**. High volatility portfolios may lead to a wider spread in pension outcomes.")
elif user_type == "Regulator":
    st.info(f"As a regulator, the model estimates an average fraud probability of **{avg_fraud:.2f}%** across the dataset, highlighting potential high-risk cases.")

# === Visualizations ===
if user_type == "Member":
    st.write("### Pension Projection Distribution")
    fig, ax = plt.subplots()
    sns.histplot(pension_preds, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

elif user_type == "Advisor":
    if "Volatility" in df.columns:
        st.write("### Portfolio Risk vs Return")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["Volatility"], y=pension_preds, ax=ax)
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Predicted Pension")
        st.pyplot(fig)

elif user_type == "Regulator" and fraud_preds is not None:
    st.write("### Fraud Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(fraud_preds, bins=20, kde=True, ax=ax, color="red")
    st.pyplot(fig)

from transformers import pipeline

summary_model = pipeline("text-generation", model="gpt2")  # Or a fine-tuned summarizer

# from ollama import Ollama
# from langchain.schema import SystemMessage, HumanMessage, BaseMessage
# from typing import List
# import numpy as np

# def langchain_messages_to_ollama(messages: List[BaseMessage]) -> List[dict]:
#     role_map = {
#         "system": "system",
#         "human": "user",
#         "ai": "assistant"
#     }
#     return [{"role": role_map.get(msg.type, "user"), "content": msg.content} for msg in messages]

# def narrate_insights(user_type, pension_preds, fraud_preds):
#     avg_pen = np.mean(pension_preds)
#     avg_frisk = np.mean(fraud_preds) * 100

#     system_prompt = SystemMessage(
#         content=(
#             "You are a financial and fraud risk insights assistant. "
#             "Provide concise, clear, and user-friendly insights based on the given data."
#         )
#     )

#     user_prompt = HumanMessage(
#         content=f"""
# User Type: {user_type}
# Avg Pension: ${avg_pen:,.2f}
# Fraud Risk: {avg_frisk:.1f}%

# Provide a one-paragraph insight tailored for the user.
# """
#     )

#     client = Ollama()

#     messages_for_ollama = langchain_messages_to_ollama([system_prompt, user_prompt])

#     response = client.chat(
#         model="llama2",
#         messages=messages_for_ollama,
#         temperature=0.7
#     )

#     return response


def generate_summary(user_type, pension_preds, fraud_preds):
    avg_pen = np.mean(pension_preds)
    avg_frisk = np.mean(fraud_preds) * 100

    if user_type == "Member":
        return f"As a member, you're projected to receive around ${avg_pen:,.2f} annually. This estimate reflects your contribution history and adjusted portfolio growth. Fraud risk appears low at {avg_frisk:.1f}%."
    elif user_type == "Advisor":
        return f"Your client base shows a projected pension average of ${avg_pen:,.2f}, with risk clustering below {avg_frisk:.1f}%. Consider reviewing high-volatility accounts for better yield."
    elif user_type == "Regulator":
        return f"The system flags an overall fraud probability of {avg_frisk:.1f}%, with projected pensions averaging ${avg_pen:,.2f}. This insight supports proactive auditing in sensitive zones."
# # === Feature Importance ===
# st.write("### NLP Summary")
# st.info(narrate_insights(user_type, pension_preds, fraud_preds))

st.write("### XGBoost Feature Importance")
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# === SHAP plots ===
st.write("### Explainability")
shap_reg_path = "outputs/shap_regressor.png"
shap_xgb_path = "outputs/shap_xgb.png"
if os.path.exists(shap_reg_path):
    st.image(shap_reg_path, caption="SHAP Summary - Pension Amount (TabNet)")
if os.path.exists(shap_xgb_path):
    st.image(shap_xgb_path, caption="SHAP Summary - Fraud Probability (XGB)")

# === Data Preview ===
st.write("### Data Preview")
st.dataframe(df.head())
