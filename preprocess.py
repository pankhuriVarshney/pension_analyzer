import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("running...")

# =====================
# 1. Load Data
# =====================
# Make sure you have installed openpyxl -> pip install openpyxl
original_df = pd.read_excel("pension_data.xlsx", engine="openpyxl")

# Keep a copy for plotting categorical counts before encoding
df = original_df.copy()

# Drop non-informative columns
drop_cols = [
    'User_ID', 'Transaction_ID', 'IP_Address', 'Device_ID',
    'Geo_Location', 'Transaction_Date'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =====================
# 2. Handle Missing Values
# =====================
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col != 'Projected_Pension_Amount']

categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# =====================
# 3. Encode Categorical Variables
# =====================
label_enc_cols = []  # binary categories
one_hot_cols = []    # multi-class

for col in categorical_cols:
    if df[col].nunique() <= 2:
        label_enc_cols.append(col)
    else:
        one_hot_cols.append(col)

le = LabelEncoder()
for col in label_enc_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)


# Separate targets
REG_TARGET = "Projected_Pension_Amount"
CLASS_TARGET = "Suspicious_Flag"

y_reg = df[REG_TARGET].values.reshape(-1, 1)
X = df.drop(columns=[REG_TARGET, CLASS_TARGET], errors='ignore')

# Scale X
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

# Scale y
y_scaler = StandardScaler()
y_reg_scaled = y_scaler.fit_transform(y_reg)

# Save scalers
joblib.dump(x_scaler, "outputs/x_scaler.pkl")
joblib.dump(y_scaler, "outputs/y_scaler.pkl")

# Save processed dataset for training
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df[REG_TARGET] = y_reg_scaled
if CLASS_TARGET in df.columns:
    processed_df[CLASS_TARGET] = df[CLASS_TARGET].values
processed_df.to_csv("pension_data_processed.csv", index=False)
print("✅ Preprocessing complete. Saved 'pension_data_processed.csv'.")

# =====================
# 6. Visualization
# =====================
# Distribution plots for numeric columns
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols[:6]):  # first 6 numeric columns
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("numeric_distributions.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Category counts from original data (before encoding)
plt.figure(figsize=(15, 8))
for i, col in enumerate(original_df.select_dtypes(include=['object']).columns[:6]):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=col, data=original_df)
    plt.xticks(rotation=45)
    plt.title(f"Count of {col}")
plt.tight_layout()
plt.savefig("categorical_counts.png")
plt.show()

print("📊 Graphs saved: numeric_distributions.png, correlation_heatmap.png, categorical_counts.png")
