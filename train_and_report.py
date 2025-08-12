"""
train_and_report.py

Inputs:
 - pension_data_processed.csv : preprocessed data created by your preprocessing script.
Outputs:
 - saved models: tabnet_regressor.zip, xgb_suspicious.model
 - reports folder: reports/<User_ID>_<role>/ (plots + text summary)
 - logs and saved plots in outputs/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import joblib

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# XGBoost
import xgboost as xgb

# SHAP for explainability
import shap

# -------- config ----------
DATA_PATH = "pension_data_processed.csv"   # produced by your preprocessing step
OUTPUT_DIR = "outputs"
REPORTS_DIR = "reports"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------- helpers ----------
def ensure_col(df, col, default=0):
    if col not in df.columns:
        df[col] = default
    return df

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# -------- load data ----------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# If you still have the original User_ID column removed, try to recover user index
if 'User_ID' not in df.columns:
    # try to use index as UserID
    df['User_ID'] = df.index.astype(str)

# Determine targets
# Regression target: Projected_Pension_Amount (if present)
REG_TARGET = "Projected_Pension_Amount"
if REG_TARGET not in df.columns:
    raise SystemExit(f"Regression target column '{REG_TARGET}' not found in {DATA_PATH}")

# Classification target: Suspicious_Flag (if present)
CLASS_TARGET = "Suspicious_Flag"
has_class_target = CLASS_TARGET in df.columns

# If Suspicious_Flag is non-binary strings like Yes/No, encode to 0/1
if has_class_target:
    if df[CLASS_TARGET].dtype == object:
        le_sf = LabelEncoder()
        df[CLASS_TARGET] = le_sf.fit_transform(df[CLASS_TARGET].astype(str))
        # store mapping
        sf_mapping = dict(zip(le_sf.classes_, le_sf.transform(le_sf.classes_)))
        print("Suspicious_Flag mapping:", sf_mapping)

# Keep user id column separately for reporting
user_ids = df['User_ID'].astype(str).values

# Drop any leftover columns that are purely identifiers (but keep User_ID for reporting)
drop_candidate_cols = ['Transaction_ID', 'IP_Address', 'Device_ID', 'Geo_Location', 'Time_of_Transaction']
for c in drop_candidate_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# Prepare features & targets
features = [c for c in df.columns if c not in [REG_TARGET, CLASS_TARGET, 'User_ID']]
X = df[features].values
feature_columns = list(df[features].columns)  # save the dataframe column names here

# Save to file
with open("outputs/feature_columns.txt", "w") as f:
    f.write("\n".join(feature_columns))

y_scaler = joblib.load("outputs/y_scaler.pkl")
y_reg = df[REG_TARGET].values.reshape(-1, 1)  # scaled already in CSV

if has_class_target:
    y_class = df[CLASS_TARGET].values

# Train/test split
X_train, X_test, y_reg_train, y_reg_test, idx_train, idx_test = train_test_split(
    X, y_reg, df.index.values, test_size=0.2, random_state=RANDOM_STATE
)

X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)


y_reg_train=y_reg_train.reshape(-1,1).astype(np.float32)
y_reg_test=y_reg_test.reshape(-1,1).astype(np.float32)
if has_class_target:
    # need classification y aligned to split — do a split where classification target exists:
    Xc = df[features].values
    yc = df[CLASS_TARGET].values
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_STATE
    )

# ---------------------------
# 1) Train TabNet Regressor
# ---------------------------
print("Training TabNet regressor for", REG_TARGET)
tabnet_params = dict(
    n_d=16, n_a=16, n_steps=5,
    gamma=1.3, lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
    device_name='cuda' if False else 'cpu'   # change to 'cuda' if you have GPU + torch installed properly
)

regressor = TabNetRegressor(**{k:v for k,v in tabnet_params.items() if k in TabNetRegressor.__init__.__code__.co_varnames})
# Note: TabNetRegressor accepts many params; above are sensible defaults.

regressor.fit(
    X_train=X_train, y_train=y_reg_train,
    eval_set=[(X_test, y_reg_test)],
    eval_name=['valid'],
    eval_metric=['mae'],
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

# Save tabnet
tabnet_save_path = os.path.join(OUTPUT_DIR, "tabnet_regressor.zip")
regressor.save_model(tabnet_save_path)
print("Saved TabNet model to:", tabnet_save_path)

# Evaluate regressor
y_reg_pred = regressor.predict(X_test).squeeze()
y_reg_pred_unscaled = y_scaler.inverse_transform(y_reg_pred.reshape(-1, 1)).flatten()
y_reg_test_unscaled = y_scaler.inverse_transform(y_reg_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_reg_test_unscaled, y_reg_pred_unscaled)
r2 = r2_score(y_reg_test_unscaled, y_reg_pred_unscaled)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"Regression MAE: {mae:.2f}, R2: {r2:.3f}")

# Feature importance from TabNet
feat_importances = regressor.feature_importances_  # aligned to features list
feat_imp_df = pd.DataFrame({"feature": features, "importance": feat_importances}).sort_values("importance", ascending=False)
feat_imp_df.to_csv(os.path.join(OUTPUT_DIR, "tabnet_feature_importances.csv"), index=False)

# Plot top 20 importances
fig = plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=feat_imp_df.head(20))
plt.title("TabNet Feature Importances (top 20)")
save_fig(fig, os.path.join(OUTPUT_DIR, "tabnet_feature_importances_top20.png"))

# ---------------------------
# 2) Train XGBoost classifier for Suspicious_Flag (if exists)
# ---------------------------
if has_class_target:
    print("Training XGBoost classifier for", CLASS_TARGET)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    xgb_clf.set_params(early_stopping_rounds=20)

    xgb_clf.fit(Xc_train, yc_train, eval_set=[(Xc_test, yc_test)], verbose=False)
    xgb_path = os.path.join(OUTPUT_DIR, "xgb_suspicious.model")
    xgb_clf.save_model(xgb_path)
    print("Saved XGBoost model to:", xgb_path)

    # Evaluate
    y_class_pred = xgb_clf.predict(Xc_test)
    y_class_proba = xgb_clf.predict_proba(Xc_test)[:,1] if hasattr(xgb_clf, "predict_proba") else None
    acc = accuracy_score(yc_test, y_class_pred)
    f1 = f1_score(yc_test, y_class_pred, zero_division=0)
    auc = roc_auc_score(yc_test, y_class_proba) if y_class_proba is not None else None
    print(f"Classification Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}" if auc is not None else f"Acc: {acc:.3f}, F1: {f1:.3f}")

    # XGBoost feature importance
    xgb_imp = xgb_clf.get_booster().get_score(importance_type='gain')
    xgb_imp_df = pd.DataFrame([{"feature":k, "importance":v} for k,v in xgb_imp.items()]).sort_values("importance", ascending=False)
    xgb_imp_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_feature_importances.csv"), index=False)

    fig = plt.figure(figsize=(8,6))
    if not xgb_imp_df.empty:
        sns.barplot(x="importance", y="feature", data=xgb_imp_df.head(20))
        plt.title("XGBoost Feature Importances (top 20)")
        save_fig(fig, os.path.join(OUTPUT_DIR, "xgb_feature_importances_top20.png"))

# ---------------------------
# 3) SHAP explainability
# ---------------------------
print("Preparing SHAP explainability (this may take time)...")
# Use a small sample for the explainer background to speed up
background = X_train[np.random.choice(X_train.shape[0], min(200, X_train.shape[0]), replace=False)]

# TabNet SHAP (use TreeExplainer won't work; use KernelExplainer or the regressor's internal masks)
# For speed, we'll use a KernelExplainer on a single instance when generating a report
# For global explanation we can approximate via feature_importances_ saved earlier.

# XGBoost SHAP explainer (fast, tree-based)
if has_class_target:
    xgb_explainer = shap.TreeExplainer(xgb_clf)
    # compute shap values for test set subset and save
    shap_vals = xgb_explainer.shap_values(Xc_test[:200])
    # save a summary plot
    shap.summary_plot(shap_vals, features=np.array(features), show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_shap_summary.png"), bbox_inches='tight', dpi=150)
    plt.close()

# ---------------------------
# 4) Reporting per stakeholder
# ---------------------------

def generate_user_report(user_index, role='member'):
    """
    role: 'member' | 'advisor' | 'regulator'
    user_index: integer index in original df (not row number in processed X)
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    uid = str(df.loc[user_index, 'User_ID'])
    rep_dir = os.path.join(REPORTS_DIR, f"{uid}_{role}")
    os.makedirs(rep_dir, exist_ok=True)

    # Build user feature vector (must match features order)
    row = df.loc[user_index]
    x_user = row[features].values.reshape(1, -1)

    # 1) Regression prediction (projected pension amount)
   # Load the exact training feature list
    with open("outputs/feature_columns.txt", "r") as f:
        feature_columns = f.read().splitlines()

    # Build x_user exactly like training data
    x_user = df.iloc[[idx]][feature_columns]  # preserve order & columns
    x_user = x_user.astype(float).to_numpy()

    pred_amount = regressor.predict(x_user).squeeze()

    # For scaled output: TabNet trained on preprocessed scaled numeric values; if your preprocessing scaled target,
    # you may need to inverse-transform. This script assumes your target was numeric raw value.

    # Plot projected pension gauge (simple bar + expected payout)
    fig = plt.figure(figsize=(6,3))
    plt.barh([0], [pred_amount], height=0.6)
    plt.xlim(0, max(pred_amount*1.6, pred_amount+1000))
    plt.title(f"Projected Pension Amount: {pred_amount:,.0f}")
    plt.yticks([])
    save_fig(fig, os.path.join(rep_dir, "projected_pension_amount.png"))

    # 2) Feature importance local (TabNet provides masks per prediction via explain - use explain here)
    explanation = regressor.explain(x_user)
    # explanation[0] is masks; compute average mask importance per feature across steps
    masks = np.mean(np.mean(explanation[0], axis=1), axis=0)  # [features]
    local_imp = pd.DataFrame({"feature": features, "mask": masks}).sort_values("mask", ascending=False)
    local_imp.to_csv(os.path.join(rep_dir, "local_tabnet_importances.csv"), index=False)

    fig = plt.figure(figsize=(6,6))
    sns.barplot(x="mask", y="feature", data=local_imp.head(15))
    plt.title("Local TabNet importances (top 15)")
    save_fig(fig, os.path.join(rep_dir, "local_tabnet_importances.png"))

    # 3) SHAP explanation for classification if role regulator and model exists
    if has_class_target and role == 'regulator':
        # XGBoost SHAP for this user
        shap_values_user = xgb_explainer.shap_values(x_user)
        # use a SHAP force plot or waterfall (matplotlib based)
        try:
            shap.waterfall_plot(shap.Explanation(values=shap_values_user[0] if isinstance(shap_values_user, list) else shap_values_user,
                                                 base_values=xgb_explainer.expected_value,
                                                 data=x_user[0],
                                                 feature_names=features))
            plt.savefig(os.path.join(rep_dir, "xgb_shap_waterfall.png"), bbox_inches='tight', dpi=150)
            plt.close()
        except Exception:
            # fallback: bar of abs mean shap
            sv = np.abs(shap_values_user).mean(axis=0) if isinstance(shap_values_user, np.ndarray) else np.abs(shap_values_user[0])
            sv_df = pd.DataFrame({"feature": features, "shap": sv}).sort_values("shap", ascending=False)
            fig = plt.figure(figsize=(6,6))
            sns.barplot(x="shap", y="feature", data=sv_df.head(15))
            save_fig(fig, os.path.join(rep_dir, "xgb_shap_bar.png"))

        # Also include the classifier score & probability
        prob = xgb_clf.predict_proba(x_user)[0,1]
        label = xgb_clf.predict(x_user)[0]
        with open(os.path.join(rep_dir, "regulator_summary.txt"), "w") as f:
            f.write(f"User_ID: {uid}\n")
            f.write(f"Suspicious probability: {prob:.4f}\n")
            f.write(f"Suspicious label: {label}\n")
            f.write("\nTop XGBoost features:\n")
            f.write(xgb_imp_df.head(10).to_string(index=False))
    # 4) Role-specific short textual summary
    summary_file = os.path.join(rep_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"User ID: {uid}\nRole: {role}\n\n")
        if role == 'member':
            f.write(f"Projected pension amount at retirement (point estimate): {pred_amount:,.2f}\n")
            f.write("Recommendations:\n")
            f.write("- Consider increasing monthly contributions to improve projected amount.\n")
            f.write("- Review high-fee funds; reducing fees improves net returns.\n")
            f.write("- Check diversification (portfolio diversity score) to manage volatility.\n")
        elif role == 'advisor':
            f.write(f"Projected pension amount (client): {pred_amount:,.2f}\n")
            f.write("Top local drivers (from TabNet):\n")
            f.write(local_imp.head(10).to_string(index=False))
            f.write("\nSuggested actions:\n- Consider rebalancing towards funds with better risk-adjusted returns.\n- Evaluate client's risk tolerance vs current allocation.\n- Discuss additional voluntary contributions or employer match optimization.\n")
        elif role == 'regulator':
            f.write("See regulator_summary.txt for suspicious flag details and model explanations.\n")
        f.write("\nGenerated using TabNet regressions + XGBoost classifier pipeline.\n")

    print("Report generated for", uid, "role:", role, "->", rep_dir)
    return rep_dir

# Example: generate 3 sample reports: one of each role for random users
sample_indices = df.sample(3, random_state=RANDOM_STATE).index.tolist()
roles = ['member', 'advisor', 'regulator']
for i, idx in enumerate(sample_indices):
    generate_user_report(idx, role=roles[i % len(roles)])

joblib.dump(regressor, "outputs/tabnet_reg.pkl")       # TabNet regressor
joblib.dump(xgb_clf, "outputs/xgb_clf.pkl")   

print("Done. Models + sample reports saved under", OUTPUT_DIR, "and", REPORTS_DIR)
